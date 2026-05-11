#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import functools
import inspect
import os
from typing import Dict, Any, Optional, List, Literal

import torch
import torch.distributed as dist
from torch import nn

import msmodelslim.quant.ir as qir
from msmodelslim import logger
from msmodelslim.core.QAL.qregistry import QABCRegistry
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.quant.processor.base import AutoSessionProcessor
from msmodelslim.utils.dist import DistHelper
from .saver import AutoSaverProcessor, AutoSaverBaseConfig
from .utils.json import JsonWriter
from .utils.safetensors import SafetensorsWriter, BufferedSafetensorsWriter
from .utils.pack import w4a8_pack_int4


class AscendV1Config(AutoSaverBaseConfig):
    """
    ascendV1 量化模型保存器配置。该配置用于配置ascendV1量化模型保存器。
    
    该配置包含以下字段：
        - type: 量化模型保存器类型，固定为"ascendv1_saver"
        - save_directory: 量化模型保存目录，默认为"."
        - part_file_size: 量化模型权重文件大小，默认为4，单位为GB，若part_file_size为0，则不进行分文件保存
        - ext: 扩展配置，用于配置量化模型保存器的扩展功能

    Notes:

    在ascendV1格式中，标准导出件包括：
        - 量化模型描述文件，使用json格式描述量化模型
        - 量化模型权重文件，使用safetensors格式保存量化模型权重（可能包含多个）
        - safetensors index文件，使用json格式保存safetensors文件的索引（可选）

    量化模型描述文件中，包含以下字段：
        - 量化模型权重键值对，键为权重名称，值为权重所属的量化类型
            example:
            {
                "model.layers.0.self_attn.q_proj.weight": "W8A8",
                "model.layers.0.self_attn.q_proj.input_scale": "W8A8",
                "model.layers.0.self_attn.q_proj.input_offset": "W8A8",
                "model.layers.0.self_attn.q_proj.deq_scale": "W8A8",
                "model.layers.0.self_attn.q_proj.quant_bias": "W8A8",
            }

            该例子中表明了model.layers.0.self_attn.q_proj所代表的nn.Linear被量化为W8A8类型，
            该类型由5个参数完全描述。

    量化模型权重文件以safetensors格式保存权重参数。考虑到文件大小的限制，可能会存在多个权重文件，
    此时则会生成safetensors index文件，用于记录各个权重所处的safetensors文件。

    """

    def set_save_directory(self, save_directory: str):
        self.save_directory = str(save_directory)

    type: Literal['ascendv1_saver'] = "ascendv1_saver"
    save_directory: str = "."
    part_file_size: int = 4
    ext: Dict[str, Any] = {}


ASCENDV1_DESC_JSON_NAME = "quant_model_description.json"
ASCENDV1_SAFETENSORS_NAME = "quant_model_weights.safetensors"


def save_this_rank_only():
    """

    该函数用于装饰on_xxx系列方法，用于在分布式模式下，过滤掉不应属于当前rank的保存动作。

    example:

    @save_this_rank_only
    def on_w8a8_static(self, prefix: str, module: qir.W8A8StaticFakeQuantLinear):
        pass

    该装饰器会自动入参判断是否属于当前rank，若不属于当前rank，则不会调用被装饰的函数。

    """

    def decorator(func):
        # check function signature
        if inspect.signature(func).parameters.keys() != {'self', 'prefix', 'module'}:
            raise ValueError(
                f"Function {func.__name__} has incorrect signature that cannot be decorated by save_this_rank_only")

        @functools.wraps(func)
        def wrapper(self_instance: 'AscendV1Saver', prefix: str, module: nn.Module) -> None:
            if not dist.is_initialized():
                func(self_instance, prefix, module)
                return

            is_local_only = self_instance.dist_helper.is_local_only(prefix)
            is_in_shared_modules_slice = prefix in self_instance.shared_modules_slice
            save_on_this_rank = is_local_only or is_in_shared_modules_slice

            if not save_on_this_rank:
                return

            func(self_instance, prefix, module)
            return

        return wrapper

    return decorator


@QABCRegistry.register(dispatch_key=AscendV1Config, abc_class=AutoSessionProcessor)
class AscendV1Saver(AutoSaverProcessor):
    """
    ascendV1 量化模型保存器。该保存器将量化模型保存为AscendV1格式。
    
    关于该格式的更多信息，请参考 AscendV1Config 中的说明。
    """

    def __init__(self, model: nn.Module, config: AscendV1Config, adapter: object, **kwargs: Dict[str, Any]):
        super().__init__(model, config, adapter, **kwargs)
        self.config = config
        self.version = "1.0.0"
        self.model_quant_type = "Unknown"
        self.json_append = dict()
        self.save_directory = self.get_rank_save_directory() if dist.is_initialized() else config.save_directory
        self.json_writer = JsonWriter(config.save_directory, ASCENDV1_DESC_JSON_NAME)
        self.safetensors_writer = self.get_safetensors_writer(config)
        self.dist_helper: Optional[DistHelper] = None
        self.shared_modules_slice: Optional[List[str]] = None

    def support_distributed(self) -> bool:
        return True

    def post_run(self) -> None:

        super().post_run()

        for key, val in self.json_append.items():
            self.json_writer.write(key, val)

        self.json_writer.write("version", self.version)
        self.json_writer.write("model_quant_type", self.model_quant_type)
        self.json_writer.close()
        self.safetensors_writer.close()

    def preprocess(self, request: BatchProcessRequest) -> None:
        if dist.is_initialized():
            self.prepare_for_distributed(request)

    def postprocess(self, request: BatchProcessRequest) -> None:
        super().postprocess(request)
        self.cleanup_for_distributed()

    def prepare_for_distributed(self, request: BatchProcessRequest) -> None:
        self.dist_helper = DistHelper(request.module)
        self.shared_modules_slice = self.dist_helper.get_shared_modules_slice(prefix=request.name)

    def cleanup_for_distributed(self) -> None:
        self.dist_helper = None

    def get_safetensors_writer(self, config: AscendV1Config) -> SafetensorsWriter:
        if config.part_file_size > 0:
            return BufferedSafetensorsWriter(
                logger=logger,
                max_gb_size=config.part_file_size,
                save_directory=self.save_directory,
                save_prefix=ASCENDV1_SAFETENSORS_NAME.removesuffix('.safetensors')
            )
        else:
            return SafetensorsWriter(
                logger=logger,
                file_path=os.path.join(self.save_directory, ASCENDV1_SAFETENSORS_NAME),
            )

    def get_rank_save_directory(self) -> str:
        return os.path.join(self.config.save_directory, f"rank_{dist.get_rank()}")

    def write_tensor(self, prefix: str, desc: str, tensor: torch.Tensor):
        self.json_writer.write(prefix, desc)
        self.safetensors_writer.write(prefix, tensor)

    def merge_ranks(self) -> None:
        if dist.get_rank() != 0:
            return
        raise NotImplementedError("merge_ranks for ascendV1 format is not implemented now")

    @save_this_rank_only()
    def on_w8a8_static(self, prefix: str, module: qir.W8A8StaticFakeQuantLinear):
        with torch.device(module.weight.device):
            input_scale, input_offset = module.input_scale, module.input_offset
            input_scale = input_scale.unsqueeze(0) if input_scale.ndim == 0 else input_scale
            input_offset = input_offset.unsqueeze(0) if input_offset.ndim == 0 else input_offset
            weight_scale = module.weight_scale
            quant_weight = module.weight
            deq_scale = input_scale * weight_scale
            deq_scale = deq_scale.squeeze(1) if deq_scale.ndim > 1 else deq_scale
            fp_weight_bias = module.bias if module.bias is not None else torch.zeros(module.weight.shape[0])
            correction = quant_weight.to(torch.float32).sum(dim=1) * input_offset.to(torch.float32)
            quant_bias = torch.round(fp_weight_bias / deq_scale - correction).to(torch.int32)
            self.write_tensor(prefix + ".weight", "W8A8", quant_weight.to(torch.int8))
            self.write_tensor(prefix + ".quant_bias", "W8A8", quant_bias.to(torch.int32))
            self.write_tensor(prefix + ".input_scale", "W8A8", input_scale.to(torch.float32))
            self.write_tensor(prefix + ".input_offset", "W8A8", input_offset.to(torch.float32))
            self.write_tensor(prefix + ".deq_scale", "W8A8", deq_scale.to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "W8A8", module.bias.to(torch.float32))
            self.model_quant_type = "W8A8"

    @save_this_rank_only()
    def on_w8a8_dynamic(self, prefix: str, module: qir.W8A8DynamicFakeQuantLinear):
        with torch.device(module.weight.device):
            weight_scale = module.weight_scale.unsqueeze(-1)
            weight_offset = module.weight_offset.unsqueeze(-1)
            self.write_tensor(prefix + ".weight", "W8A8_DYNAMIC", module.weight.to(torch.int8))
            self.write_tensor(prefix + ".weight_scale", "W8A8_DYNAMIC", weight_scale.to(torch.float32))
            self.write_tensor(prefix + ".weight_offset", "W8A8_DYNAMIC", weight_offset.to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "W8A8_DYNAMIC", module.bias.to(torch.float32))
            self.model_quant_type = "W8A8_DYNAMIC"

    @save_this_rank_only()
    def on_w4a8_dynamic(self, prefix: str, module: qir.W4A8DynamicFakeQuantLinear):
        with torch.device(module.weight.device):
            weight_scale = module.weight_scale.unsqueeze(-1)
            weight_offset = module.weight_offset.unsqueeze(-1)
            self.write_tensor(prefix + ".weight", "W4A8_DYNAMIC", w4a8_pack_int4(module.weight.to(torch.int8)))
            self.write_tensor(prefix + ".weight_scale", "W4A8_DYNAMIC", weight_scale.to(torch.float32))
            self.write_tensor(prefix + ".weight_offset", "W4A8_DYNAMIC", weight_offset.to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "W4A8_DYNAMIC", module.bias.to(torch.float32))
            self.model_quant_type = "W4A8_DYNAMIC"

    @save_this_rank_only()
    def on_float_linear(self, prefix: str, module: nn.Linear):
        return self.on_float_module(prefix, module)

    @save_this_rank_only()
    def on_float_module(self, prefix: str, module: nn.Module):
        for name, param in module.named_parameters(recurse=False, prefix=prefix):
            self.write_tensor(name, "FLOAT", param)

    @save_this_rank_only()
    def on_dynamic_cache(self, prefix: str, module: qir.FakeQuantDynamicCache):
        prefix_list = prefix.split(".")
        prefix_no_last = '.'.join(prefix_list[:-1])
        if "key_states" in prefix_list[-1]:
            self.write_tensor(prefix_no_last + ".k_proj.kv_cache_scale", "C8", module.kv_cache_scale)
            self.write_tensor(prefix_no_last + ".k_proj.kv_cache_offset", "C8", module.kv_cache_offset)
        elif "value_states" in prefix_list[-1]:
            self.write_tensor(prefix_no_last + ".v_proj.kv_cache_scale", "C8", module.kv_cache_scale)
            self.write_tensor(prefix_no_last + ".v_proj.kv_cache_offset", "C8", module.kv_cache_offset)
        else:
            raise ValueError(f"Unknown dynamic cache prefix: {prefix}")
        self.json_append['kv_cache_type'] = "C8"
        self.json_append['kv_quant_type'] = "C8"

    @save_this_rank_only()
    def on_w16a16s(self, prefix: str, module: qir.W16A16sLinear):
        for name, param in module.named_parameters(recurse=False, prefix=prefix):
            self.write_tensor(name, "W16A16S", param)
