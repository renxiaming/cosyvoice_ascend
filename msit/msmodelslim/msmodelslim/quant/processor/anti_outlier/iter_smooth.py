#  -*- coding: utf-8 -*-
#  Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

from functools import partial
from typing import Dict, Callable, List, Any, Literal, Annotated
from pydantic import AfterValidator, Field

import torch
import torch.distributed as dist
import torch.nn as nn
from msmodelslim.core.QAL.qregistry import QABCRegistry
from msmodelslim.core.QAL.qtypes import (
    RMSNormBias,
    IterSmoothConfig
)
from msmodelslim.core.api import iter_smooth
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.quant.processor.anti_outlier.smooth_base import BaseSmoothProcessor, BaseSmoothProcessorConfig
from msmodelslim.quant.processor.anti_outlier.smooth_base import GraphOpt
from msmodelslim.quant.processor.anti_outlier.smooth_base import StatKey
from msmodelslim.quant.processor.base import AutoSessionProcessor
from msmodelslim.utils.dist import DistHelper
from msmodelslim.utils.validation.value import validate_normalized_value, is_boolean, is_string_list
from msmodelslim.utils.logging import get_logger


class IterSmoothProcessorConfig(BaseSmoothProcessorConfig):
    type: Literal["iter_smooth"] = "iter_smooth"
    alpha: Annotated[float, AfterValidator(validate_normalized_value)] = 0.9
    scale_min: Annotated[float, AfterValidator(validate_normalized_value)] = 1e-5
    symmetric: Annotated[bool, AfterValidator(is_boolean)] = False
    enable_subgraph_type: Annotated[List[str], AfterValidator(is_string_list)] = Field(
        default_factory=lambda: ["norm-linear", "linear-linear", "ov", "up-down"]
    )


@QABCRegistry.register(dispatch_key=IterSmoothProcessorConfig, abc_class=AutoSessionProcessor)
class IterSmoothProcessor(BaseSmoothProcessor):
    def __init__(self, model: nn.Module, config: IterSmoothProcessorConfig, adapter: object, **kwargs):
        super().__init__(model, config, adapter)
        self.config = config
        super().validate_parameters()
        self.act_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self.dist_helper = DistHelper(self.model) if dist.is_initialized() else None

        # 存储hook句柄，用于后续删除
        self.hook_handles = {}

    def support_distributed(self) -> bool:
        return True

    def preprocess(self, request: BatchProcessRequest) -> None:
        # 先调用父类的 preprocess 方法
        super().preprocess(request)
        # 遍历 subgraph_info，将 norm_module 替换为 RMSNormBias
        for subgraph in self.subgraph_info:
            if subgraph.subgraph_type == "norm-linear" and subgraph.metadata:
                # 获取 norm_module 信息
                norm_name = subgraph.metadata.get('source_name')
                norm_module = subgraph.metadata.get('source_module')

                if norm_name and norm_module is not None:
                    try:
                        # 检查 norm_module 是否有 weight 属性
                        if hasattr(norm_module, 'weight'):
                            # 创建 RMSNormBias 实例
                            norm_bias = RMSNormBias(norm_module.weight.shape[-1])
                            norm_bias.weight.data.copy_(norm_module.weight.data)
                            norm_bias.weight.data = norm_bias.weight.data.type(norm_module.weight.data.dtype)
                            if hasattr(norm_module, 'bias') and norm_module.bias is not None:
                                norm_bias.bias.data.copy_(norm_module.bias.data)
                                norm_bias.bias.data = norm_bias.bias.data.type(norm_module.weight.data.dtype)
                            norm_bias.to(norm_module.weight.data.device)
                            GraphOpt.set_module(self.model, norm_name, norm_bias)
                            subgraph.metadata['source_module'] = norm_bias
                            get_logger().debug(f"{norm_name}: {type(norm_module)} -> {type(norm_bias)}")
                        else:
                            get_logger().warning(f"Norm module {norm_name} does not have weight attribute")
                    except Exception as e:
                        get_logger().warning(f"Failed to replace norm module {norm_name}: {e}")

        get_logger().info(f"[Smooth] Processed {len(self.subgraph_info)} subgraphs for submodule {request.name}")

    def _get_stats_hook(self, name: str) -> Callable:
        def stats_hook(name: str, module: nn.Linear, args: tuple, kwargs: dict) -> None:
            tensor = args[0]

            if name not in self.act_stats:
                self.act_stats[name] = {}
                # 存储收集的tensor到CPU，避免OOM
                self.act_stats[name][StatKey.TENSOR] = tensor.cpu()

            hidden_dim = tensor.shape[-1]
            tensor = tensor.reshape(-1, hidden_dim).detach()  # [N,C]

            if self.dist_helper is not None and self.dist_helper.is_shared(name):
                tensor = torch.cat(self.dist_helper.gather_variable_shapes(tensor), dim=0)
            coming_max = torch.max(tensor, dim=0)[0]  # [C]
            coming_min = torch.min(tensor, dim=0)[0]  # [C]

            statis_dict = self.act_stats[name]

            # collect the min-max value
            if StatKey.STAT_KEY_MAX in statis_dict:
                statis_dict[StatKey.STAT_KEY_MAX] = torch.max(statis_dict[StatKey.STAT_KEY_MAX], coming_max)  # [C]
            else:
                statis_dict[StatKey.STAT_KEY_MAX] = coming_max

            if StatKey.STAT_KEY_MIN in statis_dict:
                statis_dict[StatKey.STAT_KEY_MIN] = torch.min(statis_dict[StatKey.STAT_KEY_MIN], coming_min)  # [C]
            else:
                statis_dict[StatKey.STAT_KEY_MIN] = coming_min

            # channel shifting
            if StatKey.STAT_KEY_SHIFT in statis_dict:
                statis_dict[StatKey.STAT_KEY_SHIFT] = (statis_dict[StatKey.STAT_KEY_MAX] + statis_dict[
                    StatKey.STAT_KEY_MIN]) / 2  # [C]
            else:
                statis_dict[StatKey.STAT_KEY_SHIFT] = (coming_max + coming_min) / 2

            channel_max = torch.max(tensor.abs().detach(), dim=0)[0]

            if StatKey.STAT_KEY_SMOOTH_SCALE in statis_dict:
                statis_dict[StatKey.STAT_KEY_SMOOTH_SCALE] = torch.max(statis_dict[StatKey.STAT_KEY_SMOOTH_SCALE],
                                                                       channel_max)
            else:
                statis_dict[StatKey.STAT_KEY_SMOOTH_SCALE] = channel_max

        return partial(stats_hook, name)


    def _apply_smooth_to_subgraph(self, subgraph_obj: Any, linear_modules: List[nn.Module]) -> None:
        """
        通用的平滑应用方法

        Args:
            subgraph_obj: 子图对象
            linear_modules: 线性模块列表
        """
        try:
            # 构建SmoothContext
            smooth_context = self._build_smooth_context(linear_modules)

            # 创建平滑配置
            smooth_quant_cfg = IterSmoothConfig(
                alpha=self.config.alpha,
                shift=self.config.symmetric,
                scale_min=self.config.scale_min
            )

            # 应用平滑
            iter_smooth(subgraph_obj, smooth_quant_cfg, smooth_context)
            get_logger().info("[IterSmoothProcessor] Smooth application completed successfully for subgraph")

        except Exception as e:
            get_logger().error(f"[IterSmoothProcessor] Failed to apply smooth to subgraph: {e}")
            raise
