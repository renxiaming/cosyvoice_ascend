# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import re
from pathlib import Path
from typing import List, Any

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase, PreTrainedModel, PretrainedConfig

from msmodelslim.app.base.const import DeviceType
from msmodelslim.app.base.model import BaseModelAdapter
from msmodelslim.core.graph.adapter_types import (
    SUPPORTED_SUBGRAPH_TYPES,
    SubgraphInfo
)
from msmodelslim.utils.exception import InvalidModelError, SchemaValidateError
from msmodelslim.utils.exception_decorator import exception_handler
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security.model import SafeGenerator
from .factory import ModelFactory


@ModelFactory.register("default")
@logger_setter()
class DefaultModelAdapter(BaseModelAdapter):
    @exception_handler('Using default model',
                       err_cls=Exception,
                       ms_err_cls=InvalidModelError,
                       action='Please check the model type')
    def __init__(self,
                 model_type: str,
                 ori_path: Path,
                 device: DeviceType = DeviceType.NPU,
                 trust_remote_code: bool = False):
        super().__init__(model_type, ori_path, device, trust_remote_code)

    def get_default_global_subgraph_info(self, module: nn.Module, dummy_input: Any) -> List[SubgraphInfo]:
        """获取所有全局子图信息，优先使用子类实现"""
        result = []
        for subgraph in SUPPORTED_SUBGRAPH_TYPES:
            if subgraph == 'norm-linear':
                result.extend(self._get_default_norm_linear_subgraph(module, dummy_input))
            elif subgraph == 'linear-linear':
                result.extend(self._get_default_linear_linear_subgraph(module, dummy_input))
            elif subgraph == 'ov':
                result.extend(self._get_default_o_v_subgraph(module, dummy_input))
            elif subgraph == 'up-down':
                result.extend(self._get_default_up_down_subgraph(module, dummy_input))
            else:
                raise ValueError(f"Unsupported subgraph type: {subgraph}")
        return result

    def _get_model_pedigree(self) -> str:
        model_type = re.match(r'^[a-zA-Z]+', self.type)
        if model_type is None:
            raise SchemaValidateError(f"Invalid model_name: {self.type}.",
                                      action='Please check the model name')
        return model_type.group().lower()

    def _get_default_norm_linear_subgraph(self,
                                          module: nn.Module,
                                          dummy_input: Any) -> List[SubgraphInfo]:
        """获取Norm->Linear子图（默认实现）"""

        subgraph_type = 'norm-linear'
        norm_class = [m.__class__ for m in module.modules() if "norm" in m.__class__.__name__.lower()]
        if dummy_input is None or norm_class is None:
            raise ValueError("dummy_input and norm_class must be provided if using default model adapter")

        from msmodelslim.utils.graph_utils import extract_dag

        dag = extract_dag(module,
                          dummy_input,
                          hook_nodes=norm_class)

        norm_linear_subgraph = dag.get_norm_linear_subgraph()

        result = []
        for norm_node, linear_list in norm_linear_subgraph.items():
            # 获取norm模块的实际对象
            norm_module = None
            try:
                norm_module = module.get_submodule(norm_node)
            except (AttributeError, KeyError, ValueError):
                get_logger().warning(f"Cannot find norm module: {norm_node}")

            # 获取linear模块的实际对象列表
            linear_modules = []
            linear_names = []
            for linear_name in linear_list:
                try:
                    linear_module = module.get_submodule(linear_name)
                    linear_modules.append(linear_module)
                    linear_names.append(linear_name)
                except (AttributeError, KeyError, ValueError):
                    get_logger().warning(f"Cannot find linear module: {linear_name}")

            # 构建SubgraphInfo对象
            subgraph_info = SubgraphInfo(
                name=f"{norm_node}_{subgraph_type}_subgraph",
                subgraph_type=subgraph_type,
                metadata={
                    'source_name': norm_node,
                    'source_module': norm_module,
                    'target_names': linear_names,
                    'target_modules': linear_modules,
                }
            )
            result.append(subgraph_info)

        del dag
        return result

    def _get_default_linear_linear_subgraph(self,
                                            module: nn.Module,
                                            dummy_input: Any, subgraph_type: str = 'linear-linear') -> List[
        SubgraphInfo]:
        """获取Linear->Linear子图（默认实现）"""
        norm_class = [m.__class__ for m in module.modules() if "norm" in m.__class__.__name__.lower()]
        if dummy_input is None or norm_class is None:
            raise ValueError("dummy_input and norm_class must be provided if using default model adapter")

        from msmodelslim.utils.graph_utils import extract_dag

        dag = extract_dag(module,
                          dummy_input,
                          hook_nodes=norm_class)

        linear_linear_subgraph = dag.get_linear_linear_subgraph()

        result = []
        for linear1_name, linear2_list in linear_linear_subgraph.items():
            # 获取linear1模块的实际对象
            linear1_module = None
            try:
                linear1_module = module.get_submodule(linear1_name)
            except (AttributeError, KeyError, ValueError):
                get_logger().warning(f"Cannot find linear1 module: {linear1_name}")

            # 获取linear2模块的实际对象
            linear2_module = None
            linear2_name = ""
            if linear2_list:
                linear2_name = linear2_list[0]  # 取第一个linear2模块
                try:
                    linear2_module = module.get_submodule(linear2_name)
                except (AttributeError, KeyError, ValueError):
                    get_logger().warning(f"Cannot find linear2 module: {linear2_name}")

            # 构建SubgraphInfo对象
            subgraph_info = SubgraphInfo(
                name=f"{linear1_name}_{subgraph_type}_subgraph",
                subgraph_type=subgraph_type,
                metadata={
                    'source_name': linear1_name,
                    'source_module': linear1_module,
                    'target_names': linear2_name,
                    'target_modules': linear2_module,
                }
            )
            result.append(subgraph_info)

        del dag
        return result

    def _get_default_o_v_subgraph(self,
                                  module: nn.Module,
                                  dummy_input: Any, subgraph_type: str = 'ov') -> List[SubgraphInfo]:
        """获取OV子图（默认实现）"""
        return self._get_default_linear_linear_subgraph(module, dummy_input, subgraph_type='ov')

    def _get_default_up_down_subgraph(self,
                                      module: nn.Module,
                                      dummy_input: Any) -> List[SubgraphInfo]:
        """获取Up-Down子图（默认实现）"""
        return self._get_default_linear_linear_subgraph(module, dummy_input, subgraph_type='up-down')

    def _load_config(self) -> PretrainedConfig:
        return SafeGenerator.get_config_from_pretrained(model_path=str(self.ori))

    def _load_tokenizer(self, trust_remote_code=False) -> PreTrainedTokenizerBase:
        return SafeGenerator.get_tokenizer_from_pretrained(
            model_path=str(self.ori),
            use_fast=False,
            legacy=False,
            trust_remote_code=trust_remote_code)

    def _load_model(self, device_map=None, torch_dtype=None) -> PreTrainedModel:
        device_map = device_map if device_map is not None else self._device_map
        dtype = torch_dtype if torch_dtype is not None else self._torch_dtype

        return SafeGenerator.get_model_from_pretrained(
            model_path=str(self.ori),
            device_map=device_map,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=self._trust_remote_code)

    def _load_hook(self) -> None:
        pass

    def _persist_hook(self) -> None:
        pass
