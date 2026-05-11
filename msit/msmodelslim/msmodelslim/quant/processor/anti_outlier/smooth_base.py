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

from enum import Enum
from typing import List, Optional, Dict, Any, Callable, Annotated, Literal
from pydantic import Field, model_validator, AfterValidator

import torch
from torch import nn

from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.core.QAL.qtypes import (
    LinearLinearSubgraph,
    NormLinearSubgraph,
    SmoothContext,
    UpDownSubgraph,
    OVSubgraph
)
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.quant.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig, FusionConfig, SubgraphInfo
from msmodelslim.utils.exception import MisbehaviorError, SchemaValidateError
from msmodelslim.utils.logging import get_logger


class GraphOpt:
    @staticmethod
    def set_module(model,
                   submodule_key,
                   module):
        tokens = submodule_key.split('.')
        sub_tokens = tokens[:-1]
        cur_mod = model
        for s in sub_tokens:
            cur_mod = getattr(cur_mod, s)
        setattr(cur_mod, tokens[-1], module)


class StatKey(str, Enum):
    STAT_KEY_MAX = "max"
    STAT_KEY_MIN = "min"
    STAT_KEY_SHIFT = "shift"
    STAT_KEY_THRESHOLD_CHANNEL = "thres_c"
    STAT_KEY_THRESHOLD_TENSOR = "thres_t"
    STAT_KEY_SMOOTH_SCALE_MASK = "smooth_scale_mask"
    STAT_KEY_SMOOTH_SCALE = "smooth_scale"
    STAT_KEY_VARIANCE = "std"
    TENSOR = 'tensor'


class VirtualVModule(nn.Module):
    """虚拟 V 模块，用于处理 QKV 融合的情况，支持 MHA、MQA 和 GQA 三种结构"""

    def __init__(self, qkv_module: nn.Linear, num_attention_heads: int, num_key_value_heads: int):
        super().__init__()
        self.qkv_module = qkv_module
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        # 确定注意力机制类型
        self.attention_type = self._determine_attention_type()

        # 计算 V 部分的权重和偏置
        self._extract_v_weights()

    def forward(self, x):
        """前向传播，只返回 V 部分的输出"""
        # 计算 V 部分的输出
        v_output = torch.nn.functional.linear(x, self.v_weight, self.v_bias)
        return v_output

    def update_qkv_weights(self):
        """将更新后的 V 权重更新回原始的 QKV 模块"""
        qkv_weight = self.qkv_module.weight
        qkv_bias = getattr(self.qkv_module, 'bias', None)

        # 计算每个头的维度
        head_dim = qkv_weight.shape[1] // self.num_attention_heads

        # 根据注意力类型计算 V 部分的起始和结束索引
        v_start, v_end = self._get_v_indices(head_dim)

        # 更新 V 部分的权重
        with torch.no_grad():
            qkv_weight[v_start:v_end] = self.v_weight.data

            # 更新 V 部分的偏置
            if qkv_bias is not None and self.v_bias is not None:
                qkv_bias[v_start:v_end] = self.v_bias.data

    def _determine_attention_type(self) -> str:
        """确定注意力机制类型"""
        if self.num_key_value_heads == 1:
            return "MQA"  # Multi-Query Attention
        elif self.num_key_value_heads == self.num_attention_heads:
            return "MHA"  # Multi-Head Attention
        elif (self.num_key_value_heads < self.num_attention_heads and 
              self.num_attention_heads % self.num_key_value_heads == 0):
            return "GQA"  # Grouped-Query Attention
        else:
            get_logger().warning("InValid attention type, please check.")
            return "UNKNOWN"

    def _get_v_indices(self, head_dim: int) -> tuple:
        """根据注意力类型获取 V 部分的索引范围"""
        if self.attention_type == "MHA":
            # MHA: QKV 顺序为 [Q, K, V]，每个都有 num_attention_heads 个头
            q_size = self.num_attention_heads * head_dim
            k_size = self.num_attention_heads * head_dim
            v_start = q_size + k_size
            v_end = q_size + k_size + self.num_attention_heads * head_dim

        elif self.attention_type == "MQA":
            # MQA: QKV 顺序为 [Q, K, V]，Q 有 num_attention_heads 个头，K/V 只有 1 个头
            q_size = self.num_attention_heads * head_dim
            k_size = 1 * head_dim
            v_start = q_size + k_size
            v_end = q_size + k_size + 1 * head_dim

        elif self.attention_type == "GQA":  # GQA
            # GQA: QKV 顺序为 [Q, K, V]，Q 有 num_attention_heads 个头，K/V 有 num_key_value_heads 个头
            q_size = self.num_attention_heads * head_dim
            k_size = self.num_key_value_heads * head_dim
            v_start = q_size + k_size
            v_end = q_size + k_size + self.num_key_value_heads * head_dim
        else:
            raise ValueError(f"Invalid attention type: {self.attention_type}")
        return v_start, v_end

    def _extract_v_weights(self):
        """从 QKV 模块中提取 V 部分的权重和偏置"""
        qkv_weight = self.qkv_module.weight
        qkv_bias = getattr(self.qkv_module, 'bias', None)

        # 计算每个头的维度
        head_dim = qkv_weight.shape[1] // self.num_attention_heads

        # 根据注意力类型获取 V 部分的索引范围
        v_start, v_end = self._get_v_indices(head_dim)

        # 提取 V 部分的权重
        self.v_weight = nn.Parameter(qkv_weight[v_start:v_end].clone())

        # 提取 V 部分的偏置
        if qkv_bias is not None:
            self.v_bias = nn.Parameter(qkv_bias[v_start:v_end].clone())
        else:
            self.v_bias = None


class SubgraphProcessor:
    """
    通用子图适配器，支持通过语义化pattern和scope限定作用域自动查找子图。
    """

    def __init__(
            self,
            model: nn.Module,
            adapter_config: Optional[List[AdapterConfig]] = None,
            smooth_config: Optional[AutoProcessorConfig] = None
    ):
        self.adapter_config = adapter_config
        self.smooth_config = smooth_config
        self.model = model

    def find_subgraphs_by_config(
            self,
            subgraphs: List[SubgraphInfo],
            config: AutoProcessorConfig,
            scope: str,
            **kwargs
    ) -> List[SubgraphInfo]:
        """
        根据配置过滤子图

        Args:
            subgraphs: 子图信息列表
            config: 处理器配置
            scope: 作用域前缀
            **kwargs: 其他参数

        Returns:
            List[SubgraphInfo]: 过滤后的子图信息列表
        """

        result = []
        layer_prefix = f"{scope}." if scope != "" else ""

        include = ConfigSet(self.smooth_config.include) if self.smooth_config.include else ConfigSet(["*"])
        exclude = ConfigSet(self.smooth_config.exclude) if self.smooth_config.exclude else ConfigSet([])
        for subgraph in subgraphs:
            # 1. 检查子图类型是否支持
            if subgraph.subgraph_type not in config.enable_subgraph_type:
                continue

            # 2. 检查元数据是否存在
            if not subgraph.metadata or 'source_name' not in subgraph.metadata:
                continue

            source_name = subgraph.metadata['source_name']

            # 3. 检查是否以layer_prefix为前缀
            if not source_name.startswith(layer_prefix):
                continue

            if source_name not in include:
                continue

            if source_name in exclude:
                continue

            result.append(subgraph)

        return result

    def get_global_subgraph_info(self) -> List[SubgraphInfo]:
        """
        获取全局子图信息，优先使用子类实现

        Returns:
            List[SubgraphInfo]: 子图信息列表
        """
        result = []
        if not self.adapter_config:
            raise MisbehaviorError("adapter_config cannot be empty")
        result = self._get_adapter_based_subgraph_info()
        return result

    def _get_adapter_based_subgraph_info(self) -> List[SubgraphInfo]:
        """
        基于adapter_config获取子图信息

        Returns:
            List[SubgraphInfo]: 子图信息列表
        """
        result = []

        # 子图类型到创建方法的映射
        subgraph_creators = {
            "norm-linear": lambda m: self._create_norm_linear_subgraph_info(m),
            "ov": lambda m, fusion=None: self._create_ov_subgraph_info(m, fusion),
            "up-down": lambda m: self._create_up_down_subgraph_info(m),
            "linear-linear": lambda m: self._create_linear_linear_subgraph_info(m),
        }

        for adapter_config in self.adapter_config:
            subgraph_type = adapter_config.subgraph_type
            mapping = adapter_config.mapping

            if mapping is None or subgraph_type not in subgraph_creators:
                continue

            # 使用映射的方法创建子图信息
            if subgraph_type == "ov":
                fusion = getattr(adapter_config, 'fusion', None)
                subgraph_info = subgraph_creators[subgraph_type](mapping, fusion)
            else:
                subgraph_info = subgraph_creators[subgraph_type](mapping)

            if subgraph_info:
                result.append(subgraph_info)

        return result

    def _create_subgraph_info_base(
            self,
            mapping: 'MappingConfig',
            subgraph_type: str,
            metadata_extra: Optional[Dict[str, Any]] = None
    ) -> Optional['SubgraphInfo']:
        """
        创建子图信息的基础方法

        Args:
            mapping: 映射配置
            subgraph_type: 子图类型
            metadata_extra: 额外的元数据

        Returns:
            Optional[SubgraphInfo]: 子图信息，如果失败则返回None
        """
        try:
            source_name = mapping.source
            target_names = mapping.targets

            # 获取源模块
            source_module = None
            for name, module in self.model.named_modules():
                if name == source_name:
                    source_module = module
                    break

            if source_module is None:
                get_logger().warning(f"Cannot find source module: {source_name}")
                return None

            # 获取目标模块列表
            target_modules = []
            for target_name in target_names:
                target_module = None
                for name, module in self.model.named_modules():
                    if name == target_name:
                        target_module = module
                        break

                if target_module is not None:
                    target_modules.append(target_module)
                else:
                    get_logger().warning(f"Cannot find target module: {target_name}")

            if not target_modules:
                get_logger().warning(f"No valid target modules found for source: {source_name}")
                return None

            # 构建基础元数据
            metadata = {
                'source_name': source_name,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                'source_module': source_module,
                'target_names': target_names,
                'target_modules': target_modules,
            }

            # 添加额外的元数据
            if metadata_extra:
                metadata.update(metadata_extra)

            return SubgraphInfo(
                name=f"{source_name}_{subgraph_type}_subgraph",
                subgraph_type=subgraph_type,
                metadata=metadata
            )

        except Exception as e:
            get_logger().warning(f"Failed to create {subgraph_type} subgraph info: {e}")
            return None

    def _create_norm_linear_subgraph_info(
            self,
            mapping: 'MappingConfig',
    ) -> Optional['SubgraphInfo']:
        """创建norm-linear子图信息"""
        subgraph_info = self._create_subgraph_info_base(mapping, "norm-linear")
        if subgraph_info is None:
            raise MisbehaviorError(
                "Failed to create norm-linear subgraph info. "
                "Please check if the required modules exist in the model "
                "and the mapping configuration is correct."
            )
        return subgraph_info

    def _create_ov_subgraph_info(
            self,
            mapping: 'MappingConfig',
            fusion: Optional['FusionConfig'] = None
    ) -> Optional['SubgraphInfo']:
        """创建ov子图信息"""
        # 准备额外的元数据
        metadata_extra = None
        if fusion:
            metadata_extra = {
                'fusion_type': fusion.fusion_type,
                'num_attention_heads': fusion.num_attention_heads,
                'num_key_value_heads': fusion.num_key_value_heads
            }

        subgraph_info = self._create_subgraph_info_base(mapping, "ov", metadata_extra)
        if subgraph_info is None:
            raise MisbehaviorError(
                "Failed to create ov subgraph info. "
                "Please check if the required modules exist in the model "
                "and the mapping configuration is correct."
            )
        return subgraph_info

    def _create_up_down_subgraph_info(
            self,
            mapping: 'MappingConfig'
    ) -> Optional['SubgraphInfo']:
        """创建up-down子图信息"""
        subgraph_info = self._create_subgraph_info_base(mapping, "up-down")
        if subgraph_info is None:
            raise MisbehaviorError(
                "Failed to create up-down subgraph info. "
                "Please check if the required modules exist in the model "
                "and the mapping configuration is correct."
            )
        return subgraph_info

    def _create_linear_linear_subgraph_info(
            self,
            mapping: 'MappingConfig'
    ) -> Optional['SubgraphInfo']:
        """创建linear-linear子图信息"""
        subgraph_info = self._create_subgraph_info_base(mapping, "linear-linear")
        if subgraph_info is None:
            raise MisbehaviorError(
                "Failed to create linear-linear subgraph info. "
                "Please check if the required modules exist in the model "
                "and the mapping configuration is correct."
            )
        return subgraph_info


class BaseSmoothProcessorConfig(AutoProcessorConfig):
    type: Literal["base_smooth"] = "base_smooth" 
    alpha: Optional[float] = None      # 可选参数
    beta: Optional[float] = None  
    scale_min: Optional[float] = None
    symmetric: Optional[bool] = None
    enable_subgraph_type: Optional[List[str]] = None
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None
    # 子图处理优先级配置（数字越小优先级越高）
    subgraph_priority: Dict[str, int] = Field(
        default_factory=lambda: {"up-down": 1, "ov": 2, "norm-linear": 3, "linear-linear": 4},
        frozen=True  # 单独为subgraph_priority字段启用frozen
    )

    @model_validator(mode='before')
    @classmethod
    def validate_no_fixed_overrides(cls, data: Any):
        if isinstance(data, dict):
            fixed_fields = {'subgraph_priority'}
            attempted_overrides = fixed_fields.intersection(data.keys())
            if attempted_overrides:
                get_logger().warning(
                    f"Fields {attempted_overrides} have fixed values and cannot be overridden. "
                    f"Using default subgraph_priority configuration: "
                    f"up-down(1) -> ov(2) -> norm-linear(3) -> linear-linear(4)"
                )
                # 移除尝试覆盖的字段，使用默认值
                for override_field in attempted_overrides:
                    data.pop(override_field, None)
        return data


class BaseSmoothProcessor(AutoSessionProcessor):

    def __init__(self, model: nn.Module, config: AutoProcessorConfig, adapter: Optional[Any] = None):
        super().__init__(model)
        self.adapter = adapter
        self.config = config
        self.act_stats = {}

    def validate_parameters(self):
        """
        验证所有参数的合法性
        """
        # 验证 enable_subgraph_type 中的元素
        valid_subgraph_types = ["norm-linear", "linear-linear", "ov", "up-down"]
        for subgraph_type in self.config.enable_subgraph_type:
            if subgraph_type not in valid_subgraph_types:
                raise SchemaValidateError(
                    f"Elements in enable_subgraph_type must be in {valid_subgraph_types}, "
                    f"current element: {subgraph_type}",
                    action=f"Please use only valid subgraph types: {valid_subgraph_types}")

    def get_num_attention_heads(self):
        num_attention_heads = None
        key_attention_heads = ["num_attention_heads", "n_head", "num_heads", "heads_num"]
        for key in key_attention_heads:
            if hasattr(self.model.config, key):
                num_attention_heads = getattr(self.model.config, key)
        if not num_attention_heads:
            raise MisbehaviorError(
                f"the config of model must have num_attention_heads, n_head or num_heads, \
                                please check or modify the config file"
            )
        return num_attention_heads

    def get_num_key_value_heads(self):
        # 如果不存在num_key_value_heads，则设置为num_attention_heads的值
        if not hasattr(self.model.config, "num_key_value_heads"):
            num_key_value_heads = self.model.config.num_attention_heads
            get_logger().warning("Failed to obtain `num_key_value_heads`, assuming Multi-head Attention by default.")
        else:
            num_key_value_heads = self.model.config.num_key_value_heads

        if not num_key_value_heads:
            raise MisbehaviorError(
                f"the config of model must have num_key_value_heads, \
                                please check or modify the config file"
            )
        return num_key_value_heads

    def is_data_free(self) -> bool:
        _ = self
        return False

    def preprocess(self, request: BatchProcessRequest) -> None:
        adapter_config = self.adapter.get_adapter_config_for_subgraph()
        # 创建子图处理器
        smooth_processor = SubgraphProcessor(model=self.model, adapter_config=adapter_config, smooth_config=self.config)
        # 获取全量子图
        self.subgraph_info = smooth_processor.get_global_subgraph_info()
        # 根据配置过滤子图
        self.subgraph_info = smooth_processor.find_subgraphs_by_config(
            self.subgraph_info,
            self.config,
            request.name
        )
        get_logger().info(f"[Smooth] Processed {len(self.subgraph_info)} subgraphs for submodule {request.name}")
        self._install_statis_hook(request.name, request.module)

    def postprocess(self, request: BatchProcessRequest) -> None:
        self._apply_smooth(request.name, request.module)

    def _build_smooth_context(self, linear_modules: List[nn.Linear]) -> SmoothContext:
        """
        构建 SmoothContext
        
        Args:
            linear_modules: 线性模块列表，用于获取激活统计信息
            
        Returns:
            SmoothContext: 平滑上下文
        """

        for linear_module in linear_modules:
            # 获取模块名称
            module_name = None
            for name, module in self.model.named_modules():
                if module is linear_module:
                    module_name = name
                    break

            if module_name is None:
                get_logger().warning(f"Cannot find module name for {linear_module}")
                continue

            # 获取激活统计信息
            if module_name in self.act_stats:
                stats = self.act_stats[module_name]

                # 获取 smooth_scale
                if StatKey.STAT_KEY_SMOOTH_SCALE in stats:
                    a_smooth_scale = stats[StatKey.STAT_KEY_SMOOTH_SCALE]
                else:
                    raise MisbehaviorError(f"smooth_scale is not found for module {module_name}")

                # 获取 shift
                if StatKey.STAT_KEY_SHIFT in stats:
                    shift = stats[StatKey.STAT_KEY_SHIFT]
                else:
                    shift = None
                
                if StatKey.TENSOR in stats:
                    tensors = stats[StatKey.TENSOR]
                else:
                    tensors = None

        w_smooth_scale = torch.ones_like(a_smooth_scale)

        # 创建扩展信息
        ext = {}

        # 创建 SmoothContext
        smooth_context = SmoothContext(
            version=1,
            a_smooth_scale=a_smooth_scale,
            w_smooth_scale=w_smooth_scale,
            tensors=tensors,
            shift=shift,
            ext=ext
        )

        return smooth_context

    def _get_stats_hook(self, name: str) -> Callable:
        """
        获取统计钩子的抽象方法，子类必须实现
        
        Args:
            name: 模块名称
            
        Returns:
            Callable: 统计钩子函数
        """
        raise NotImplementedError("Subclasses must implement _get_stats_hook method")
    
    def _get_target_names_for_hook(self, subgraph: SubgraphInfo) -> List[str]:
        """
        根据子图类型获取需要安装钩子的模块名称列表
        
        Args:
            subgraph: 子图信息
            
        Returns:
            List[str]: 目标模块名称列表
        """
        subgraph_type = subgraph.subgraph_type
        metadata = subgraph.metadata

        if subgraph_type == "norm-linear" or subgraph_type == "up-down":
            # Norm-Linear: 为所有target_names安装钩子
            return metadata.get('target_names', [])

        if subgraph_type == "linear-linear" or subgraph_type == "ov":
            # Linear-Linear: 为linear2模块安装钩子
            target_name = metadata.get('target_names', [''])[0]
            return [target_name] if target_name else []

        # 默认情况：返回空列表
        return []

    def _install_hook_for_module(self, module_name: str) -> None:
        """
        为指定模块安装统计钩子
        
        Args:
            module_name: 模块名称
        """
        try:
            module = self.model.get_submodule(module_name)
            if isinstance(module, nn.Linear):
                # 保存hook句柄，用于后续删除
                hook_handle = module.register_forward_hook(self._get_stats_hook(module_name))
                self.hook_handles[module_name] = hook_handle
                get_logger().debug(f"Successfully installed statistics hook for module {module_name}")
            else:
                get_logger().warning(f"Module {module_name} is not Linear type, skipping hook installation")
        except Exception as e:
            get_logger().warning(f"Failed to install statistics hook for module {module_name}: {e}")
    


    def _install_statis_hook(self, name: str, module: nn.Module) -> None:
        """
        为所有子图中的linear模块安装统计钩子
        
        Args:
            name: 模块名称
            module: 目标模块
        """
        for subgraph in self.subgraph_info:
            if not subgraph.metadata:
                continue

            # 根据子图类型获取需要安装钩子的模块名称
            target_names = self._get_target_names_for_hook(subgraph)

            # 为每个目标模块安装钩子
            for target_name in target_names:
                if target_name:
                    self._install_hook_for_module(target_name)

    def _remove_all_hooks(self) -> None:
        """
        删除所有已安装的hook
        """
        for module_name, hook_handle in self.hook_handles.items():
            try:
                hook_handle.remove()
                get_logger().debug(f"Successfully removed hook for module {module_name}")
            except Exception as e:
                get_logger().warning(f"Failed to remove hook for module {module_name}: {e}")

        # 清空hook句柄字典
        self.hook_handles.clear()

    def _extract_common_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取公共的metadata信息
        
        Args:
            metadata: 原始元数据
            
        Returns:
            Dict[str, Any]: 提取的公共信息
        """
        return {
            'source_name': metadata.get('source_name', ''),
            'source_module': metadata.get('source_module'),
            'target_modules': metadata.get('target_modules', []),
            'target_names': metadata.get('target_names', []),
            'fusion_flag': metadata.get('fusion_flag', False),
            'num_attention_heads': metadata.get('num_attention_heads'),
            'num_key_value_heads': metadata.get('num_key_value_heads')
        }

    def _apply_norm_linear_smooth(self, common_metadata: Dict[str, Any]) -> None:
        """应用Norm-Linear平滑"""
        # 应用平滑
        self._apply_smooth_to_subgraph(
            NormLinearSubgraph(common_metadata['source_module'], common_metadata['target_modules']),
            common_metadata['target_modules']
        )

    def _apply_linear_linear_smooth(self, common_metadata: Dict[str, Any]) -> None:
        """应用Linear-Linear平滑"""  # 获取第一个目标模块（Linear-Linear通常只有一个目标）
        target_module = common_metadata['target_modules'][0] if common_metadata['target_modules'] else None

        if not target_module:
            get_logger().warning("Linear-Linear subgraph missing target module")
            return

        # 应用平滑
        self._apply_smooth_to_subgraph(
            LinearLinearSubgraph(common_metadata['source_module'], target_module),
            [target_module]
        )

    def _apply_ov_smooth(self, common_metadata: Dict[str, Any]) -> None:
        """应用OV平滑（输出-值投影）"""
        # 获取OV特定的信息
        o_name = common_metadata['target_names'][0] if common_metadata['target_names'] else ''
        v_name = common_metadata['source_name']
        o_module = common_metadata['target_modules'][0] if common_metadata['target_modules'] else None
        v_module = common_metadata['source_module']
        fusion_flag = common_metadata['fusion_flag']

        # 获取注意力头数量
        num_attention_heads = common_metadata['num_attention_heads']
        num_key_value_heads = common_metadata['num_key_value_heads']

        if num_attention_heads is None or num_key_value_heads is None:
            num_attention_heads = self.get_num_attention_heads()
            num_key_value_heads = self.get_num_key_value_heads()

        if not o_name or not v_name:
            get_logger().warning(f"OV subgraph missing necessary name information: o_name={o_name}, v_name={v_name}")
            return

        try:
            if not o_module or not v_module:
                get_logger().warning(
                    f"Cannot get OV subgraph modules: "
                    f"o_module={o_module is not None}, v_module={v_module is not None}")
                return

            if not isinstance(o_module, nn.Linear):
                get_logger().warning(f"O module {o_name} is not Linear type, skipping")
                return

            # 根据融合标志选择平滑方法
            if fusion_flag:
                self._apply_qkv_fusion_smooth(v_module, o_module, v_name, o_name,
                                              num_attention_heads, num_key_value_heads)
            else:
                self._apply_standard_ov_smooth(v_module, o_module, v_name, o_name,
                                               num_attention_heads, num_key_value_heads)

        except Exception as e:
            get_logger().error(f"Error occurred while applying OV smoothing: {e}")

    def _apply_qkv_fusion_smooth(self, v_module: nn.Module, o_module: nn.Module,
                                 v_name: str, o_name: str,
                                 num_attention_heads: int, num_key_value_heads: int) -> None:
        """
        应用QKV融合平滑
        
        Args:
            v_module: V投影模块
            o_module: O投影模块
            v_name: V模块名称
            o_name: O模块名称
            num_attention_heads: 注意力头数量
            num_key_value_heads: 键值头数量
        """
        if not isinstance(v_module, nn.Linear):
            get_logger().warning(f"V module {v_name} is not Linear type, skipping QKV fusion")
            return

        # 创建虚拟V模块
        virtual_v_module = VirtualVModule(v_module, num_attention_heads, num_key_value_heads)

        # 应用平滑
        self._apply_smooth_to_subgraph(
            OVSubgraph(
                o_proj=o_module,
                v_proj=virtual_v_module,
                num_attention_heads=num_attention_heads,
                key_value_heads=num_key_value_heads
            ),
            [o_module]
        )

        # 更新原始QKV模块权重
        virtual_v_module.update_qkv_weights()

        get_logger().debug(f"Successfully applied QKV fusion smoothing: {v_name} -> {o_name}")

    def _apply_standard_ov_smooth(self, v_module: nn.Module, o_module: nn.Module,
                                  v_name: str, o_name: str,
                                  num_attention_heads: int, num_key_value_heads: int) -> None:
        """
        应用标准OV平滑
        
        Args:
            v_module: V投影模块
            o_module: O投影模块
            v_name: V模块名称
            o_name: O模块名称
            num_attention_heads: 注意力头数量
            num_key_value_heads: 键值头数量
        """
        if not isinstance(v_module, nn.Linear):
            get_logger().warning(f"V module {v_name} is not Linear type, skipping standard OV smoothing")
            return

        # 应用平滑
        self._apply_smooth_to_subgraph(
            OVSubgraph(
                o_proj=o_module,
                v_proj=v_module,
                num_attention_heads=num_attention_heads,
                key_value_heads=num_key_value_heads
            ),
            [o_module]
        )

        get_logger().debug(f"Successfully applied standard OV smoothing: {v_name} -> {o_name}")

    def _apply_up_down_smooth(self, common_metadata: Dict[str, Any]) -> None:
        """应用Up-Down平滑（MLP门控机制）"""
        # 获取Up-Down特定的模块
        up_module = common_metadata['source_module']
        down_module = common_metadata['target_modules'][0] if len(common_metadata['target_modules']) > 0 else None
        gate_module = common_metadata['target_modules'][1] if len(common_metadata['target_modules']) > 1 else None

        if not all([up_module, down_module]):
            get_logger().warning(
                f"Up-Down subgraph missing necessary module information: "
                f"up={up_module is not None}, down={down_module is not None}")
            return

        # 应用平滑
        self._apply_smooth_to_subgraph(
            UpDownSubgraph(up_module, down_module, gate_module),
            [down_module]
        )

    def _apply_smooth_to_subgraph(self, subgraph_obj: Any, linear_modules: List[nn.Module]) -> None:
        """
        应用平滑到子图
        
        Args:
            subgraph_obj: 子图对象
            linear_modules: 线性模块列表
        """
        raise NotImplementedError("Subclasses must implement _apply_smooth_to_subgraph method")


    def _process_single_subgraph(self, subgraph: SubgraphInfo) -> None:
        """
        处理单个子图
        
        Args:
            subgraph: 子图信息
        """
        subgraph_type = subgraph.subgraph_type
        metadata = subgraph.metadata or {}

        get_logger().debug(
            f"Processing subgraph type: {subgraph_type}, name: {subgraph.name}"
        )
        common_metadata = self._extract_common_metadata(metadata)
        # 根据子图类型调用相应的处理方法
        if subgraph_type == "norm-linear":
            self._apply_norm_linear_smooth(common_metadata)
        elif subgraph_type == "linear-linear":
            self._apply_linear_linear_smooth(common_metadata)
        elif subgraph_type == "ov":
            self._apply_ov_smooth(common_metadata)
        elif subgraph_type == "up-down":
            self._apply_up_down_smooth(common_metadata)
        else:
            get_logger().warning(f"Unsupported subgraph type: {subgraph_type}")

    def _process_subgraphs_by_priority(self) -> None:
        """
        按优先级顺序处理子图
        
        优先级顺序（可配置）：
        1. up-down (最高优先级，MLP门控机制)
        2. ov (中等优先级，注意力机制)
        3. norm-linear (较低优先级，归一化层)
        4. linear-linear (最低优先级，线性层)
        """
        # 使用配置中的优先级设置
        priority_order = self.config.subgraph_priority

        # 按优先级排序子图
        sorted_subgraphs = sorted(
            self.subgraph_info,
            key=lambda x: priority_order.get(x.subgraph_type, 999)  # 未知类型优先级最低
        )

        get_logger().debug(f"Subgraph processing order after priority sorting:")
        for i, subgraph in enumerate(sorted_subgraphs):
            priority = priority_order.get(subgraph.subgraph_type, 999)
            get_logger().debug(f"  {i + 1}. {subgraph.subgraph_type} (priority: {priority}) - {subgraph.name}")

        # 按优先级顺序处理子图
        for subgraph in sorted_subgraphs:
            try:
                priority = priority_order.get(subgraph.subgraph_type, 999)
                self._process_single_subgraph(subgraph)
            except Exception as e:
                get_logger().error(f"Error occurred while processing subgraph {subgraph.name}: {e}")
                continue

    def _apply_smooth(self, name: str, module: nn.Module) -> None:
        get_logger().debug(f"Starting smoothing application to module: {name}")

        # 按优先级顺序处理子图：up-down -> ov -> norm-linear -> linear-linear
        self._process_subgraphs_by_priority()

        # 清理统计信息
        self.act_stats.clear()

        # 删除所有已安装的hook
        self._remove_all_hooks()

        get_logger().debug(f"Completed M4 smoothing, cleared statistics and hooks")
