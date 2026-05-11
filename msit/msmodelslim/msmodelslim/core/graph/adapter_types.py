# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# 支持的子图类型常量
SUPPORTED_SUBGRAPH_TYPES = [
    'norm-linear',
    'linear-linear',
    'ov',
    'up-down'
]


@dataclass
class MappingConfig:
    """映射关系配置"""
    source: str
    targets: List[str]


@dataclass
class FusionConfig:
    """融合配置联合体
    
    用于管理不同类型的融合配置，支持扩展新的融合类型
    """
    fusion_type: str = "none"  # 融合类型：none, qkv, custom等

    # QKV融合相关配置
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None

    # 自定义融合配置（可扩展）
    custom_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """验证融合配置的有效性"""
        if self.fusion_type == "qkv":
            if self.num_attention_heads is None or self.num_key_value_heads is None:
                raise ValueError("QKV融合类型必须提供num_attention_heads和num_key_value_heads")
        elif self.fusion_type == "custom":
            if not self.custom_config:
                raise ValueError("自定义融合类型必须提供custom_config")
        elif self.fusion_type != "none":
            raise ValueError(f"不支持的融合类型: {self.fusion_type}")


@dataclass
class AdapterConfig:
    """融合配置结构体
    
    包含模型融合和子图处理的所有配置参数
    """
    # 子图类型（必需）
    subgraph_type: str
    # 自定义的映射关系（必需）
    mapping: Optional[MappingConfig] = None
    # 融合配置（可选）
    fusion: FusionConfig = field(default_factory=lambda: FusionConfig())

    def __post_init__(self):
        """验证配置的有效性"""
        if self.subgraph_type is None:
            raise ValueError("subgraph_type is required")

        if self.subgraph_type not in SUPPORTED_SUBGRAPH_TYPES:
            raise ValueError(f"subgraph_type: {self.subgraph_type} 不是支持的子图类型")

        if self.mapping is None:
            raise ValueError("mapping is required")


@dataclass
class SubgraphInfo:
    """
    子图信息数据类
    
    包含子图的名称、类型、节点列表等基本信息。
    """
    name: str
    subgraph_type: str
    metadata: Optional[Dict[str, Any]] = None
