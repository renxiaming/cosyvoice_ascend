# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from dataclasses import dataclass, field
from typing import List


@dataclass
class Metadata:
    # ID of the quantization config, e.g., 'Qwen3-32B W8A8'
    config_id: str
    # score of the quantization config, used to sort the quantization configs
    score: float
    # label of the quantization config, used to filter the quantization configs.
    # e.g., # {'w_bit': 8, 'a_bit': 8, 'is_sparse': True, 'kv_cache': True}
    label: dict
    # verified model types, e.g., ['LLaMa3.1-70B', 'Qwen2.5-72B']
    verified_model_types: List[str] = field(default_factory=list)


@dataclass
class BaseQuantConfig:
    apiversion: str  # API version
    metadata: Metadata  # metadata of the quantization config
    spec: object  # spec of the quantization config
