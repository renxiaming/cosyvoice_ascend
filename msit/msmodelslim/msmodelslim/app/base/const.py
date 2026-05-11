# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from enum import Enum


class ExtendedEnum(Enum):
    def __str__(self):
        return self.value

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def names(cls):
        return list(map(lambda c: c.name, cls))


class DeviceType(ExtendedEnum):
    NPU = "npu"  # 昇腾NPU
    CPU = "cpu"  # CPU


class QuantType(ExtendedEnum):
    W4A8 = "w4a8"  # 权重INT4量化，激活值INT8量化
    W8A16 = "w8a16"  # 权重INT8量化，激活值不量化
    W8A8 = "w8a8"  # 权重INT8量化，激活值INT8量化
    W8A8S = "w8a8s"  # 权重INT8稀疏量化，激活值INT8量化
    W8A8C8 = "w8a8c8"  # 权重INT8量化，激活值INT8量化，KVCache INT8量化


class PipelineType(Enum):
    """Pipeline类型枚举"""
    AUTO = "auto"
    MODEL_WISE = "model_wise"
    LAYER_WISE = "layer_wise"
