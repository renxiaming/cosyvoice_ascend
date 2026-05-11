# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
__all__ = ['NaiveQuantizationApplication', 'PracticeManagerInterface']

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass
from .application import NaiveQuantizationApplication
from .practice_interface import PracticeManagerInterface
