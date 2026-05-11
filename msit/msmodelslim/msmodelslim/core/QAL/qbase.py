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
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


class QDType(Enum):
    FLOAT = "float"

    INT8 = "int8"
    INT4 = "int4"

    PLACEHOLDER = "placeholder"


class QScope(Enum):
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_GROUP = "per_group"
    PER_BLOCK = "per_block"
    PER_TOKEN = "per_token"

    PLACEHOLDER = "placeholder"


@dataclass(frozen=True)
class QScheme:
    scope: QScope = QScope.PLACEHOLDER
    dtype: QDType = QDType.PLACEHOLDER
    symmetric: bool = True


@dataclass
class QParam:
    """

    QParam用于描述量化参数，包括量化范围、量化类型、量化对称性等。

    """

    scheme: QScheme
    ext: Any = None


_TORCH_FLOAT_TYPE: torch.dtype = torch.float32


@dataclass
class QStorage:
    """

    QStorage用于描述某一种自定义数据类型的Tensor。

    """

    dtype: QDType
    value: torch.Tensor
    ext: Any = None

    @staticmethod
    @contextmanager
    def set_value_float_type(dtype: torch.dtype):
        global _TORCH_FLOAT_TYPE
        old_dtype = _TORCH_FLOAT_TYPE
        _TORCH_FLOAT_TYPE = dtype
        yield
        _TORCH_FLOAT_TYPE = old_dtype

    @property
    def T(self):
        return self.same_like(self.value.T)

    def same_like(self, new_val: torch.Tensor):
        return QStorage(dtype=self.dtype, value=new_val, ext=self.ext)

    def to(self, dtype: QDType):
        self.dtype = dtype

        if self.dtype == QDType.FLOAT:
            self.value = self.value.to(_TORCH_FLOAT_TYPE)

        if self.dtype in [QDType.INT8, QDType.INT4]:
            self.value = self.value.to(torch.int8)

        return self

    def reshape(self, shape: torch.Size):
        self.value.reshape(shape)
        return self
