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

import torch
import torch.nn as nn
import torch.nn.functional as F

from msmodelslim.core import QDType, QParam, QStorage, calculate_qparam
from msmodelslim.core import dequantize, fake_quantize
from msmodelslim.core.QAL import QABCRegistry
from msmodelslim.core.QAL import QScope
from msmodelslim.quant.ir import AutoFakeQuantDynamicCache
from .const import (
    int8_per_channel_sym,
    int8_per_channel_asym
)


@QABCRegistry.multi_register(
    dispatch_key=[
        int8_per_channel_sym,
        int8_per_channel_asym,
    ],
    abc_type=AutoFakeQuantDynamicCache
)
class FakeQuantDynamicCache(AutoFakeQuantDynamicCache):
    def __init__(
            self,
            x_q_param: QParam,
    ):
        super().__init__()

        self.x_q_param = x_q_param

        self.kv_cache_scale = nn.Parameter(x_q_param.ext["scale"], requires_grad=False)
        self.kv_cache_offset = nn.Parameter(x_q_param.ext["offset"], requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-2, -3)
        x_shape = x.shape
        x = x.reshape(-1, x.shape[-1] * x.shape[-2])
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x), self.x_q_param).value
        x_q_dq = x_q_dq.reshape(x_shape)
        x_q_dq = x_q_dq.transpose(-2, -3)
        return x_q_dq

