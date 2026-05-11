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

from typing import Optional

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from pydantic import validate_call
from torch import nn

from msmodelslim.core import QDType, QStorage
from msmodelslim.quant.ir import AutoFakeQuantLinear
from msmodelslim.utils.logging import logger_setter
from .base import AutoActQuantizer, AutoWeightQuantizer, QConfig


class LinearQConfig(BaseModel):
    act: QConfig
    weight: QConfig

    model_config = ConfigDict(extra="forbid")


@logger_setter()
class LinearQuantizer(nn.Module):

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(self, config: LinearQConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.input_quantizer = AutoActQuantizer.from_config(config.act)
        self.weight_quantizer = AutoWeightQuantizer.from_config(config.weight)
        self.weight: Optional[nn.Parameter] = None
        self.bias: Optional[nn.Parameter] = None
        self.q_weight: Optional[QStorage] = None

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def setup(self, linear: nn.Linear):
        self.weight = linear.weight
        self.bias = linear.bias
        self.weight_quantizer.init_weight(QStorage(QDType.FLOAT, value=self.weight), self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with QStorage.set_value_float_type(x.dtype):
            x = self.input_quantizer(x)
            weight = self.weight_quantizer(x)
        return F.linear(x, weight, self.bias)

    def deploy(self):
        return AutoFakeQuantLinear.create(
            self.input_quantizer.get_q_param(),
            self.weight_quantizer.get_q_param(),
            self.weight_quantizer.get_q_storage(),
            self.bias
        )
