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
from pydantic import validate_call

import msmodelslim.quant.ir as qir
from msmodelslim.core import fake_quantize, quantize, dequantize, calculate_qparam
from msmodelslim.core.QAL import QABCRegistry, QDType, QStorage, QParam, QScope
from msmodelslim.quant.observer import MsMinMaxObserver, MinMaxObserverConfig
from msmodelslim.utils.exception import SpecError, SchemaValidateError
from msmodelslim.utils.logging import logger_setter
from ..base import AutoActQuantizer, AutoWeightQuantizer, QConfig


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.int8_per_tensor_sym, "minmax"),
        (qir.int8_per_tensor_asym, "minmax"),
    ],
    abc_type=AutoActQuantizer
)
@logger_setter()
class ActPerTensorMinmax(AutoActQuantizer):

    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        minmax_config = MinMaxObserverConfig.model_validate({})
        self.minmax_observer = MsMinMaxObserver(minmax_config)
        self.q_param: Optional[QParam] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.minmax_observer.update(x)
        min_val, max_val = self.minmax_observer.get_min_max()
        self.q_param = calculate_qparam(
            min_val=min_val,
            max_val=max_val,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope(self.config.scope),
            symmetric=self.config.symmetric,
        )
        return fake_quantize(QStorage(dtype=QDType.FLOAT, value=x), self.q_param).value

    def get_q_param(self) -> QParam:
        if self.q_param is None:
            raise SpecError("No q_param was set", action="Please call forward first")
        return self.q_param


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.int8_per_token_asym, "minmax"),
        (qir.int8_per_token_sym, "minmax"),
    ],
    abc_type=AutoActQuantizer
)
@logger_setter()
class ActPerTokenMinmax(AutoActQuantizer):

    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        minmax_config = MinMaxObserverConfig.model_validate({})
        self.minmax_observer = MsMinMaxObserver(minmax_config)
        self.q_param: Optional[QParam] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x_reshaped = x.reshape(-1, x.shape[-1])
        self.minmax_observer.reset()
        self.minmax_observer.update(x_reshaped)
        min_val, max_val = self.minmax_observer.get_min_max()
        self.q_param = calculate_qparam(
            min_val=min_val,
            max_val=max_val,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope(self.config.scope),
            symmetric=self.config.symmetric,
        )
        return fake_quantize(QStorage(dtype=QDType.FLOAT, value=x), self.q_param).value.reshape(x_shape)

    def get_q_param(self) -> QParam:
        if self.q_param is None:
            return QParam(scheme=self.config.to_scheme())
        return self.q_param


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.int8_per_channel_sym, "minmax"),
        (qir.int8_per_channel_asym, "minmax"),
    ],
    abc_type=AutoActQuantizer
)
class ActPerChannelMinmax(AutoActQuantizer):
    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        minmax_config = MinMaxObserverConfig(dim=0, keepdim=False)
        self.minmax_observer = MsMinMaxObserver(minmax_config)
        self.q_param: Optional[QParam] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x_reshaped = x.reshape(-1, x.shape[-1])
        self.minmax_observer.reset()
        self.minmax_observer.update(x_reshaped)
        min_val, max_val = self.minmax_observer.get_min_max()
        self.q_param = calculate_qparam(
            min_val=min_val,
            max_val=max_val,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope(self.config.scope),
            symmetric=self.config.symmetric,
        )
        return fake_quantize(QStorage(dtype=QDType.FLOAT, value=x), self.q_param).value.reshape(x_shape)

    def get_q_param(self) -> QParam:
        if self.q_param is None:
            raise RuntimeError("No q_param was set, please call forward first")
        return self.q_param


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.int8_per_channel_sym, "minmax"),
        (qir.int4_per_channel_sym, "minmax")
    ],
    abc_type=AutoWeightQuantizer
)
@logger_setter()
class WeightPerChannelMinmax(AutoWeightQuantizer):
    def __init__(self, config: QConfig):
        super().__init__()
        minmax_config = MinMaxObserverConfig(dim=0, keepdim=False)
        self.config = config
        self.minmax_observer = MsMinMaxObserver(minmax_config)
        self.weight: Optional[QStorage] = None
        self.bias: Optional[torch.Tensor] = None
        self.w_q_param: Optional[QParam] = None
        self.w_q_storage: Optional[QStorage] = None

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.weight is None:
            raise SpecError("No weight was set", action="please call init_weight first")
        self.minmax_observer.update(self.weight.T.value)
        min_val, max_val = self.minmax_observer.get_min_max()
        self.w_q_param = calculate_qparam(
            min_val=min_val,
            max_val=max_val,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope(self.config.scope),
            symmetric=self.config.symmetric,
        )
        self.w_q_storage = quantize(self.weight.T, self.w_q_param).T
        return dequantize(self.w_q_storage.T, self.w_q_param).T.value

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def init_weight(self, weight: QStorage, bias: Optional[torch.Tensor] = None) -> None:
        self.weight = weight
        self.bias = bias

    def get_q_storage(self) -> QStorage:
        if self.w_q_storage is None:
            _ = self.forward(None)
        return self.w_q_storage

    def get_q_param(self) -> QParam:
        if self.w_q_param is None:
            _ = self.forward(None)
        return self.w_q_param


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.int8_per_group_sym, "minmax"),
    ],
    abc_type=AutoWeightQuantizer
)
@logger_setter()
class WeightPerGroupMinmax(AutoWeightQuantizer):

    def __init__(self, config: QConfig):
        super().__init__()

        if config.ext is None or "group_size" not in config.ext:
            raise SchemaValidateError("\"group_size\" is needed in config.ext field",
                                      action="Please make sure group_size in config.ext")

        if not isinstance(config.ext.get("group_size"), int) or config.ext.get("group_size") <= 0:
            raise SchemaValidateError("\"group_size\" must be a positive integer",
                                      action="Please make sure group_size is a positive integer")

        minmax_config = MinMaxObserverConfig(dim=0, keepdim=False)
        self.config = config
        self.minmax_observer = MsMinMaxObserver(minmax_config)
        self.group_size = config.ext.get("group_size")
        self.weight: Optional[QStorage] = None
        self.bias: Optional[torch.Tensor] = None
        self.w_q_param: Optional[QParam] = None
        self.w_q_storage: Optional[QStorage] = None

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.weight is None:
            raise SpecError("No weight was set", action="Please call init_weight first")
        if self.weight.value.shape[-1] % self.group_size != 0:
            raise SpecError("The last dimension of weight must be divisible by group_size",
                            action="Please make sure the last dimension of weight can be divisible by group_size")
        grouped_w = self.weight.value.reshape(-1, self.group_size)
        self.minmax_observer.update(grouped_w)
        min_val, max_val = self.minmax_observer.get_min_max()
        self.w_q_param = calculate_qparam(
            min_val=min_val,
            max_val=max_val,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope(self.config.scope),
            symmetric=self.config.symmetric,
        )
        self.w_q_storage = quantize(self.weight, self.w_q_param)
        return dequantize(self.w_q_storage, self.w_q_param).value

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def init_weight(self, weight: QStorage, bias: Optional[torch.Tensor] = None) -> None:
        self.weight = weight
        self.bias = bias

    def get_q_storage(self) -> QStorage:
        if self.w_q_storage is None:
            _ = self.forward(None)
        return self.w_q_storage

    def get_q_param(self) -> QParam:
        if self.w_q_param is None:
            _ = self.forward(None)
        return self.w_q_param
