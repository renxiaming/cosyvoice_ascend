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


from dataclasses import dataclass
from typing import Union, List, Dict, Any, Optional

import torch
from torch import nn as nn


class RMSNormBias(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states + self.bias


class Subgraph:
    pass


@dataclass
class NormLinearSubgraph(Subgraph):
    """

    该子图可以用以下计算公式表示：

    x = norm(x)
    y = torch.cat([linear(x) for linear in linears], dim=-1)

    其中，norm是归一化层，linears是线性层列表。

    """

    norm: Union[RMSNormBias]
    linears: List[nn.Linear]


@dataclass
class LinearLinearSubgraph(Subgraph):
    """
    该子图可以用以下计算公式表示：

    y = linear2(linear1(x))

    其中，linear1和linear2是线性层，linear1的输出是linear2的输入。
    """

    linear1: nn.Linear
    linear2: nn.Linear


@dataclass
class OVSubgraph(Subgraph):
    """
    该子图代表了Attention中的O和V的子图。

    对于MHA，通常有
        num_attention_heads = key_value_heads

    对于MQA，通常有
        key_value_heads = 1

    对于GQA，通常有
        num_attention_heads > key_value_heads
        num_attention_heads % key_value_heads == 0

    """
    o_proj: nn.Linear
    v_proj: nn.Linear
    num_attention_heads: int
    key_value_heads: int


@dataclass
class UpDownSubgraph(Subgraph):
    """
    该子图代表了MLP中的UpDown子图，该子图通常可以用以下计算公式表示：

    y = down_proj(ReLU(gate_proj(x)) * up_proj(x))

    其中，up_proj和down_proj是线性层，gate_proj是sigmoid激活函数，ReLU是激活函数。
    """

    up_proj: nn.Linear
    down_proj: nn.Linear
    gate_proj: nn.Linear


@dataclass
class SmoothContext:
    version: int
    a_smooth_scale: torch.Tensor
    w_smooth_scale: torch.Tensor
    tensors: List[torch.Tensor]
    shift: torch.Tensor
    ext: Dict[str, Any]


@dataclass
class IterSmoothConfig:
    """

    iter_smooth算法的配置项。
    允许后续扩展配置项，但仅可新增新字段，且不得修改已有字段，
    version用于指定配置版本号，每次修改后，版本号需要加1。

    """

    version: int = 1
    alpha: float = 0.9
    shift: bool = False
    scale_min: float = 1e-5


@dataclass
class FlexSmoothQuantConfig:
    """

    flex_smooth_quant算法的配置项。
    允许后续扩展配置项，但仅可新增新字段，且不得修改已有字段，
    version用于指定配置版本号，每次修改后，版本号需要加1。

    """

    version: int = 1
    alpha: Optional[float] = None
    beta: Optional[float] = None   

