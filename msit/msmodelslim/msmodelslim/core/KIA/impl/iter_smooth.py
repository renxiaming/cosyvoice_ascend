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

from msmodelslim.core.QAL.qregistry import QFuncRegistry
from msmodelslim.core.QAL.qtypes import (
    Subgraph,
    NormLinearSubgraph,
    LinearLinearSubgraph,
    OVSubgraph,
    UpDownSubgraph,
    SmoothContext,
    IterSmoothConfig,
)


@torch.no_grad()
def compute_smooth_scale(w_scale: torch.Tensor, a_scale: torch.Tensor, config: IterSmoothConfig):
    device = w_scale.device
    dtype = w_scale.dtype
    w_scale = w_scale.max(dim=0)[0].to(torch.float32).clamp(min=1e-5).to(dtype=dtype)
    scales = (a_scale.pow(config.alpha).to(device) / 
              w_scale.pow(1 - config.alpha)).to(torch.float32).clamp(
                  min=config.scale_min).to(dtype=dtype)
    return scales


@torch.no_grad()
def apply_smooth_scale_shift(layer, scales, shift=None):
    device = layer.weight.device
    dtype = layer.weight.dtype
    layer.weight.mul_(scales)
    if shift is not None:
        if layer.bias is None:
            layer.bias = nn.Parameter(shift).to(device).to(dtype)
        else:
            layer.bias.add_(shift.to(device).to(dtype))


@torch.no_grad()
def prepare_mqga_parameters(num_attention_heads, num_key_value_heads):
    ratio = num_attention_heads // num_key_value_heads
    scales_pad_size = 0
    return ratio, scales_pad_size


@torch.no_grad()
def reduce_scales_for_mqga(scales, shape_ratio, num_attention_heads):
    # Assuming heads for K V activations are broadcasted with following pattern:
    # [h1, h2, h3, h4] -> [h1, ... , h1, h2,..., h2, h3, ..., h3, h4, ..., h4]
    num_q_heads = num_attention_heads
    num_kv_heads = num_q_heads // shape_ratio
    head_emb_size = scales.size(0) // num_q_heads
    reduced_scales, updated_scales = [], []
    copied_scales_slices = scales.split(scales.size(0) // num_kv_heads)
    
    for gr_idx in range(num_kv_heads):
        subset_of_scales = copied_scales_slices[gr_idx].view(-1, head_emb_size)
        repeat_num = subset_of_scales.size(0)
        reduced_scales.append(subset_of_scales.mean(0))
        updated_scales.append(torch.cat([reduced_scales[-1]] * repeat_num))
    return torch.cat(updated_scales), torch.cat(reduced_scales)


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(OVSubgraph, 1), api_name="iter_smooth")
def iter_smooth_impl_OV(subgraph: Subgraph, config: IterSmoothConfig, context: SmoothContext) -> None:
    # 计算smooth scale
    a_scale = context.a_smooth_scale
    w_scale = subgraph.o_proj.weight
    scales = compute_smooth_scale(w_scale, a_scale, config)
    shape_ratio, scales_pad_size = prepare_mqga_parameters(subgraph.num_attention_heads, subgraph.key_value_heads)
    o_scales, v_scales = reduce_scales_for_mqga(scales, shape_ratio, subgraph.num_attention_heads)
    o_shift = None
    v_shift = None
    if config.shift:
        # 使用非原地操作避免梯度问题
        o_shift = torch.mm(context.shift.unsqueeze(0), subgraph.o_proj.weight.data.clone().T).squeeze(0)
        v_shift = context.shift * -1
    apply_smooth_scale_shift(subgraph.o_proj, o_scales.view(1, -1), o_shift)
    apply_smooth_scale_shift(subgraph.v_proj, 1.0 / v_scales.view(-1, 1), v_shift)
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(UpDownSubgraph, 1), api_name="iter_smooth")
def iter_smooth_impl_UpDown(subgraph: Subgraph, config: IterSmoothConfig, context: SmoothContext) -> None:
    # 计算smooth scale
    a_scale = context.a_smooth_scale
    w_scale = subgraph.down_proj.weight
    scales = compute_smooth_scale(w_scale, a_scale, config)
    down_shift = None
    up_shift = None
    if config.shift:
        down_shift = torch.mm(context.shift.unsqueeze(0), subgraph.down_proj.weight.data.clone().T).squeeze(0)
        up_shift = context.shift * -1
    apply_smooth_scale_shift(subgraph.down_proj, scales.view(1, -1), down_shift)
    apply_smooth_scale_shift(subgraph.up_proj, 1.0 / scales.view(-1, 1), up_shift)
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(LinearLinearSubgraph, 1), api_name="iter_smooth")
def iter_smooth_impl_LinearLinear(subgraph: Subgraph, config: IterSmoothConfig, context: SmoothContext) -> None:
    # 计算smooth scale
    a_scale = context.a_smooth_scale
    w_scale = subgraph.linear2.weight
    scales = compute_smooth_scale(w_scale, a_scale, config)
    linear2_shift = None
    linear1_shift = None
    if config.shift:
        linear2_shift = torch.mm(context.shift.unsqueeze(0), subgraph.linear2.weight.data.clone().T).squeeze(0)
        linear1_shift = context.shift * -1
    apply_smooth_scale_shift(subgraph.linear2, scales.view(1, -1), linear2_shift)
    apply_smooth_scale_shift(subgraph.linear1, 1.0 / scales.view(-1, 1), linear1_shift)
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(NormLinearSubgraph, 1), api_name="iter_smooth")
def iter_smooth_impl_NormLinear(subgraph: Subgraph, config: IterSmoothConfig, context: SmoothContext) -> None:    
    # 计算smooth scale
    a_scale = context.a_smooth_scale
    w_scale = []
    for fc in subgraph.linears:
        fc_weight = fc.weight
        stat = fc_weight.abs().max(dim=0, keepdim=True)[0]
        w_scale.append(stat)
    w_scale = torch.cat(w_scale, dim=0)
    scales = compute_smooth_scale(w_scale, a_scale, config)

    for fc in subgraph.linears:
        linear_shift = None
        if config.shift:
            linear_shift = torch.mm(context.shift.unsqueeze(0), fc.weight.data.clone().T).squeeze(0)
        apply_smooth_scale_shift(fc, scales.view(1, -1), linear_shift)
    
    norm_shift = None
    if config.shift:
        norm_shift = context.shift * -1 
    apply_smooth_scale_shift(subgraph.norm, (1.0 / scales).squeeze(), norm_shift)
    return