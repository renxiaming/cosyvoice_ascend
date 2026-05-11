# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
import torch


def _pack_int4(weight) -> torch.Tensor:
    """
    Pack int4 weight to int8 weight
    @param weight: torch.Tensor, int4 weight
    @return: torch.Tensor, int8 weight
    """
    weight = weight.to(torch.int8)
    e = 0  # number of experts
    if len(weight.shape) == 2:
        k, n = weight.shape
    elif len(weight.shape) == 3:
        e, k, n = weight.shape
    n_new = n // 2 + n % 2

    if n_new != n // 2:
        raise AssertionError("n dimension should be even")
    
    weight = weight.reshape(-1, 2)
    weight0 = weight[:, :1]
    weight1 = weight[:, 1:]

    weight1_4 = torch.bitwise_left_shift(weight1, 4)
    weight2_4 = weight0 & 0b00001111

    weight_add = torch.bitwise_or(weight1_4, weight2_4)
    if e == 0:
        weight_res = weight_add.reshape(k, n_new)
    else:
        weight_res = weight_add.reshape(e, k, n_new)
    return weight_res


def w4a8_pack_int4(save_quant_weight):
    """
    Pack int4 weight to int8 weight
    @param save_quant_weight: torch.Tensor, int4 weight
    @return: torch.Tensor, int8 weight
    """
    weight = save_quant_weight.transpose(-1, -2).contiguous()
    packed_weight_tensor = _pack_int4(weight)
    packed_weight_tensor = packed_weight_tensor.transpose(-1, -2).contiguous()
    return packed_weight_tensor
