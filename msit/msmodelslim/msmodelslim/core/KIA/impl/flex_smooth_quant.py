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
import numpy as np

from msmodelslim.core.QAL.qregistry import QFuncRegistry
from msmodelslim.core.QAL.qtypes import (
    Subgraph,
    NormLinearSubgraph,
    LinearLinearSubgraph,
    OVSubgraph,
    UpDownSubgraph,
    SmoothContext,
    FlexSmoothQuantConfig,
)
from msmodelslim.utils.exception import MisbehaviorError, SchemaValidateError, UnexpectedError
from msmodelslim.utils.logging import get_logger


@torch.no_grad()
def quant_int8sym(x, dim=-1):
    xmax = torch.abs(x).max(dim=dim, keepdim=True)[0]
    interval = xmax / 127
    quanted = x / interval
    quanted = torch.round(quanted).clip(min=-127, max=127)
    recovered = quanted * interval
    return recovered


@torch.no_grad()
def quant_int8tasym(x: torch.Tensor):
    x_max = torch.max(x)
    x_min = torch.min(x)
    eps = torch.tensor([torch.finfo(torch.float32).eps]).type_as(x_min)
    scale = (x_max - x_min) / 255
    scale = torch.max(scale, eps)
    zero_point = -1 * x_min / scale
    zero_point = zero_point.round() - 128
    qx = (x / scale + zero_point).round()
    recovered = (qx - zero_point) * scale
    return recovered


@torch.no_grad()
def scale_descale(act, fc_weights, alpha, beta, act_asym=True):
    fp_golden = torch.matmul(act, fc_weights.T)
    normal = torch.mean(fp_golden ** 2) ** 0.5
    scale = torch.max(torch.abs(act), dim=0, keepdims=True)[0] ** alpha * \
        torch.max(torch.abs(fc_weights), dim=0, keepdims=True)[0] ** (-beta)
    
    scaled_act = act / scale
    scaled_w_scale = fc_weights * scale

    quant_weight = quant_int8sym(scaled_w_scale)

    if act_asym:
        quant_act = quant_int8sym(scaled_act)
    else:
        quant_act = quant_int8tasym(scaled_act)

    quant_result = torch.matmul(quant_act, quant_weight.T)
    normal_mse = (torch.mean((torch.abs(quant_result - fp_golden) ** 2)) ** 0.5) / normal
    return normal_mse


@torch.no_grad()
def search_alpha_beta(act, fc_weights, normal_mse_best=1.0, best_alpha=None):
    best_hyper_p = 0.0
    for hyper_p in np.round(np.arange(0.0, 1.05, 0.05), decimals=2):
        if best_alpha is None:
            hyper_alpha = hyper_p
            hyper_beta = 1 - hyper_p
        else:
            hyper_alpha = best_alpha
            hyper_beta = hyper_p
        normal_mse = scale_descale(act, fc_weights, hyper_alpha, hyper_beta)
        if normal_mse <= normal_mse_best:
            normal_mse_best = normal_mse
            best_hyper_p = hyper_p
    return best_hyper_p, normal_mse_best


@torch.no_grad()
def compute_smooth_scale(a_scale: torch.Tensor, w_scale: torch.Tensor, alpha: float, beta: float):
    scales = (a_scale.pow(alpha) / w_scale.pow(beta)).to(torch.float32).clamp(min=1e-5).to(dtype=a_scale.dtype)
    return scales


@torch.no_grad()
def apply_smooth_scale_shift(layer, scales):
    layer.weight.mul_(scales)


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
@QFuncRegistry.register(dispatch_key=(OVSubgraph, 1), api_name="flex_smooth_quant")
def flex_smooth_impl_OV(subgraph: Subgraph, config: FlexSmoothQuantConfig, context: SmoothContext) -> None:
    """
    实现 OV（Output-Value）子图的平滑处理
    
    该函数用于优化注意力机制中的输出投影层（O）和值投影层（V），
    通过计算和应用平滑尺度来改善模型的量化性能。
    
    Args:
        subgraph: 包含 v_proj 和 o_proj 的子图对象
        config: 平滑配置参数，包含 alpha 和 beta 值
        context: 平滑上下文，包含激活统计信息和平滑尺度
    
    Raises:
        MisbehaviorError: 当输入参数无效或处理失败时
        UnexpectedError: 当遇到未预期的错误时
    
    Returns:
        None: 函数直接修改子图参数，无返回值
    """
    
    # 检查子图是否包含必要的投影层
    if not hasattr(subgraph, 'v_proj') or subgraph.v_proj is None:
        raise MisbehaviorError(
            "Subgraph must have a valid v_proj attribute",
            action="Please ensure the subgraph contains a valid v_proj layer"
        )
    
    if not hasattr(subgraph, 'o_proj') or subgraph.o_proj is None:
        raise MisbehaviorError(
            "Subgraph must have a valid o_proj attribute",
            action="Please ensure the subgraph contains a valid o_proj layer"
        )
    
    # 步骤1: 获取设备和数据类型信息
    # 从 V 投影层获取设备信息，从 O 投影层获取数据类型
    tmp_device = next(subgraph.v_proj.parameters()).device
    dtype = subgraph.o_proj.weight.dtype
    
    get_logger().debug(f"Using device: {tmp_device}, dtype: {dtype}")

    
    # 步骤2: 处理激活张量
    # 将激活张量列表转换为单个张量，统一数据类型和设备
    if context.tensors is not None:
        # 过滤掉 None 值并转换张量
        valid_tensors = [
            tensor.to(dtype=dtype).to(tmp_device)
            for tensor in context.tensors
            if tensor is not None and torch.is_tensor(tensor)
        ]
        
        if not valid_tensors:
            get_logger().warning("No valid tensors found in context, skipping smoothing")
            return
        
        # 连接所有有效的张量
        tensors = torch.cat(valid_tensors, dim=0)
        get_logger().debug(f"Successfully processed {len(valid_tensors)} tensors, total shape: {tensors.shape}")
    else:
        get_logger().warning("Context tensors is None, skipping smoothing")
        return
    
    # 重塑张量维度为 (batch_size * sequence_length, hidden_dim)
    act = tensors.view(-1, tensors.shape[-1])

    # 步骤3: 获取激活平滑尺度和权重
    a_scale = context.a_smooth_scale
    if a_scale is None:
        raise MisbehaviorError(
            "Activation smooth scale is None in context",
            action="Please ensure the context contains valid activation smooth scale data"
        )
    
    # 获取输出投影层的权重
    fc_weights = subgraph.o_proj.weight
    
    get_logger().debug(f"Activation scale shape: {a_scale.shape}, Weight shape: {fc_weights.shape}")


    try:
        # 步骤4: 确定最优的 alpha 和 beta 参数
        if config.alpha is None or config.beta is None:
            get_logger().debug("Alpha or beta not provided, searching for optimal values...")
            
            # 搜索最优的 alpha 值
            best_alpha, normal_mse_best = search_alpha_beta(
                act, fc_weights, normal_mse_best=1.0, best_alpha=None
            )
            get_logger().debug(f"Found optimal alpha: {best_alpha:.6f}, MSE: {normal_mse_best:.6f}")
            
            # 基于最优 alpha 搜索最优的 beta 值
            best_beta, normal_mse_best = search_alpha_beta(
                act, fc_weights, normal_mse_best=normal_mse_best, best_alpha=best_alpha
            )
            get_logger().debug(f"Found optimal beta: {best_beta:.6f}, final MSE: {normal_mse_best:.6f}")
        else:
            # 使用配置中提供的参数
            best_alpha = config.alpha
            best_beta = config.beta
            get_logger().debug(f"Using provided alpha: {best_alpha}, beta: {best_beta}")
        
    except Exception as e:
        raise UnexpectedError(
            f"Failed to determine optimal alpha/beta parameters: {e}",
            action="Please check the search_alpha_beta function and ensure input tensors are valid"
        ) from e
    
    # 步骤5: 计算权重统计和尺度
    # 计算权重绝对值的最大值，用于归一化
    weight_abs = subgraph.o_proj.weight.abs()
    weight_max = weight_abs.max(dim=0, keepdim=True)[0]
    
    # 计算权重尺度，应用最小阈值限制
    w_scale = weight_max.max(dim=0)[0]
    w_scale = w_scale.to(torch.float32).clamp(min=1e-5).to(dtype=dtype)
    
    get_logger().debug(f"Weight scale shape: {w_scale.shape}, min: {w_scale.min():.6f}, max: {w_scale.max():.6f}")
    
    try:
        # 步骤6: 计算平滑尺度
        # 基于激活尺度、权重尺度和最优参数计算平滑尺度
        scales = compute_smooth_scale(a_scale, w_scale, best_alpha, best_beta)
        get_logger().debug(f"Computed smooth scales shape: {scales.shape}")
        
        # 准备 MQGA（Multi-Query Grouped Attention）参数
        shape_ratio, scales_pad_size = prepare_mqga_parameters(
            subgraph.num_attention_heads, subgraph.key_value_heads
        )
        get_logger().debug(f"MQGA parameters - shape_ratio: {shape_ratio}, pad_size: {scales_pad_size}")
        
        # 为 O 和 V 投影层分别计算尺度
        o_scales, v_scales = reduce_scales_for_mqga(
            scales, shape_ratio, subgraph.num_attention_heads
        )
        get_logger().debug(f"O scales shape: {o_scales.shape}, V scales shape: {v_scales.shape}")
        
    except Exception as e:
        raise UnexpectedError(
            f"Failed to compute smooth scales: {e}",
            action="Please check the smooth scale computation functions and MQGA parameters"
        ) from e
    
    try:
        # 步骤7: 应用平滑到投影层
        # 对输出投影层应用平滑尺度
        apply_smooth_scale_shift(subgraph.o_proj, o_scales.view(1, -1))
        get_logger().debug("Successfully applied smoothing to output projection layer")
        
        # 对值投影层应用平滑尺度的倒数（补偿输出层的变化）
        apply_smooth_scale_shift(subgraph.v_proj, 1.0 / v_scales.view(-1, 1))
        get_logger().debug("Successfully applied smoothing to value projection layer")
        
        get_logger().debug("OV smoothing completed successfully")
        
    except Exception as e:
        raise UnexpectedError(
            f"Failed to apply smoothing to projection layers: {e}",
            action="Please check the projection layer configurations and ensure they support parameter modification"
        ) from e
    
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(UpDownSubgraph, 1), api_name="flex_smooth_quant")
def flex_smooth_impl_UpDown(subgraph: Subgraph, config: FlexSmoothQuantConfig, context: SmoothContext) -> None:
    """
    实现 Up-Down 子图的平滑处理
    
    该函数用于优化 MLP 门控机制中的上投影层（Up）和下投影层（Down），
    通过计算和应用平滑尺度来改善模型的量化性能。
    
    Args:
        subgraph: 包含 up_proj、down_proj 和 gate_proj 的子图对象
        config: 平滑配置参数，包含 alpha 和 beta 值
        context: 平滑上下文，包含激活统计信息和平滑尺度
    
    Raises:
        MisbehaviorError: 当输入参数无效或处理失败时
        UnexpectedError: 当遇到未预期的错误时
    
    Returns:
        None: 函数直接修改子图参数，无返回值
    """
    
    
    # 检查子图是否包含必要的投影层
    if not hasattr(subgraph, 'up_proj') or subgraph.up_proj is None:
        raise MisbehaviorError(
            "Subgraph must have a valid up_proj attribute",
            action="Please ensure the subgraph contains a valid up_proj layer"
        )
    
    if not hasattr(subgraph, 'down_proj') or subgraph.down_proj is None:
        raise MisbehaviorError(
            "Subgraph must have a valid down_proj attribute",
            action="Please ensure the subgraph contains a valid down_proj layer"
        )
    
    # gate_proj 是可选的
    gate_proj = getattr(subgraph, 'gate_proj', None)

    tmp_device = next(subgraph.up_proj.parameters()).device
    dtype = subgraph.down_proj.weight.dtype
    
    get_logger().debug(f"Using device: {tmp_device}, dtype: {dtype}")

    # 步骤2: 处理激活张量
    # 将激活张量列表转换为单个张量，统一数据类型和设备
    if context.tensors is not None:
        # 过滤掉 None 值并转换张量
        valid_tensors = [
            tensor.to(dtype=dtype).to(tmp_device)
            for tensor in context.tensors
            if tensor is not None and torch.is_tensor(tensor)
        ]
        
        if not valid_tensors:
            get_logger().warning("No valid tensors found in context, skipping smoothing")
            return
        
        # 连接所有有效的张量
        tensors = torch.cat(valid_tensors, dim=0)
        get_logger().debug(f"Successfully processed {len(valid_tensors)} tensors, total shape: {tensors.shape}")
    else:
        get_logger().warning("Context tensors is None, skipping smoothing")
        return
    
    # 重塑张量维度为 (batch_size * sequence_length, hidden_dim)
    act = tensors.view(-1, tensors.shape[-1])
    # 步骤3: 获取激活平滑尺度和权重
    a_scale = context.a_smooth_scale
    if a_scale is None:
        raise MisbehaviorError(
            "Activation smooth scale is None in context",
            action="Please ensure the context contains valid activation smooth scale data"
        )
    
    # 获取下投影层的权重
    fc_weights = subgraph.down_proj.weight
    
    get_logger().debug(f"Activation scale shape: {a_scale.shape}, Weight shape: {fc_weights.shape}")

    try:
        # 步骤4: 确定最优的 alpha 和 beta 参数
        if config.alpha is None or config.beta is None:
            get_logger().debug("Alpha or beta not provided, searching for optimal values...")
            
            # 搜索最优的 alpha 值
            best_alpha, normal_mse_best = search_alpha_beta(
                act, fc_weights, normal_mse_best=1.0, best_alpha=None
            )
            get_logger().debug(f"Found optimal alpha: {best_alpha:.6f}, MSE: {normal_mse_best:.6f}")
            
            # 基于最优 alpha 搜索最优的 beta 值
            best_beta, normal_mse_best = search_alpha_beta(
                act, fc_weights, normal_mse_best=normal_mse_best, best_alpha=best_alpha
            )
            get_logger().debug(f"Found optimal beta: {best_beta:.6f}, final MSE: {normal_mse_best:.6f}")
        else:
            # 使用配置中提供的参数
            best_alpha = config.alpha
            best_beta = config.beta
            get_logger().debug(f"Using provided alpha: {best_alpha}, beta: {best_beta}")
        
    except Exception as e:
        raise UnexpectedError(
            f"Failed to determine optimal alpha/beta parameters: {e}",
            action="Please check the search_alpha_beta function and ensure input tensors are valid"
        ) from e
    
    # 步骤5: 计算权重统计和尺度
    # 计算权重绝对值的最大值，用于归一化
    weight_abs = subgraph.down_proj.weight.abs()
    weight_max = weight_abs.max(dim=0, keepdim=True)[0]
    
    # 计算权重尺度，应用最小阈值限制
    w_scale = weight_max.max(dim=0)[0]
    w_scale = w_scale.to(torch.float32).clamp(min=1e-5).to(dtype=dtype)
    
    get_logger().debug(f"Weight scale shape: {w_scale.shape}, min: {w_scale.min():.6f}, max: {w_scale.max():.6f}")

    
    try:
        # 步骤6: 计算平滑尺度
        # 基于激活尺度、权重尺度和最优参数计算平滑尺度
        scales = compute_smooth_scale(a_scale, w_scale, best_alpha, best_beta)
        get_logger().debug(f"Computed smooth scales shape: {scales.shape}")
        
    except Exception as e:
        raise UnexpectedError(
            f"Failed to compute smooth scales: {e}",
            action="Please check the smooth scale computation functions"
        ) from e
    
    try:
        # 步骤7: 应用平滑到投影层
        # 对下投影层应用平滑尺度
        apply_smooth_scale_shift(subgraph.down_proj, scales.view(1, -1))
        get_logger().debug("Successfully applied smoothing to down projection layer")
        
        # 对上投影层应用平滑尺度的倒数（补偿下投影层的变化）
        apply_smooth_scale_shift(subgraph.up_proj, 1.0 / scales.view(-1, 1))
        get_logger().debug("Successfully applied smoothing to up projection layer")        
        get_logger().debug("Up-Down smoothing completed successfully")
        
    except Exception as e:
        raise UnexpectedError(
            f"Failed to apply smoothing to projection layers: {e}",
            action="Please check the projection layer configurations and ensure they support parameter modification"
        ) from e
    
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(LinearLinearSubgraph, 1), api_name="flex_smooth_quant")
def flex_smooth_impl_LinearLinear(subgraph: Subgraph, config: FlexSmoothQuantConfig, context: SmoothContext) -> None:
    """
    实现 Linear-Linear 子图的平滑处理
    
    该函数用于优化连续线性层的组合，通过计算和应用平滑尺度
    来改善模型的量化性能。
    
    Args:
        subgraph: 包含 linear1 和 linear2 的子图对象
        config: 平滑配置参数，包含 alpha 和 beta 值
        context: 平滑上下文，包含激活统计信息和平滑尺度
    
    Raises:
        MisbehaviorError: 当输入参数无效或处理失败时
        UnexpectedError: 当遇到未预期的错误时
    
    Returns:
        None: 函数直接修改子图参数，无返回值
    """
    
    # 检查子图是否包含必要的线性层
    if not hasattr(subgraph, 'linear1') or subgraph.linear1 is None:
        raise MisbehaviorError(
            "Subgraph must have a valid linear1 attribute",
            action="Please ensure the subgraph contains a valid linear1 layer"
        )
    
    if not hasattr(subgraph, 'linear2') or subgraph.linear2 is None:
        raise MisbehaviorError(
            "Subgraph must have a valid linear2 attribute",
            action="Please ensure the subgraph contains a valid linear2 layer"
        )
    
    # 步骤1: 获取设备和数据类型信息
    # 从 linear1 获取设备信息，从 linear2 获取数据类型
    tmp_device = next(subgraph.linear1.parameters()).device
    dtype = subgraph.linear2.weight.dtype
    
    get_logger().debug(f"Using device: {tmp_device}, dtype: {dtype}")

    # 步骤2: 处理激活张量
    # 将激活张量列表转换为单个张量，统一数据类型和设备
    if context.tensors is not None:
        # 过滤掉 None 值并转换张量
        valid_tensors = [
            tensor.to(dtype=dtype).to(tmp_device)
            for tensor in context.tensors
            if tensor is not None and torch.is_tensor(tensor)
        ]
        
        if not valid_tensors:
            get_logger().warning("No valid tensors found in context, skipping smoothing")
            return
        
        # 连接所有有效的张量
        tensors = torch.cat(valid_tensors, dim=0)
        get_logger().debug(f"Successfully processed {len(valid_tensors)} tensors, total shape: {tensors.shape}")
    else:
        get_logger().warning("Context tensors is None, skipping smoothing")
        return
    
    # 重塑张量维度为 (batch_size * sequence_length, hidden_dim)
    act = tensors.view(-1, tensors.shape[-1])

    # 步骤3: 获取激活平滑尺度和权重
    a_scale = context.a_smooth_scale
    if a_scale is None:
        raise MisbehaviorError(
            "Activation smooth scale is None in context",
            action="Please ensure the context contains valid activation smooth scale data"
        )
    
    # 获取第二个线性层的权重
    fc_weights = subgraph.linear2.weight
    
    get_logger().debug(f"Activation scale shape: {a_scale.shape}, Weight shape: {fc_weights.shape}")

    
    try:
        # 步骤4: 确定最优的 alpha 和 beta 参数
        if config.alpha is None or config.beta is None:
            get_logger().debug("Alpha or beta not provided, searching for optimal values...")
            
            # 搜索最优的 alpha 值
            best_alpha, normal_mse_best = search_alpha_beta(
                act, fc_weights, normal_mse_best=1.0, best_alpha=None
            )
            get_logger().debug(f"Found optimal alpha: {best_alpha:.6f}, MSE: {normal_mse_best:.6f}")
            
            # 基于最优 alpha 搜索最优的 beta 值
            best_beta, normal_mse_best = search_alpha_beta(
                act, fc_weights, normal_mse_best=normal_mse_best, best_alpha=best_alpha
            )
            get_logger().debug(f"Found optimal beta: {best_beta:.6f}, final MSE: {normal_mse_best:.6f}")
        else:
            # 使用配置中提供的参数
            best_alpha = config.alpha
            best_beta = config.beta
            get_logger().debug(f"Using provided alpha: {best_alpha}, beta: {best_beta}")
        
    except Exception as e:
        raise UnexpectedError(
            f"Failed to determine optimal alpha/beta parameters: {e}",
            action="Please check the search_alpha_beta function and ensure input tensors are valid"
        ) from e
    
    try:
        # 步骤5: 计算权重统计和尺度
        # 计算权重绝对值的最大值，用于归一化
        weight_abs = subgraph.linear2.weight.abs()
        weight_max = weight_abs.max(dim=0, keepdim=True)[0]
        
        # 计算权重尺度，应用最小阈值限制
        w_scale = weight_max.max(dim=0)[0]
        w_scale = w_scale.to(torch.float32).clamp(min=1e-5).to(dtype=dtype)
        
        get_logger().debug(f"Weight scale shape: {w_scale.shape}, min: {w_scale.min():.6f}, max: {w_scale.max():.6f}")
        
    except Exception as e:
        raise UnexpectedError(
            f"Failed to compute weight statistics: {e}",
            action="Please check the linear2 layer weights and ensure they are valid"
        ) from e
    
    try:
        # 步骤6: 计算平滑尺度
        # 基于激活尺度、权重尺度和最优参数计算平滑尺度
        scales = compute_smooth_scale(a_scale, w_scale, best_alpha, best_beta)
        get_logger().debug(f"Computed smooth scales shape: {scales.shape}")
        
    except Exception as e:
        raise UnexpectedError(
            f"Failed to compute smooth scales: {e}",
            action="Please check the smooth scale computation functions"
        ) from e
    
    try:
        # 步骤7: 应用平滑到线性层
        # 对第二个线性层应用平滑尺度
        apply_smooth_scale_shift(subgraph.linear2, scales.view(1, -1))
        get_logger().debug("Successfully applied smoothing to linear2 layer")
        
        # 对第一个线性层应用平滑尺度的倒数（补偿第二个线性层的变化）
        apply_smooth_scale_shift(subgraph.linear1, 1.0 / scales.view(-1, 1))
        get_logger().debug("Successfully applied smoothing to linear1 layer")
        
        get_logger().debug("Linear-Linear smoothing completed successfully")
        
    except Exception as e:
        raise UnexpectedError(
            f"Failed to apply smoothing to linear layers: {e}",
            action="Please check the linear layer configurations and ensure they support parameter modification"
        ) from e
    
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(NormLinearSubgraph, 1), api_name="flex_smooth_quant")
def flex_smooth_impl_NormLinear(subgraph: Subgraph, config: FlexSmoothQuantConfig, context: SmoothContext) -> None:
    """
    实现 Norm-Linear 子图的平滑处理
    
    该函数用于优化归一化层（Norm）和线性层（Linear）的组合，
    通过计算和应用平滑尺度来改善模型的量化性能。
    
    Args:
        subgraph: 包含 norm 和 linear 层的子图对象
        config: 平滑配置参数，包含 alpha 和 beta 值
        context: 平滑上下文，包含激活统计信息和平滑尺度
    
    Raises:
        MisbehaviorError: 当输入参数无效或处理失败时
        UnexpectedError: 当遇到未预期的错误时
    
    Returns:
        None: 函数直接修改子图参数，无返回值
    """
    
    # 检查子图是否包含必要的层
    if not hasattr(subgraph, 'norm') or subgraph.norm is None:
        raise MisbehaviorError(
            "Subgraph must have a valid norm attribute",
            action="Please ensure the subgraph contains a valid normalization layer"
        )
    
    if not hasattr(subgraph, 'linears') or subgraph.linears is None:
        raise MisbehaviorError(
            "Subgraph must have a valid linear attribute",
            action="Please ensure the subgraph contains a valid linear layer"
        )
    
    # 步骤1: 获取设备和数据类型信息
    # 从线性层获取设备信息，从归一化层获取数据类型
    tmp_device = next(subgraph.norm.parameters()).device
    dtype = subgraph.linears[0].weight.dtype
    
    get_logger().debug(f"Using device: {tmp_device}, dtype: {dtype}")

    # 步骤2: 处理激活张量
    # 将激活张量列表转换为单个张量，统一数据类型和设备
    if context.tensors is not None:
        # 过滤掉 None 值并转换张量
        valid_tensors = [
            tensor.to(dtype=dtype).to(tmp_device)
            for tensor in context.tensors
            if tensor is not None and torch.is_tensor(tensor)
        ]
        
        if not valid_tensors:
            get_logger().warning("No valid tensors found in context, skipping smoothing")
            return
        
        # 连接所有有效的张量
        tensors = torch.cat(valid_tensors, dim=0)
        get_logger().debug(f"Successfully processed {len(valid_tensors)} tensors, total shape: {tensors.shape}")
    else:
        get_logger().warning("Context tensors is None, skipping smoothing")
        return
    
    # 重塑张量维度为 (batch_size * sequence_length, hidden_dim)
    act = tensors.view(-1, tensors.shape[-1])
    
    try:
        # 步骤3: 获取激活平滑尺度和权重
        a_scale = context.a_smooth_scale
        if a_scale is None:
            raise MisbehaviorError(
                "Activation smooth scale is None in context",
                action="Please ensure the context contains valid activation smooth scale data"
            )
        
        # 获取线性层的权重
        fc_weights = torch.cat([fc.weight for fc in subgraph.linears], dim=0)
        
        get_logger().debug(f"Activation scale shape: {a_scale.shape}, Weight shape: {fc_weights.shape}")
        
    except Exception as e:
        raise MisbehaviorError(
            f"Failed to get activation scale or weights: {e}",
            action="Please check the context data and subgraph layer configurations"
        ) from e
    
    try:
        # 步骤4: 确定最优的 alpha 和 beta 参数
        if config.alpha is None or config.beta is None:
            get_logger().debug("Alpha or beta not provided, searching for optimal values...")
            
            # 搜索最优的 alpha 值
            best_alpha, normal_mse_best = search_alpha_beta(
                act, fc_weights, normal_mse_best=1.0, best_alpha=None
            )
            get_logger().debug(f"Found optimal alpha: {best_alpha:.6f}, MSE: {normal_mse_best:.6f}")
            
            # 基于最优 alpha 搜索最优的 beta 值
            best_beta, normal_mse_best = search_alpha_beta(
                act, fc_weights, normal_mse_best=normal_mse_best, best_alpha=best_alpha
            )
            get_logger().debug(f"Found optimal beta: {best_beta:.6f}, final MSE: {normal_mse_best:.6f}")
        else:
            # 使用配置中提供的参数
            best_alpha = config.alpha
            best_beta = config.beta
            get_logger().debug(f"Using provided alpha: {best_alpha}, beta: {best_beta}")
        
    except Exception as e:
        raise UnexpectedError(
            f"Failed to determine optimal alpha/beta parameters: {e}",
            action="Please check the search_alpha_beta function and ensure input tensors are valid"
        ) from e
    
    weight_stat = []
    for fc in subgraph.linears:
        weight_stat.append(fc.weight.abs().max(dim=0, keepdim=True)[0])

    w_scale = torch.cat(weight_stat, dim=0)
    w_scale = w_scale.max(dim=0)[0].to(torch.float32).clamp(min=1e-5).to(dtype=dtype)
    
    get_logger().debug(f"Weight scale shape: {w_scale.shape}, min: {w_scale.min():.6f}, max: {w_scale.max():.6f}")
    
    try:
        # 步骤6: 计算平滑尺度
        # 基于激活尺度、权重尺度和最优参数计算平滑尺度
        scales = compute_smooth_scale(a_scale, w_scale, best_alpha, best_beta)
        get_logger().debug(f"Computed smooth scales shape: {scales.shape}")
        
        # 为 Norm-Linear 子图准备参数
        # Norm-Linear 通常不需要特殊的注意力头参数
        norm_scales = scales
        
    except Exception as e:
        raise UnexpectedError(
            f"Failed to compute smooth scales: {e}",
            action="Please check the smooth scale computation functions"
        ) from e
    
    try:
        # 步骤7: 应用平滑到层
        # 对归一化层应用平滑尺度
        for fc in subgraph.linears:
            apply_smooth_scale_shift(fc, scales.view(1, -1))
        apply_smooth_scale_shift(subgraph.norm, (1.0 / scales).squeeze())
        get_logger().debug("Successfully applied smoothing to linear layer")
        get_logger().debug("Norm-Linear smoothing completed successfully")
        
    except Exception as e:
        raise UnexpectedError(
            f"Failed to apply smoothing to layers: {e}",
            action="Please check the layer configurations and ensure they support parameter modification"
        ) from e
    
    return