#  -*- coding: utf-8 -*-
#  Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

from functools import partial
from typing import Dict, Callable, List, Any, Literal, Annotated
from pydantic import AfterValidator, Field

import torch
import torch.distributed as dist
import torch.nn as nn
from msmodelslim.core.QAL.qregistry import QABCRegistry
from msmodelslim.core.QAL.qtypes import FlexSmoothQuantConfig
from msmodelslim.core.api import flex_smooth_quant
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.quant.processor.anti_outlier.smooth_interface import FlexSmoothQuantInterface
from msmodelslim.quant.processor.anti_outlier.smooth_base import BaseSmoothProcessor, StatKey, BaseSmoothProcessorConfig
from msmodelslim.quant.processor.base import AutoSessionProcessor
from msmodelslim.utils.validation.value import validate_normalized_value, is_string_list
from msmodelslim.utils.dist import DistHelper
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import get_logger


class FlexSmoothQuantProcessorConfig(BaseSmoothProcessorConfig):
    type: Literal["flex_smooth_quant"] = "flex_smooth_quant"
    alpha: Annotated[float, AfterValidator(validate_normalized_value)] = None
    beta: Annotated[float, AfterValidator(validate_normalized_value)] = None
    enable_subgraph_type: Annotated[List[str], AfterValidator(is_string_list)] = Field(
        default_factory=lambda: ["norm-linear", "linear-linear", "ov", "up-down"]
    )


@QABCRegistry.register(dispatch_key=FlexSmoothQuantProcessorConfig, abc_class=AutoSessionProcessor)
class FlexSmoothQuantProcessor(BaseSmoothProcessor):
    def __init__(self, model: nn.Module, config: FlexSmoothQuantProcessorConfig, adapter: object, **kwargs):
        super().__init__(model, config, adapter)
        if not isinstance(adapter, FlexSmoothQuantInterface):
            raise UnsupportedError(f'{adapter.__class__.__name__} does not support FlexSmooth',
                                   action='Please provide a valid model adapter '
                                          'which implements FlexSmoothQuantInterface')
        self.config = config
        super().validate_parameters()
        self.act_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self.dist_helper = DistHelper(self.model) if dist.is_initialized() else None

        # 存储hook句柄，用于后续删除
        self.hook_handles = {}

    def support_distributed(self) -> bool:
        return True

    def preprocess(self, request: BatchProcessRequest) -> None:
        return super().preprocess(request)

    def _get_stats_hook(self, name: str) -> Callable:
        def stats_hook(name: str, module: nn.Linear, args: tuple, kwargs: dict) -> None:
            tensor = args[0]

            if name not in self.act_stats:
                self.act_stats[name] = {}

            hidden_dim = tensor.shape[-1]
            tensor = tensor.reshape(-1, hidden_dim).detach()  # [N,C]

            if self.dist_helper is not None and self.dist_helper.is_shared(name):
                tensor = torch.cat(self.dist_helper.gather_variable_shapes(tensor), dim=0)

            statis_dict = self.act_stats[name]
            if StatKey.TENSOR not in statis_dict:
                statis_dict[StatKey.TENSOR] = [tensor.to("cpu").reshape(-1, tensor.shape[-1])]
            else:
                statis_dict[StatKey.TENSOR].append(tensor.to("cpu").reshape(-1, tensor.shape[-1]))

            channel_max = torch.max(tensor.abs().detach(), dim=0)[0]

            if StatKey.STAT_KEY_SMOOTH_SCALE in statis_dict:
                statis_dict[StatKey.STAT_KEY_SMOOTH_SCALE] = torch.max(statis_dict[StatKey.STAT_KEY_SMOOTH_SCALE],
                                                                       channel_max)
            else:
                statis_dict[StatKey.STAT_KEY_SMOOTH_SCALE] = channel_max

        return partial(stats_hook, name)

  
    def _apply_smooth_to_subgraph(self, subgraph_obj: Any, linear_modules: List[nn.Module]) -> None:
        """
        通用的平滑应用方法
        
        Args:
            subgraph_obj: 子图对象
            linear_modules: 线性模块列表
        """
        try:
        # 构建SmoothContext
            smooth_context = self._build_smooth_context(linear_modules)

            # 创建平滑配置
            smooth_quant_cfg = FlexSmoothQuantConfig(
                alpha=self.config.alpha,
                beta=self.config.beta
            )

            # 应用平滑
            flex_smooth_quant(subgraph_obj, smooth_quant_cfg, smooth_context)
            get_logger().info(f"[FlexSmoothQuantProcessor] Smooth application completed successfully for subgraph")

        except Exception as e:
            get_logger().error(f"[FlexSmoothQuantProcessor] Failed to apply smooth to subgraph: {e}")
            raise
