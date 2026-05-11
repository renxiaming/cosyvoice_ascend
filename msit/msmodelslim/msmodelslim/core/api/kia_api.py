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

from typing import Type, Tuple

from msmodelslim.core.KIA.manager import KIAManager
from msmodelslim.core.QAL.qregistry import QFuncRegistry
from msmodelslim.core.QAL.qtypes import Subgraph, SmoothContext, IterSmoothConfig, FlexSmoothQuantConfig


@KIAManager.mark_require_version(min_version="1.0.0")
@QFuncRegistry.register_api(dispatch_key=Tuple[Type[Subgraph], int])
def iter_smooth(subgraph: Subgraph, config: IterSmoothConfig, context: SmoothContext) -> None:
    """
    使用iter_smooth算法进行异常值抑制
    
    Args:
        subgraph: 应用iter_smooth算法的子图，支持以下类型：
            NormLinearSubgraph
            LinearLinearSubgraph
            OVSubgraph
            UpDownSubgraph
        config: IterSmooth算法配置
        context: 上下文，用于输入激活的smooth_scale，并记录权重的smooth_scale
        
    Returns:
        None: 无返回值
        
    """
    return QFuncRegistry.dispatch("iter_smooth",
                                  (type(subgraph), config.version),
                                  *(subgraph, config, context))


@KIAManager.mark_require_version(min_version="1.0.0")
@QFuncRegistry.register_api(dispatch_key=Tuple[Type[Subgraph], int])
def flex_smooth_quant(subgraph: Subgraph, config: FlexSmoothQuantConfig, context: SmoothContext) -> None:
    """
    使用iter_smooth算法进行异常值抑制
    
    Args:
        subgraph: 应用flex_smooth_quant算法的子图，支持以下类型：
            NormLinearSubgraph
            LinearLinearSubgraph
            OVSubgraph
            UpDownSubgraph
        config: FlexSmoothQuant算法配置
        context: 上下文，用于输入激活的smooth_scale，并记录权重的smooth_scale
        
    Returns:
        None: 无返回值
        
    """
    return QFuncRegistry.dispatch("flex_smooth_quant",
                                  (type(subgraph), config.version),
                                  *(subgraph, config, context))
