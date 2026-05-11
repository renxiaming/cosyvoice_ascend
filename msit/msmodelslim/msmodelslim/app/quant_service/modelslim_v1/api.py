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

from typing import Optional, List, Any, Literal

from torch import nn as nn

from msmodelslim.app.base.const import PipelineType
from msmodelslim.core.runner.generated_runner import GeneratedRunner
from msmodelslim.model.default import BaseModelAdapter
from msmodelslim.quant.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.quant.processor.memory.load import LoadProcessorConfig
from msmodelslim.utils.exception import InvalidDatasetError, UnsupportedError
from msmodelslim.utils.logging import get_logger


def process_model(
        model: nn.Module,
        process_cfgs: List[AutoProcessorConfig],
        pipeline: Literal[PipelineType.MODEL_WISE, PipelineType.LAYER_WISE] = PipelineType.MODEL_WISE,
        calib_data: Optional[List[Any]] = None,
        adapter: Optional[BaseModelAdapter] = None,
        execution_device: str = "cpu",
        offload_device: str = "cpu",
        post_offload: bool = False,
) -> None:
    model_adapter = adapter

    # 使用BaseModelAdapter中的方法获取pipeline函数
    if model_adapter is not None:
        generated_forward_func, generated_visit_func = model_adapter.get_pipeline_functions(pipeline)
    else:
        # 如果没有adapter，使用默认的model_wise函数
        from msmodelslim.core.runner.generated_runner import GeneratedForwardFuncType, GeneratedVisitFuncType
        from msmodelslim.core.runner.model_wise_forward import model_wise_forward_func, model_wise_visit_func
        from typing import cast

        get_logger().warning("No adapter provided, using default model_wise functions")

        generated_forward_func = cast(GeneratedForwardFuncType, model_wise_forward_func)
        generated_visit_func = cast(GeneratedVisitFuncType, model_wise_visit_func)

    if pipeline == PipelineType.LAYER_WISE:
        process_cfgs.insert(0, LoadProcessorConfig(device=execution_device, mode="load", post_offload=post_offload))
        process_cfgs.append(LoadProcessorConfig(device=offload_device, mode="offload", cleanup=True))

    runner = GeneratedRunner(model, generated_forward_func, generated_visit_func)

    processors = [AutoSessionProcessor.from_config(model, cfg, model_adapter) for cfg in process_cfgs]

    need_kv_cache = any([processor.need_kv_cache() for processor in processors])

    get_logger().info(f"KV cache requirement: {need_kv_cache}")

    try:
        adapter.enable_kv_cache(need_kv_cache)
    except (AttributeError, NotImplementedError) as e:
        if need_kv_cache:
            raise UnsupportedError("Some processors need enable kv cache, but failed to enable kv cache") from e
        else:
            get_logger().warning("Failed to disable kv cache, this may cause more memory usage")

    for processor in processors:
        if not processor.is_data_free() and calib_data is None:
            raise InvalidDatasetError(f"Calib data is needed because {processor} is not data-free")
        stage_data = None if processor.is_data_free() else calib_data
        runner.add_processor(processor, stage_data)

    runner.run()
