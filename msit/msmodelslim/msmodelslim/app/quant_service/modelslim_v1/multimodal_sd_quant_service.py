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
import functools
import os
from pathlib import Path
from typing import Optional

from msmodelslim.utils.cache import load_cached_data, to_device
from msmodelslim.app.quant_service.base import BaseModelAdapter, BaseQuantConfig, BaseQuantService
from msmodelslim.app.quant_service.dataset_interface import DatasetLoaderInterface
from msmodelslim.app.base.const import PipelineType
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import get_logger
from .api import process_model
from .multimodal_sd_quant_config import MultimodalSDModelslimV1QuantConfig


class MultimodalSDModelslimV1QuantService(BaseQuantService):
    backend_name: str = "multimodal_sd_modelslim_v1"

    def __init__(self, dataset_loader: DatasetLoaderInterface):
        super().__init__(dataset_loader)

    def quantize(self, model: BaseModelAdapter, quant_config: BaseQuantConfig, save_path: Optional[Path] = None):
        if not isinstance(model, BaseModelAdapter):
            raise SchemaValidateError("model must be a BaseModelAdapter")
        if not isinstance(quant_config, BaseQuantConfig):
            raise SchemaValidateError("task must be a BaseTask")
        if save_path is not None and not isinstance(save_path, Path):
            raise SchemaValidateError("save_path must be a Path or None")

        return self.quant_process(model, MultimodalSDModelslimV1QuantConfig.from_base(quant_config), save_path)

    def quant_process(self, model: BaseModelAdapter, quant_config: MultimodalSDModelslimV1QuantConfig,
                      save_path: Optional[Path]):

        # 覆盖配置
        model.set_model_args(quant_config.spec.multimodal_sd_config.model_extra['model_config'])
        # 加载模型
        model.load_pipeline()

        get_logger().info(f"==========QUANTIZATION: Prepare Dataset==========")

        config_dump_data_dir = quant_config.spec.multimodal_sd_config.dump_config.dump_data_dir
        if config_dump_data_dir:
            pth_file_path = os.path.join(config_dump_data_dir, "calib_data.pth")
        else:
            # 默认在保存路径下
            pth_file_path = os.path.join(save_path, "calib_data.pth")
        calib_data = load_cached_data(
            pth_file_path=pth_file_path,
            generate_func=model.run_calib_inference,
            model=model.transformer,
            dump_config=quant_config.spec.multimodal_sd_config.dump_config
        )

        calib_data = to_device(calib_data, model.device.value)

        for save_cfg in quant_config.spec.save:
            save_cfg.set_save_directory(save_path)

        final_process_cfg = quant_config.spec.process + quant_config.spec.save

        if not model.support_layer_wise_schedule():
            raise UnsupportedError("Model does not support layer-wise schedule.")

        # 多模态每个模型都不同处理，在模型内实现
        model.apply_quantization(functools.partial(process_model,
                                                   process_cfgs=final_process_cfg,
                                                   calib_data=calib_data,
                                                   pipeline=PipelineType.LAYER_WISE,
                                                   execution_device="npu:0",
                                                   offload_device="meta",
                                                   post_offload=True))

        model.persisted(save_path)

        get_logger().info(f"quantized model: \n {model.transformer}")
        get_logger().info(f"==========QUANTIZATION: END==========")
