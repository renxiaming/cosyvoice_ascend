# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import safe_copy_file
from .quant_config import ModelslimV0QuantConfig
from ..base import BaseQuantService
from ..dataset_interface import DatasetLoaderInterface
from ...base import DeviceType, BaseModelAdapter, BaseQuantConfig


def get_padding_data(tokenizer, calib_list, device_type):
    """
    Get the padding data for the calibration.
    """
    calib_dataset = []
    max_len = 0
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt', add_special_tokens=False)
        calib_dataset.append(
            inputs.data['input_ids'].to(device_type)
        )
        max_len = max(max_len, inputs.data['input_ids'].size(1))
    new_calib_dataset = []
    for inputs in calib_dataset:
        new_inputs = F.pad(inputs, (0, max_len - inputs.size(1)), value=0)
        new_calib_dataset.append(new_inputs)
    return [torch.cat(new_calib_dataset)]


def get_batch_tokenized_data(model_tokenizer, calib_list, batch_size, device="npu"):
    """
    Get the batch tokenized data for the calibration.
    """
    calib_dataset = []
    calib_list = [calib_list[i:i + batch_size] for i in range(0, len(calib_list), batch_size)]
    for calib_data in calib_list:
        tmp = get_padding_data(model_tokenizer, calib_data, device)
        calib_dataset.append(tmp)
    return calib_dataset


def get_tokenized_data(tokenizer, calib_list, device,
                       input_ids_name='input_ids',
                       attention_mask_name='attention_mask'):
    tokenized_data = []
    for input_text in calib_list:
        inputs = tokenizer(input_text, return_tensors='pt', padding=True).to(device)
        tokenized_data.append(
            [inputs.data[input_ids_name], inputs.data[attention_mask_name]])
    return tokenized_data


def copy_files(input_path, output_path):
    """
    复制模型配置文件
    @param input_path: 源目录
    @param output_path: 目标目录
    """
    for file in os.listdir(input_path):
        if not any((file.endswith(subfix) for subfix in ['.json', '.py'])):
            continue

        if any((file.endswith(subfix) for subfix in ['index.json'])):
            continue

        ori_file = os.path.join(input_path, file)
        dest_file = os.path.join(output_path, file)
        safe_copy_file(src_path=ori_file, dest_path=dest_file)
        os.chmod(dest_file, int("600", 8))


@logger_setter('msmodelslim.app.quant_service.modelslim_v0')
class ModelslimV0QuantService(BaseQuantService):
    backend_name: str = "modelslim_v0"

    def __init__(self, dataset_loader: DatasetLoaderInterface):
        super().__init__(dataset_loader)

    def quantize(self, model: BaseModelAdapter, quant_config: BaseQuantConfig, save_path: Optional[Path] = None):
        if not isinstance(model, BaseModelAdapter):
            raise SchemaValidateError("model must be a BaseModelAdapter",
                                      action='Please make sure the model is a BaseModelAdapter')
        if not isinstance(quant_config, BaseQuantConfig):
            raise SchemaValidateError("task must be a BaseTask",
                                      action='Please make sure the task is a BaseTask')
        if save_path is not None and not isinstance(save_path, Path):
            raise SchemaValidateError("save_path must be a Path or None",
                                      action='Please make sure the save_path is a Path or None')

        return self.quant_process(model, ModelslimV0QuantConfig.from_base(quant_config), save_path)

    def quant_process(self, model: BaseModelAdapter, quant_config: ModelslimV0QuantConfig, save_path: Optional[Path]):
        # init
        if model.device == DeviceType.NPU:
            # 如果使用npu进行量化需开启二进制编译，避免在线编译算子
            torch.npu.set_compile_mode(jit_compile=False)

        # handle dataset
        get_logger().info(f"==========QUANTIZATION: Prepare Dataset==========")
        anti_dataset = quant_config.spec.anti_dataset
        calib_dataset = quant_config.spec.calib_dataset
        batch_size = quant_config.spec.batch_size

        calib_data = None

        if calib_dataset is not None:
            get_logger().info(f"prepare calib_data from {calib_dataset}")
            dataset = self.dataset_loader.get_dataset_by_name(calib_dataset)
            calib_data = get_tokenized_data(model.tokenizer, dataset, device=model.device.value)
            get_logger().info(f"prepare calib_data success")

        anti_data = calib_data

        if anti_dataset is not None:
            get_logger().info(f"prepare anti_data from {anti_dataset}")
            dataset = self.dataset_loader.get_dataset_by_name(anti_dataset)
            anti_data = get_batch_tokenized_data(model.tokenizer, dataset, batch_size, device=model.device.value)
            get_logger().info(f"prepare anti_data success")

        # anti outlier
        if quant_config.spec.anti_cfg is not None:
            get_logger().info(f"==========QUANTIZATION: ANTI OUTLIER==========")
            get_logger().debug(f"anti outlier config: {quant_config.spec.anti_cfg}")
            anti_cfg = AntiOutlierConfig(dev_type=model.device.value, **quant_config.spec.anti_cfg)
            anti_outlier = AntiOutlier(
                model=model.model,
                calib_data=anti_data,
                cfg=anti_cfg,
                **quant_config.spec.anti_params)
            anti_outlier.process()
            get_logger().info(f"anti outlier success")

        # quantization
        get_logger().info(f"==========QUANTIZATION: CALIBRATION==========")
        get_logger().debug(f"calibration config: {quant_config.spec.calib_cfg}")

        use_fa_quant = bool(quant_config.spec.calib_cfg.pop('use_fa_quant', False))
        fa_amp = quant_config.spec.calib_cfg.pop('fa_amp', 0)
        calib_cfg = QuantConfig(dev_type=model.device.value, **quant_config.spec.calib_cfg)
        if use_fa_quant:
            calib_cfg = calib_cfg.fa_quant(fa_amp=fa_amp)

        calibrator = Calibrator(
            model=model.model,
            cfg=calib_cfg,
            calib_data=calib_data,
            **quant_config.spec.calib_params)
        calibrator.run()
        get_logger().info(f"calibration success")

        # persist stage
        if not save_path:
            get_logger().warning(f"save_path is not provided, skip persist")
            get_logger().info(f"==========QUANTIZATION: END==========")
            return

        # persist
        get_logger().info(f"==========QUANTIZATION: PERSIST==========")
        calibrator.save(
            output_path=str(save_path),
            json_name='quant_model_description.json',
            save_type=['ascendV1'],
            **quant_config.spec.calib_save_params
        )
        model.persisted(save_path)

        # copy .json and .py files
        copy_files(str(model.ori), str(model.path))

        get_logger().info(f"==========QUANTIZATION: END==========")
