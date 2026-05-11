# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
import re
from pathlib import Path
from typing import Callable, Optional, Type

from msmodelslim.app.base import QuantType
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.exception_decorator import exception_catcher
from msmodelslim.utils.logging import logger_setter, get_logger
from .practice_interface import PracticeManagerInterface
from ..base import BaseQuantConfig, BaseModelAdapter, DeviceType
from ..quant_service import BaseQuantService

DEFAULT_PEDIGREE = 'default'
DEFAULT_CONFIG_ID = 'default'


@logger_setter('msmodelslim.app.naive_quantization')
class NaiveQuantizationApplication:
    def __init__(self,
                 practice_manager: PracticeManagerInterface,
                 quant_service: BaseQuantService,
                 model_factory: Callable[[str], Type[BaseModelAdapter]]):
        self.practice_manager = practice_manager
        self.quant_service = quant_service
        self.model_factory = model_factory

    @staticmethod
    def check_label(label, w_bit, a_bit, use_kv_cache, is_sparse):
        """Check if the label matches the quantization parameters"""
        if label.get('w_bit') != w_bit:
            return False
        if label.get('a_bit') != a_bit:
            return False
        if is_sparse ^ label.get('is_sparse', False):
            return False
        if use_kv_cache ^ label.get('kv_cache', False):
            return False
        return True

    def get_default_practice(self,
                             prompt="No configuration found.",
                             error_msg="The corresponding configuration is not currently supported"
                             ) -> BaseQuantConfig:
        user_input = input(
            prompt +
            " Default configuration will be used. (Enter y to continue, otherwise it will exit): ").strip().lower()[:3]
        if user_input != 'y':
            raise ValueError(error_msg)
        return self.practice_manager.get_config_by_id(DEFAULT_PEDIGREE, DEFAULT_CONFIG_ID)

    def get_best_practice(self,
                          model_type: str,
                          model_pedigree: str,
                          quant_type: Optional[QuantType] = None,
                          config_path: Optional[Path] = None,
                          ) -> BaseQuantConfig:

        # Handle explicit config path
        if config_path is not None:
            config = self.practice_manager.get_config_by_path(config_path)
            if config is None:
                raise ValueError(f"Configuration not found at {config_path}")
            get_logger().info(f"Naive Quant apply config_path: {config_path}")
            return config

        # Handle quant_type matching
        if quant_type is None:
            raise ValueError(f"Quant_type must be provided")

        # Parse quant_type parameters
        match_result = re.match(r'^w(\d+)a(\d+)(c?8?)(s?)$', quant_type.value)
        if not match_result:
            raise ValueError(f"Invalid quant_type format: {quant_type.value}")
        w_bit = int(match_result.group(1))
        a_bit = int(match_result.group(2))
        use_kv_cache = bool(match_result.group(3))
        is_sparse = bool(match_result.group(4))

        for config in self.practice_manager.iter_config(model_pedigree):
            if config.metadata.verified_model_types and model_type not in config.metadata.verified_model_types:
                continue

            if not self.check_label(config.metadata.label, w_bit, a_bit, use_kv_cache, is_sparse):
                continue

            get_logger().info(f"Naive Quant apply config_id: {config.metadata.config_id}")
            return config

        return self.get_default_practice(prompt=f"No matching configuration found for model_type={model_type}.")

    @exception_catcher
    def quant(self,
              model_type: str,
              model_path: Path,
              save_path: Path,
              device: DeviceType = DeviceType.NPU,
              quant_type: Optional[QuantType] = None,
              config_path: Optional[Path] = None,
              trust_remote_code: bool = False):
        """
        Run the naive quantization application.
        Args:
            model_type: str, the type of the model
            model_path: str, the path of the model
            save_path: str, the path to save the quantized model
            device: str, the device to run the quantization, only 'npu' and 'cpu' are supported
            quant_type: Optional[str], the type of quantization, config_path and quant_type only one can be provided
            config_path: Optional[str], the path to config file, config_path and quant_type only one can be provided
            trust_remote_code: bool, whether to trust the remote code
        """
        if not isinstance(model_type, str):
            raise SchemaValidateError(f"model_type must be a string, but got {type(model_type)}")
        if not isinstance(model_path, Path):
            raise SchemaValidateError(f"model_path must be a Path, but got {type(model_path)}")
        if not isinstance(save_path, Path):
            raise SchemaValidateError(f"save_path must be a Path, but got {type(save_path)}")
        if not isinstance(device, DeviceType):
            raise SchemaValidateError(f"device must be a DeviceType")
        if not ((quant_type is None) ^ (config_path is None)):
            raise SchemaValidateError(f"quant_type and config_path only one can be provided")
        if quant_type is not None and not isinstance(quant_type, QuantType):
            raise SchemaValidateError(f"quant_type must be a QuantType")
        if config_path is not None and not isinstance(config_path, Path):
            raise SchemaValidateError(f"config_path must be a Path, but got {type(config_path)}")
        if not isinstance(trust_remote_code, bool):
            raise SchemaValidateError(f"trust_remote_code must be a bool")

        get_logger().info(f'quantization with following parameters:')
        get_logger().info(f"model_type: {model_type}")
        get_logger().info(f"model_path: {model_path}")
        get_logger().info(f"save_path: {save_path}")
        get_logger().info(f"device: {device}")
        if quant_type is not None:
            get_logger().info(f"quant_type: {quant_type}")
        if config_path is not None:
            get_logger().info(f"config_path: {config_path}")
        get_logger().info(f"trust_remote_code: {trust_remote_code}")

        self._quant(model_type, model_path, save_path, device, quant_type, config_path, trust_remote_code)

    def _quant(
            self,
            model_type: str,
            model_path: Path,
            save_path: Path,
            device: DeviceType = DeviceType.NPU,
            quant_type: Optional[QuantType] = None,
            config_path: Optional[Path] = None,
            trust_remote_code: bool = False
    ):
        get_logger().info(f"===========LOAD MODEL===========")
        model = self.model_factory(model_type)(
            model_type=model_type,
            ori_path=model_path,
            device=device,
            trust_remote_code=trust_remote_code)
        get_logger().info(f"Load model {model_type} from {model_path} to {device} success.")

        get_logger().info(f"===========GET BEST PRACTICE===========")
        quant_config = self.get_best_practice(
            model_type=model.type,
            model_pedigree=model.pedigree,
            quant_type=quant_type,
            config_path=config_path
        )
        get_logger().info(f"Get best practice {quant_config.metadata.config_id} success.")

        get_logger().info(f"===========QUANTIZE MODEL===========")
        self.quant_service.quantize(model=model, quant_config=quant_config, save_path=save_path)
        get_logger().info(f"===========SUCCESS===========")
