# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from msmodelslim.app.quant_service.modelslim_v1.multimodal_sd_quant_service import MultimodalSDModelslimV1QuantService
from msmodelslim.app.quant_service.modelslim_v1.multimodal_sd_quant_config import (
    MultimodalSDModelslimV1QuantConfig, MultimodalSDServiceConfig, MultimodalSDConfig, DumpConfig
)
from msmodelslim.app.quant_service.base import BaseModelAdapter, BaseQuantConfig
from msmodelslim.app.quant_service.dataset_interface import DatasetLoaderInterface
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError


class TestMultimodalSDModelslimV1QuantService:
    """测试MultimodalSDModelslimV1QuantService服务类"""

    @pytest.fixture
    def mock_dataset_loader(self):
        """创建模拟的数据集加载器"""
        return Mock(spec=DatasetLoaderInterface)

    @pytest.fixture
    def mock_model_adapter(self):
        """创建模拟的模型适配器"""
        mock_adapter = Mock(spec=BaseModelAdapter)
        mock_adapter.transformer = Mock()
        mock_adapter.device = Mock()
        mock_adapter.device.value = "npu:0"
        mock_adapter.support_layer_wise_schedule.return_value = True
        mock_adapter.set_model_args = Mock()
        mock_adapter.load_pipeline = Mock()
        mock_adapter.apply_quantization = Mock()
        mock_adapter.persisted = Mock()
        mock_adapter.run_calib_inference = Mock(return_value={"calib": "data"})
        return mock_adapter

    @pytest.fixture
    def mock_quant_config(self):
        """创建模拟的量化配置"""
        dump_config = DumpConfig(capture_mode="args", dump_data_dir="/test/dump")
        # 使用字典方式创建配置，包含额外参数
        multimodal_config_dict = {
            "dump_config": dump_config,
            "model_config": {"test_param": "test_value"}
        }
        multimodal_config = MultimodalSDConfig(**multimodal_config_dict)

        service_config = MultimodalSDServiceConfig(multimodal_sd_config=multimodal_config)
        service_config.save = [Mock()]
        service_config.process = [Mock()]

        # 设置save配置的set_save_directory方法
        for save_cfg in service_config.save:
            save_cfg.set_save_directory = Mock()

        return MultimodalSDModelslimV1QuantConfig(
            apiversion="v1",
            metadata={"name": "test"},
            spec=service_config
        )

    @pytest.fixture
    def mock_base_quant_config(self):
        """创建模拟的基础量化配置"""
        mock_config = Mock(spec=BaseQuantConfig)
        mock_config.apiversion = "v1"
        mock_config.metadata = {"name": "base_test"}
        mock_config.spec = {
            "multimodal_sd_config": {
                "dump_config": {"capture_mode": "args", "dump_data_dir": "/base/test"}
            }
        }
        return mock_config

    def test_service_initialization(self, mock_dataset_loader):
        """测试服务初始化"""
        service = MultimodalSDModelslimV1QuantService(mock_dataset_loader)
        assert service.backend_name == "multimodal_sd_modelslim_v1"
        assert service.dataset_loader is mock_dataset_loader

    def test_quantize_with_valid_parameters(self, mock_dataset_loader, mock_model_adapter, mock_base_quant_config):
        """测试使用有效参数调用quantize方法"""
        service = MultimodalSDModelslimV1QuantService(mock_dataset_loader)

        # Mock from_base方法
        with patch(
                'msmodelslim.app.quant_service.modelslim_v1.multimodal_sd_quant_service'
                '.MultimodalSDModelslimV1QuantConfig.from_base') as mock_from_base:
            mock_quant_config = Mock()
            mock_from_base.return_value = mock_quant_config

            # Mock quant_process方法
            with patch.object(service, 'quant_process') as mock_quant_process:
                mock_quant_process.return_value = "quantization_result"

                result = service.quantize(mock_model_adapter, mock_base_quant_config, Path("/test/save"))

                # 验证调用
                mock_from_base.assert_called_once_with(mock_base_quant_config)
                mock_quant_process.assert_called_once_with(mock_model_adapter, mock_quant_config, Path("/test/save"))
                assert result == "quantization_result"

    def test_quantize_with_invalid_model_type(self, mock_dataset_loader, mock_base_quant_config):
        """测试使用无效模型类型调用quantize方法"""
        service = MultimodalSDModelslimV1QuantService(mock_dataset_loader)
        invalid_model = "not_a_model_adapter"

        with pytest.raises(SchemaValidateError, match="model must be a BaseModelAdapter"):
            service.quantize(invalid_model, mock_base_quant_config)

    def test_quantize_with_invalid_quant_config_type(self, mock_dataset_loader, mock_model_adapter):
        """测试使用无效量化配置类型调用quantize方法"""
        service = MultimodalSDModelslimV1QuantService(mock_dataset_loader)
        invalid_config = "not_a_quant_config"

        with pytest.raises(SchemaValidateError, match="task must be a BaseTask"):
            service.quantize(mock_model_adapter, invalid_config)

    def test_quantize_with_invalid_save_path_type(self, mock_dataset_loader, mock_model_adapter,
                                                  mock_base_quant_config):
        """测试使用无效保存路径类型调用quantize方法"""
        service = MultimodalSDModelslimV1QuantService(mock_dataset_loader)
        invalid_save_path = "/invalid/path"  # 字符串而不是Path对象

        with pytest.raises(SchemaValidateError, match="save_path must be a Path or None"):
            service.quantize(mock_model_adapter, mock_base_quant_config, invalid_save_path)

    def test_quantize_with_none_save_path(self, mock_dataset_loader, mock_model_adapter, mock_base_quant_config):
        """测试使用None作为保存路径调用quantize方法"""
        service = MultimodalSDModelslimV1QuantService(mock_dataset_loader)

        with patch(
                'msmodelslim.app.quant_service.modelslim_v1.multimodal_sd_quant_service'
                '.MultimodalSDModelslimV1QuantConfig.from_base') as mock_from_base:
            mock_quant_config = Mock()
            mock_from_base.return_value = mock_quant_config

            with patch.object(service, 'quant_process') as mock_quant_process:
                mock_quant_process.return_value = "quantization_result"

                result = service.quantize(mock_model_adapter, mock_base_quant_config, None)

                # 验证调用
                mock_quant_process.assert_called_once_with(mock_model_adapter, mock_quant_config, None)
                assert result == "quantization_result"


class TestMultimodalSDModelslimV1QuantServiceQuantProcess:
    """测试MultimodalSDModelslimV1QuantService的quant_process方法"""

    @pytest.fixture
    def mock_dataset_loader(self):
        """创建模拟的数据集加载器"""
        return Mock(spec=DatasetLoaderInterface)

    @pytest.fixture
    def mock_model_adapter(self):
        """创建模拟的模型适配器"""
        mock_adapter = Mock(spec=BaseModelAdapter)
        mock_adapter.transformer = Mock()
        mock_adapter.device = Mock()
        mock_adapter.device.value = "npu:0"
        mock_adapter.support_layer_wise_schedule.return_value = True
        mock_adapter.set_model_args = Mock()
        mock_adapter.load_pipeline = Mock()
        mock_adapter.apply_quantization = Mock()
        mock_adapter.persisted = Mock()
        mock_adapter.run_calib_inference = Mock(return_value={"calib": "data"})
        return mock_adapter

    @pytest.fixture
    def mock_quant_config(self):
        """创建模拟的量化配置"""
        dump_config = DumpConfig(capture_mode="args", dump_data_dir="/test/dump")
        # 使用字典方式创建配置，包含额外参数
        multimodal_config_dict = {
            "dump_config": dump_config,
            "model_config": {"test_param": "test_value"}
        }
        multimodal_config = MultimodalSDConfig(**multimodal_config_dict)

        service_config = MultimodalSDServiceConfig(multimodal_sd_config=multimodal_config)
        service_config.save = [Mock()]
        service_config.process = [Mock()]

        # 设置save配置的set_save_directory方法
        for save_cfg in service_config.save:
            save_cfg.set_save_directory = Mock()

        return MultimodalSDModelslimV1QuantConfig(
            apiversion="v1",
            metadata={"name": "test"},
            spec=service_config
        )

    def test_quant_process_basic_flow(self, mock_dataset_loader, mock_model_adapter, mock_quant_config):
        """测试quant_process方法的基本流程"""
        service = MultimodalSDModelslimV1QuantService(mock_dataset_loader)
        save_path = Path("/test/save")

        # Mock load_cached_data
        with patch(
                'msmodelslim.app.quant_service.modelslim_v1.multimodal_sd_quant_service.load_cached_data'
        ) as mock_load_cached:
            mock_calib_data = {"test": "calib_data"}
            mock_load_cached.return_value = mock_calib_data

            # Mock to_device
            with patch(
                    'msmodelslim.app.quant_service.modelslim_v1.multimodal_sd_quant_service.to_device'
            ) as mock_to_device:
                mock_to_device.return_value = mock_calib_data

                # Mock process_model
                with patch(
                        'msmodelslim.app.quant_service.modelslim_v1.multimodal_sd_quant_service.process_model'
                ) as mock_process_model:
                    service.quant_process(mock_model_adapter, mock_quant_config, save_path)

                    # 验证基本流程
                    mock_model_adapter.set_model_args.assert_called_once()
                    mock_model_adapter.load_pipeline.assert_called_once()
                    mock_load_cached.assert_called_once()
                    mock_to_device.assert_called_once()

                    # 验证save配置设置
                    for save_cfg in mock_quant_config.spec.save:
                        save_cfg.set_save_directory.assert_called_once_with(save_path)

                    # 验证apply_quantization调用
                    mock_model_adapter.apply_quantization.assert_called_once()
                    mock_model_adapter.persisted.assert_called_once_with(save_path)

    def test_quant_process_with_custom_dump_data_dir(self, mock_dataset_loader, mock_model_adapter):
        """测试使用自定义dump数据目录的quant_process方法"""
        service = MultimodalSDModelslimV1QuantService(mock_dataset_loader)
        save_path = Path("/test/save")

        # 创建带有自定义dump_data_dir的配置
        dump_config = DumpConfig(capture_mode="args", dump_data_dir="/custom/dump")
        # 使用字典方式创建配置，包含额外参数
        multimodal_config_dict = {
            "dump_config": dump_config,
            "model_config": {"test_param": "test_value"}
        }
        multimodal_config = MultimodalSDConfig(**multimodal_config_dict)

        service_config = MultimodalSDServiceConfig(multimodal_sd_config=multimodal_config)
        service_config.save = [Mock()]
        service_config.process = [Mock()]

        for save_cfg in service_config.save:
            save_cfg.set_save_directory = Mock()

        quant_config = MultimodalSDModelslimV1QuantConfig(
            apiversion="v1",
            metadata={"name": "test"},
            spec=service_config
        )

        # Mock load_cached_data
        with patch(
                'msmodelslim.app.quant_service.modelslim_v1.multimodal_sd_quant_service.load_cached_data'
        ) as mock_load_cached:
            mock_calib_data = {"test": "calib_data"}
            mock_load_cached.return_value = mock_calib_data

            with patch(
                    'msmodelslim.app.quant_service.modelslim_v1.multimodal_sd_quant_service.to_device'
            ) as mock_to_device:
                mock_to_device.return_value = mock_calib_data

                with patch(
                        'msmodelslim.app.quant_service.modelslim_v1.multimodal_sd_quant_service.process_model'
                ) as mock_process_model:
                    service.quant_process(mock_model_adapter, quant_config, save_path)

                    # 验证使用了自定义的dump_data_dir
                    # 注意：在Windows系统上，路径分隔符可能是反斜杠
                    expected_pth_path = "/custom/dump/calib_data.pth"
                    mock_load_cached.assert_called_once()
                    call_args = mock_load_cached.call_args
                    actual_path = call_args[1]['pth_file_path']
                    # 使用os.path.normpath来标准化路径，然后比较
                    import os
                    assert os.path.normpath(actual_path) == os.path.normpath(expected_pth_path)

    def test_quant_process_with_default_dump_data_dir(self, mock_dataset_loader, mock_model_adapter):
        """测试使用默认dump数据目录的quant_process方法"""
        service = MultimodalSDModelslimV1QuantService(mock_dataset_loader)
        save_path = Path("/test/save")

        # 创建没有dump_data_dir的配置（使用默认值）
        dump_config = DumpConfig(capture_mode="args", dump_data_dir="")
        # 使用字典方式创建配置，包含额外参数
        multimodal_config_dict = {
            "dump_config": dump_config,
            "model_config": {"test_param": "test_value"}
        }
        multimodal_config = MultimodalSDConfig(**multimodal_config_dict)

        service_config = MultimodalSDServiceConfig(multimodal_sd_config=multimodal_config)
        service_config.save = [Mock()]
        service_config.process = [Mock()]

        for save_cfg in service_config.save:
            save_cfg.set_save_directory = Mock()

        quant_config = MultimodalSDModelslimV1QuantConfig(
            apiversion="v1",
            metadata={"name": "test"},
            spec=service_config
        )

        # Mock load_cached_data
        with patch(
                'msmodelslim.app.quant_service.modelslim_v1.multimodal_sd_quant_service.load_cached_data'
        ) as mock_load_cached:
            mock_calib_data = {"test": "calib_data"}
            mock_load_cached.return_value = mock_calib_data

            with patch(
                    'msmodelslim.app.quant_service.modelslim_v1.multimodal_sd_quant_service.to_device'
            ) as mock_to_device:
                mock_to_device.return_value = mock_calib_data

                with patch(
                        'msmodelslim.app.quant_service.modelslim_v1.multimodal_sd_quant_service.process_model'
                ) as mock_process_model:
                    service.quant_process(mock_model_adapter, quant_config, save_path)

                    # 验证使用了默认的dump_data_dir（save_path）
                    expected_pth_path = str(save_path / "calib_data.pth")
                    mock_load_cached.assert_called_once()
                    call_args = mock_load_cached.call_args
                    actual_path = call_args[1]['pth_file_path']
                    # 使用os.path.normpath来标准化路径，然后比较
                    import os
                    assert os.path.normpath(actual_path) == os.path.normpath(expected_pth_path)

    def test_quant_process_model_does_not_support_layer_wise_schedule(self, mock_dataset_loader, mock_model_adapter,
                                                                      mock_quant_config):
        """测试模型不支持层级别调度时的错误处理"""
        service = MultimodalSDModelslimV1QuantService(mock_dataset_loader)
        save_path = Path("/test/save")

        # 设置模型不支持层级别调度
        mock_model_adapter.support_layer_wise_schedule.return_value = False

        # Mock load_cached_data以避免文件系统操作
        with patch(
                'msmodelslim.app.quant_service.modelslim_v1.multimodal_sd_quant_service.load_cached_data'
        ) as mock_load_cached:
            mock_calib_data = {"test": "calib_data"}
            mock_load_cached.return_value = mock_calib_data

            with pytest.raises(UnsupportedError, match="Model does not support layer-wise schedule"):
                service.quant_process(mock_model_adapter, mock_quant_config, save_path)

    def test_quant_process_with_empty_save_and_process_configs(self, mock_dataset_loader, mock_model_adapter):
        """测试使用空的save和process配置的quant_process方法"""
        service = MultimodalSDModelslimV1QuantService(mock_dataset_loader)
        save_path = Path("/test/save")

        # 创建空的save和process配置
        dump_config = DumpConfig(capture_mode="args", dump_data_dir="/test/dump")
        # 使用字典方式创建配置，包含额外参数
        multimodal_config_dict = {
            "dump_config": dump_config,
            "model_config": {"test_param": "test_value"}
        }
        multimodal_config = MultimodalSDConfig(**multimodal_config_dict)

        service_config = MultimodalSDServiceConfig(multimodal_sd_config=multimodal_config)
        service_config.save = []  # 空的save配置
        service_config.process = []  # 空的process配置

        quant_config = MultimodalSDModelslimV1QuantConfig(
            apiversion="v1",
            metadata={"name": "test"},
            spec=service_config
        )

        # Mock load_cached_data
        with patch(
                'msmodelslim.app.quant_service.modelslim_v1.multimodal_sd_quant_service.load_cached_data'
        ) as mock_load_cached:
            mock_calib_data = {"test": "calib_data"}
            mock_load_cached.return_value = mock_calib_data

            service.quant_process(mock_model_adapter, quant_config, save_path)

            # 验证即使配置为空也能正常处理
            mock_model_adapter.apply_quantization.assert_called_once()
            mock_model_adapter.persisted.assert_called_once_with(save_path)