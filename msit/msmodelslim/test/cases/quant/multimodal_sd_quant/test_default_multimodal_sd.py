# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest

import pytest
import torch
from abc import ABC
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

from msmodelslim.app.base.const import DeviceType
from msmodelslim.model.default_multimodal_sd import MultimodalSDModelAdapter, BaseModelAdapter


class TestMultimodalSDModelAdapterAbstractClass:
    """测试MultimodalSDModelAdapter抽象类"""

    def test_cannot_instantiate_directly(self):
        """测试不能直接实例化抽象类"""
        with pytest.raises(TypeError) as exc_info:
            MultimodalSDModelAdapter()
        
        # 只检查核心错误信息，忽略抽象方法列表（不同Python版本可能不同）
        assert "Can't instantiate abstract class MultimodalSDModelAdapter" in str(exc_info.value)


class TestMultimodalSDModelAdapterConstructor(unittest.TestCase):
    """测试MultimodalSDModelAdapter构造函数功能"""

    def setUp(self):
        """创建一个实现了所有抽象方法的测试子类"""

        class ConcreteAdapter(MultimodalSDModelAdapter, ABC):
            def _set_model_args(self, **kwargs) -> None:
                # 调用父类方法应该抛出异常
                super()._set_model_args()

            # 该方法必须实现，不然实例化不了了
            def _get_default_model_args(self, **kwargs) -> None:
                pass

            def _load_pipeline(self, **kwargs) -> None:
                # 调用父类方法应该抛出异常
                super()._load_pipeline()

            def _get_transformer(self) -> Any:
                # 调用父类方法应该抛出异常
                super()._get_transformer()

            def _check_import_dependency(self):
                # 调用父类方法应该抛出异常
                super()._check_import_dependency()

            def _get_model_pedigree(self) -> str:
                pass

            def _load_config(self):
                pass

            def _load_tokenizer(self, trust_remote_code=False):
                pass

            def _load_model(self, device_map=None, torch_dtype=None):
                pass

            def _load_hook(self) -> None:
                pass

            def _persist_hook(self) -> None:
                pass

            def _initialize_torch_dtype(self):
                return torch.float32

        self.ConcreteAdapter = ConcreteAdapter
        self.model_type = "test_sd"
        self.model_path = Path("/test/model/path")
        self.device = DeviceType.CPU
        self.trust_remote_code = True

    def test_constructor_initializes_instance_variables(self):
        """验证构造函数正确初始化实例变量"""
        adapter = self.ConcreteAdapter(
            model_type=self.model_type,
            model_path=self.model_path,
            device=self.device
        )

        self.assertIsNone(adapter.pipeline)
        self.assertIsNone(adapter.transformer)
        self.assertIsNone(adapter.model_args)
        self.assertEqual(adapter.type, self.model_type)
        self.assertEqual(adapter.ori, self.model_path)
        self.assertEqual(adapter.device, self.device)

    def test_abstract_methods_raise_error(self):
        """验证抽象方法在未被子类实现时会抛出UnsupportedError"""

        adapter = self.ConcreteAdapter(
            model_type=self.model_type,
            model_path=self.model_path,
            device=self.device
        )

        # 验证调用该方法时抛出NotImplementedError
        with self.assertRaises(NotImplementedError):
            adapter._set_model_args()

        # 验证调用该方法时抛出NotImplementedError
        with self.assertRaises(NotImplementedError):
            adapter._load_pipeline()

        # 验证调用该方法时抛出NotImplementedError
        with self.assertRaises(NotImplementedError):
            adapter._get_transformer()

        # 验证调用该方法时抛出NotImplementedError
        with self.assertRaises(NotImplementedError):
            adapter._check_import_dependency()

        # 验证调用该方法时抛出NotImplementedError
        with self.assertRaises(NotImplementedError):
            adapter.get_model_for_quantization()

    def test_abstract_method_get_default_model_args_raise_error(self):
        """验证抽象方法在未被子类实现时会抛出UnsupportedError"""

        # 创建一个完全实现了抽象方法的临时子类
        class ConcreteSubclass(MultimodalSDModelAdapter, ABC):
            def _get_model_pedigree(self) -> str:
                pass

            def _load_config(self):
                pass

            def _load_tokenizer(self, trust_remote_code=False):
                pass

            def _load_model(self, device_map=None, torch_dtype=None):
                pass

            def _load_hook(self) -> None:
                pass

            def _persist_hook(self) -> None:
                pass

            def _initialize_torch_dtype(self):
                return torch.float32

            def _set_model_args(self):
                pass

            def _get_default_model_args(self):
                # 调用父类方法应该抛出异常
                super()._get_default_model_args()

            def _load_pipeline(self):
                pass

            def _get_transformer(self):
                pass

            def _check_import_dependency(self):
                pass

        # 验证 _get_default_model_args 未实现时抛出NotImplementedError
        with self.assertRaises(NotImplementedError):
            instance = ConcreteSubclass(
                model_type="test",
                model_path=Path("test/path"),
                device=DeviceType.CPU
            )


if __name__ == '__main__':
    unittest.main()