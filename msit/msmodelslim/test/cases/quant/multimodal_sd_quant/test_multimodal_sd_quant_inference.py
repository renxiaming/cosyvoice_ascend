# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import pytest
from unittest.mock import Mock
from abc import ABC

from msmodelslim.app.quant_service.modelslim_v1.multimodal_sd_quant_inference import MultimodalSDQuantInference
from msmodelslim.utils.exception import UnsupportedError


class TestMultimodalSDQuantInference:
    """测试MultimodalSDQuantInference抽象基类"""

    def test_multimodal_sd_quant_inference_is_abstract_base_class(self):
        """测试MultimodalSDQuantInference是抽象基类"""
        assert issubclass(MultimodalSDQuantInference, ABC)

    def test_multimodal_sd_quant_inference_cannot_be_instantiated_directly(self):
        """测试MultimodalSDQuantInference不能直接实例化"""
        with pytest.raises(TypeError):
            MultimodalSDQuantInference()

    def test_multimodal_sd_quant_inference_has_required_abstract_methods(self):
        """测试MultimodalSDQuantInference包含必需的抽象方法"""
        # 检查是否存在必需的抽象方法
        assert hasattr(MultimodalSDQuantInference, 'run_calib_inference')
        assert hasattr(MultimodalSDQuantInference, 'apply_quantization')

        # 检查这些方法是否是抽象方法
        assert MultimodalSDQuantInference.run_calib_inference.__isabstractmethod__
        assert MultimodalSDQuantInference.apply_quantization.__isabstractmethod__


class TestConcreteMultimodalSDQuantInference:
    """测试MultimodalSDQuantInference的具体实现类"""

    class ConcreteMultimodalSDQuantInference(MultimodalSDQuantInference):
        """MultimodalSDQuantInference的具体实现类，用于测试"""

        def run_calib_inference(self):
            """实现校准推理方法"""
            return "calib_inference_completed"

        def apply_quantization(self, quant_model_func, quant_config, calib_data):
            """实现量化应用方法"""
            return f"quantization_applied_with_{quant_model_func}_{quant_config}_{calib_data}"

    def test_concrete_class_can_be_instantiated(self):
        """测试具体实现类可以被实例化"""
        instance = self.ConcreteMultimodalSDQuantInference()
        assert isinstance(instance, MultimodalSDQuantInference)

    def test_concrete_class_run_calib_inference_works(self):
        """测试具体实现类的run_calib_inference方法正常工作"""
        instance = self.ConcreteMultimodalSDQuantInference()
        result = instance.run_calib_inference()
        assert result == "calib_inference_completed"

    def test_concrete_class_apply_quantization_works(self):
        """测试具体实现类的apply_quantization方法正常工作"""
        instance = self.ConcreteMultimodalSDQuantInference()
        result = instance.apply_quantization("test_func", "test_config", "test_data")
        assert result == "quantization_applied_with_test_func_test_config_test_data"


class TestMultimodalSDQuantInferenceErrorHandling:
    """测试MultimodalSDQuantInference的错误处理"""

    class IncompleteMultimodalSDQuantInference(MultimodalSDQuantInference):
        """不完整的实现类，用于测试错误处理"""
        pass

    def test_incomplete_class_raises_error_on_instantiation(self):
        """测试不完整的实现类在实例化时抛出错误"""
        with pytest.raises(TypeError):
            self.IncompleteMultimodalSDQuantInference()

    def test_abstract_methods_raise_error(self):
        """验证抽象方法在未被子类实现时会抛出UnsupportedError"""

        # 创建一个完全实现了抽象方法的临时子类
        class ConcreteSubclass(MultimodalSDQuantInference):
            def run_calib_inference(self):
                # 调用父类方法应该抛出异常
                super().run_calib_inference()

            def apply_quantization(self, quant_model_func, quant_config, calib_data):
                # 调用父类方法应该抛出异常
                super().apply_quantization(quant_model_func, quant_config, calib_data)

        instance = ConcreteSubclass()

        # 测试run_calib_inference方法
        with pytest.raises(UnsupportedError) as excinfo:
            instance.run_calib_inference()
        assert "You should implement the run_calib_inference method for ConcreteSubclass" in str(excinfo.value)

        # 测试apply_quantization方法
        with pytest.raises(UnsupportedError) as excinfo:
            instance.apply_quantization(None, None, None)
        assert "You should implement the apply_quantization method for ConcreteSubclass" in str(excinfo.value)


class TestMultimodalSDQuantInferenceMethodSignatures:
    """测试MultimodalSDQuantInference方法的签名"""

    def test_run_calib_inference_method_signature(self):
        """测试run_calib_inference方法的签名"""
        # 获取方法的签名信息
        method = MultimodalSDQuantInference.run_calib_inference

        # 检查方法是否有正确的装饰器标记
        assert hasattr(method, '__isabstractmethod__')
        assert method.__isabstractmethod__ is True

    def test_apply_quantization_method_signature(self):
        """测试apply_quantization方法的签名"""
        # 获取方法的签名信息
        method = MultimodalSDQuantInference.apply_quantization

        # 检查方法是否有正确的装饰器标记
        assert hasattr(method, '__isabstractmethod__')
        assert method.__isabstractmethod__ is True

        # 检查方法是否有正确的文档字符串
        assert "应用模型量化的抽象方法" in method.__doc__
        assert "量化函数" in method.__doc__
        assert "量化配置对象" in method.__doc__
        assert "校准数据" in method.__doc__


class TestMultimodalSDQuantInferenceIntegration:
    """测试MultimodalSDQuantInference的集成场景"""

    class TestableMultimodalSDQuantInference(MultimodalSDQuantInference):
        """可测试的实现类，用于集成测试"""

        def __init__(self):
            self.calib_called = False
            self.quant_called = False
            self.calib_args = None
            self.quant_args = None

        def run_calib_inference(self):
            """记录调用并返回结果"""
            self.calib_called = True
            return {"status": "success", "data": "calib_data"}

        def apply_quantization(self, quant_model_func, quant_config, calib_data):
            """记录调用参数并返回结果"""
            self.quant_called = True
            self.quant_args = (quant_model_func, quant_config, calib_data)
            return {"status": "success", "quantized": True}

    def test_integration_workflow(self):
        """测试完整的集成工作流程"""
        instance = self.TestableMultimodalSDQuantInference()

        # 第一步：运行校准推理
        calib_result = instance.run_calib_inference()
        assert instance.calib_called is True
        assert calib_result["status"] == "success"
        assert calib_result["data"] == "calib_data"

        # 第二步：应用量化
        quant_result = instance.apply_quantization("test_func", "test_config", calib_result["data"])
        assert instance.quant_called is True
        assert instance.quant_args == ("test_func", "test_config", "calib_data")
        assert quant_result["status"] == "success"
        assert quant_result["quantized"] is True

    def test_multiple_instances_independence(self):
        """测试多个实例之间的独立性"""
        instance1 = self.TestableMultimodalSDQuantInference()
        instance2 = self.TestableMultimodalSDQuantInference()

        # 实例1调用方法
        instance1.run_calib_inference()
        instance1.apply_quantization("func1", "config1", "data1")

        # 实例2应该保持未调用状态
        assert instance2.calib_called is False
        assert instance2.quant_called is False

        # 实例2调用方法
        instance2.run_calib_inference()
        instance2.apply_quantization("func2", "config2", "data2")

        # 验证两个实例的参数不同
        assert instance1.quant_args != instance2.quant_args