# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict

from msmodelslim.app.base.const import DeviceType
from msmodelslim.app.base.model import BaseModelAdapter


class MultimodalSDModelAdapter(BaseModelAdapter):
    """多模态模型适配器基类"""

    def __init__(self,
                 model_type: str,
                 model_path: Path,
                 device: DeviceType = DeviceType.NPU,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(model_type, model_path, device, trust_remote_code)
        self.pipeline = None
        self.transformer = None
        self.model_args = None

        self._get_default_model_args(**kwargs)

    @abstractmethod
    def _set_model_args(self, **kwargs) -> None:
        """设置模型参数，由子类实现"""
        raise NotImplementedError

    @abstractmethod
    def _get_default_model_args(self, **kwargs) -> None:
        """加载模型默认参数，由子类实现"""
        raise NotImplementedError

    @abstractmethod
    def _load_pipeline(self, **kwargs) -> None:
        """加载完整pipeline，由子类实现"""
        raise NotImplementedError

    @abstractmethod
    def _get_transformer(self) -> Any:
        """获取需要量化的transformer部分"""
        raise NotImplementedError

    @abstractmethod
    def _check_import_dependency(self):
        """检查加载模型需要的依赖库"""
        raise NotImplementedError

    def get_model_for_quantization(self) -> Any:
        """返回用于量化的模型部分"""
        return self._get_transformer()
