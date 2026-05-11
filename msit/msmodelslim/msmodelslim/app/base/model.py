# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast, Tuple, Literal

import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel, PretrainedConfig

from msmodelslim.app.base.const import DeviceType, PipelineType
from msmodelslim.core.runner.generated_runner import GeneratedForwardFuncType, GeneratedVisitFuncType
from msmodelslim.core.runner.layer_wise_forward import transformers_generated_forward_func, \
    generated_decoder_layer_visit_func
from msmodelslim.core.runner.model_wise_forward import model_wise_forward_func, model_wise_visit_func


class BaseModelAdapter(ABC):
    def __init__(self, model_type: str, ori_path: Path, device: DeviceType = DeviceType.NPU, trust_remote_code=False):
        self._type = model_type
        self._ori_path = ori_path
        self._device = device
        self._path = None

        self._trust_remote_code = trust_remote_code

        self._pedigree = self._get_model_pedigree()
        self._config = self._load_config()
        self._tokenizer = self._load_tokenizer(trust_remote_code=trust_remote_code)
        self._model = None

        # default device_map based on device type
        self._device_map = 'cpu' if device is DeviceType.CPU else 'auto'
        self._torch_dtype = self._initialize_torch_dtype()

    @property
    def ori(self) -> Path:
        """
        The original path of the model.
        """
        return self._ori_path

    @property
    def path(self) -> Path:
        """
        The path of the quantized model.
        """
        if self._path is None:
            raise ValueError("Model is not persisted. Please call persisted() to persist the model.")
        return self._path

    @property
    def device(self) -> DeviceType:
        """
        The device of the model.
        """
        return self._device

    @property
    def config(self) -> PretrainedConfig:
        """
        The config of the model.
        """
        return self._config

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """
        The tokenizer of the model.
        """
        return self._tokenizer

    @property
    def model(self) -> PreTrainedModel:
        """
        The model of the model.
        """
        if self._model is None:
            self._model = self._load_model(device_map=self._device_map,
                                           torch_dtype=self._torch_dtype)
            self._load_hook()
        return self._model

    @property
    def type(self) -> str:
        """
        The type of the model.
        """
        return self._type

    @property
    def pedigree(self) -> str:
        """
        The pedigree of the model.
        """
        return self._pedigree

    @abstractmethod
    def _get_model_pedigree(self) -> str:
        """
        Set the pedigree of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def _load_config(self) -> PretrainedConfig:
        """
        Load the config of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def _load_tokenizer(self, trust_remote_code=False) -> PreTrainedTokenizerBase:
        """
        Load the tokenizer of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def _load_model(self, device_map=None, torch_dtype=None) -> PreTrainedModel:
        """
        Load the model of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def _load_hook(self) -> None:
        """
        Run after the model is loaded.
        """
        pass

    @abstractmethod
    def _persist_hook(self) -> None:
        """
        Run after the model is persisted.
        """
        pass

    def persisted(self, save_path: Path) -> None:
        if not isinstance(save_path, Path):
            raise ValueError("save_path must be a Path object")
        if not save_path.exists():
            raise ValueError("save_path must exist")
        self._path = save_path
        self._persist_hook()

    def set_loading_options(self, device_map=None, torch_dtype=None) -> None:
        """Configure device_map and torch_dtype for lazy model loading.

        This must be called before the first access to `model`.
        """
        if self._model is not None:
            raise ValueError("Model already loaded. set_loading_options must be called before accessing model.")
        if device_map is not None:
            self._device_map = device_map
        if torch_dtype is not None:
            self._torch_dtype = torch_dtype

    def support_layer_wise_schedule(self) -> bool:
        return True

    def enable_kv_cache(self, enable: bool):
        self.model.config.use_cache = enable

    def get_pipeline_functions(self, pipeline: Literal[PipelineType.MODEL_WISE, PipelineType.LAYER_WISE]) -> Tuple[
        GeneratedForwardFuncType, GeneratedVisitFuncType]:
        """根据pipeline类型获取对应的forward和visit函数。

        Args:
            pipeline: pipeline类型，'model_wise' 或 'layer_wise'

        Returns:
            Tuple[GeneratedForwardFuncType, GeneratedVisitFuncType]: forward函数和visit函数的元组
        """
        if pipeline == PipelineType.MODEL_WISE:
            generated_forward_func = cast(GeneratedForwardFuncType, model_wise_forward_func)
            generated_visit_func = cast(GeneratedVisitFuncType, model_wise_visit_func)
        else:
            generated_forward_func = cast(GeneratedForwardFuncType, transformers_generated_forward_func)
            generated_visit_func = cast(GeneratedVisitFuncType, generated_decoder_layer_visit_func)

        return generated_forward_func, generated_visit_func

    def _initialize_torch_dtype(self):
        """初始化torch dtype，子类可覆盖实现"""
        return self._config.torch_dtype if self._device is DeviceType.NPU else torch.float32
