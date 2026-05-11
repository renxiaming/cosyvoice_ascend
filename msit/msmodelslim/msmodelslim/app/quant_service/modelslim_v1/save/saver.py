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

from abc import abstractmethod
from collections.abc import Callable
from typing import Optional, Dict, Type, Any, Set, List, Literal

import torch.distributed as dist
from pydantic.functional_validators import BeforeValidator
from torch import nn
from typing_extensions import Annotated

import msmodelslim.quant.ir as qir
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.quant.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.utils.dist import DistHelper


class AutoSaverBaseConfig(AutoProcessorConfig):
    type: Literal['auto_save'] = "auto_save"

    @abstractmethod
    def set_save_directory(self, save_directory):
        pass


def validate_auto_saver_processor_config_list(v: Any) -> List['AutoProcessorConfig']:
    if isinstance(v, list):
        validated_configs = []
        for item in v:
            if isinstance(item, dict):
                validated_configs.append(AutoSaverBaseConfig.model_validate(item))
            elif isinstance(item, AutoSaverBaseConfig):
                validated_configs.append(item)
            else:
                raise ValueError(f"Invalid config item type: {type(item)}")
        if not isinstance(validated_configs[-1], AutoSaverBaseConfig):
            raise TypeError("The save config you provide is not a saver config")
        return validated_configs
    raise ValueError("Expected a list of AutoSaverBaseConfig or dict")


AutoSaverConfigList = Annotated[
    List[AutoSaverBaseConfig],
    BeforeValidator(validate_auto_saver_processor_config_list)
]


class AutoSaverProcessor(AutoSessionProcessor):

    def __init__(self, model: nn.Module, config: AutoProcessorConfig, adapter: object, **kwargs: Dict[str, Any]):
        super().__init__(model)
        self.dist_helper: Optional[DistHelper] = None
        self.processed_modules: Set[nn.Module] = set()
        self.process_map: Dict[Type[nn.Module], Callable[[str, nn.Module], None]] = {
            qir.W8A8StaticFakeQuantLinear: self.on_w8a8_static,
            qir.W8A8DynamicFakeQuantLinear: self.on_w8a8_dynamic,
            qir.W4A8DynamicFakeQuantLinear: self.on_w4a8_dynamic,
            nn.Linear: self.on_float_linear,
            nn.Module: self.on_float_module,
            qir.FakeQuantDynamicCache: self.on_dynamic_cache,
            qir.W16A16sLinear: self.on_w16a16s,
        }

    def support_distributed(self) -> bool:

        """
        是否支持分布式校准模式下保存导出件。

        目前分布式校准仅使用DP模式。

        分布式校准启用后，每个rank都会实例化一个AutoSaverProcessor，每个rank的
        实例将会收到全量的on_xxx系列调用，子类应当自行筛选属于本rank的内容进行保存。
        
        最后，在post_run中，会调用一次merge_ranks方法，用于合并各个rank的导出件。
        """

        return False

    def is_data_free(self) -> bool:
        return True

    def pre_run(self) -> None:
        pass

    def post_run(self) -> None:

        for name, sub_module in self.model.named_modules(memo=self.processed_modules):
            self.on_float_module(name, sub_module)

        if self.support_distributed() and dist.is_initialized():
            self.merge_ranks()

    def postprocess(self, request: BatchProcessRequest) -> None:
        prefix, module = request.name, request.module
        for name, sub_module in module.named_modules(memo=self.processed_modules, prefix=prefix):
            if type(sub_module) in self.process_map:
                self.process_map[type(sub_module)](name, sub_module)
                continue
            self.on_float_module(name, sub_module)

    @abstractmethod
    def merge_ranks(self) -> None:

        """
        合并各个rank所保存的导出件。
        """
        pass

    def on_w8a8_static(self, prefix: str, module: qir.W8A8StaticFakeQuantLinear):
        raise NotImplementedError(f"You should implement the on_w8a8_static method for {self.__class__.__name__}")

    def on_w8a8_dynamic(self, prefix: str, module: qir.W8A8DynamicFakeQuantLinear):
        raise NotImplementedError(f"You should implement the on_w8a8_dynamic method for {self.__class__.__name__}")

    def on_w4a8_dynamic(self, prefix: str, module: qir.W4A8DynamicFakeQuantLinear):
        raise NotImplementedError(f"You should implement the on_w4a8_dynamic method for {self.__class__.__name__}")

    def on_float_linear(self, prefix: str, module: nn.Linear):
        raise NotImplementedError(f"You should implement the on_linear method for {self.__class__.__name__}")

    def on_float_module(self, prefix: str, module: nn.Module):
        raise NotImplementedError(f"You should implement the on_float_module method for {self.__class__.__name__}")

    def on_dynamic_cache(self, prefix: str, module: qir.FakeQuantDynamicCache):
        raise NotImplementedError(f"You should implement the on_dynamic_cache method for {self.__class__.__name__}")

    def on_w16a16s(self, prefix: str, module: qir.W16A16sLinear):
        raise NotImplementedError(f"You should implement the on_w16a16s method for {self.__class__.__name__}")
