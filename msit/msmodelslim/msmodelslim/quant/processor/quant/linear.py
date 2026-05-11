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
from typing import List, Optional, Literal

from pydantic import Field, ConfigDict
from torch import nn

from msmodelslim.core.QAL import QScope
from msmodelslim.core.QAL.qregistry import QABCRegistry
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.quant.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.quant.quantizer.linear import LinearQuantizer, LinearQConfig
from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.utils.logging import get_logger, logger_setter


class LinearProcessorConfig(AutoProcessorConfig):
    type: Literal["linear_quant"] = "linear_quant"
    qconfig: LinearQConfig = Field(description="量化配置")
    include: List[str] = Field(default_factory=list, description="包含的模块名称")
    exclude: List[str] = Field(default_factory=list, description="排除的模块名称")

    model_config = ConfigDict(extra="forbid")


def _warning_unmatched_pattern(name: str, config_set: ConfigSet) -> None:
    unmatched_keys = config_set.unmatched_keys()
    unmatched_keys = list(filter(lambda x: x != "*", unmatched_keys))
    if unmatched_keys:
        get_logger().warning(
            f"These {name} patterns are not matched any module, please ensure this is as expected: {unmatched_keys}")


@QABCRegistry.register(dispatch_key=LinearProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter()
class LinearQuantProcessor(AutoSessionProcessor):
    def __init__(
            self,
            model: nn.Module,
            config: LinearProcessorConfig,
            adapter: Optional[object] = None,
    ):
        super().__init__(model)
        self.config = config
        self.include = ConfigSet(config.include) if config.include else ConfigSet(["*"])
        self.exclude = ConfigSet(config.exclude) if config.exclude else ConfigSet([])

    def is_data_free(self) -> bool:
        return self.config.qconfig.act.scope == QScope.PER_TOKEN

    def support_distributed(self) -> bool:
        return True

    def post_run(self) -> None:
        _warning_unmatched_pattern("include", self.include)
        _warning_unmatched_pattern("exclude", self.exclude)

    def preprocess(self, request: BatchProcessRequest) -> None:
        self._install_quantizer(request.name, request.module)

    def postprocess(self, request: BatchProcessRequest) -> None:
        self._deploy(request.name, request.module)

    def _install_quantizer(self, prefix: str, module: nn.Module) -> None:
        for name, submodule in module.named_modules():
            full_name = f"{prefix}.{name}" if prefix != "" else name

            if not isinstance(submodule, nn.Linear):
                continue

            if full_name not in self.include:
                continue

            if full_name in self.exclude:
                continue

            self._process_linear(full_name, submodule)

    def _deploy(self, prefix: str, module: nn.Module) -> None:
        for name, submodule in module.named_modules():
            full_name = f"{prefix}.{name}" if prefix != "" else name
            if hasattr(submodule, "deploy"):
                self.model.set_submodule(full_name, submodule.deploy())

    def _process_linear(self, full_name: str, module: nn.Linear) -> None:
        quantizer = LinearQuantizer(self.config.qconfig)
        quantizer.setup(module)
        self.model.set_submodule(full_name, quantizer)
