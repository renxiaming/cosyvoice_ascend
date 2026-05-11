#  -*- coding: utf-8 -*-
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


from typing import List, Optional, Any, Generator, Callable

from torch import nn

from msmodelslim.core.base.processor import BaseProcessor
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.base.runner import BaseRunner
from msmodelslim.core.runner.generated_schedule import ProcessUnit, generated_schedule

GeneratedForwardFuncType = Callable[[nn.Module, Any], Generator[ProcessRequest, Any, None]]
GeneratedVisitFuncType = Callable[[nn.Module], Generator[ProcessRequest, Any, None]]


class GeneratedProcessUnit(ProcessUnit):
    def __init__(
            self, processor: BaseProcessor,
            generated_forward_func: GeneratedForwardFuncType,
            generated_visit_func: GeneratedVisitFuncType,
            input_datas: Optional[List[Any]] = None,
    ):
        self.generated_forward_func = generated_forward_func
        self.generated_visit_func = generated_visit_func
        super().__init__(processor, input_datas)

    def build_generators(self) -> List[Generator[ProcessRequest, Any, None]]:
        if self.input_datas:
            dataloader = self._create_dataloader(self.input_datas, 0, 1, 1)
            return [self.generated_forward_func(self.processor.model, data) for data in dataloader]
        else:
            return [self.generated_visit_func(self.processor.model)]


class GeneratedRunner(BaseRunner):

    def __init__(
            self,
            model: nn.Module,
            generated_forward_func: GeneratedForwardFuncType,
            generated_visit_func: GeneratedVisitFuncType,
    ):
        super().__init__()
        self.model = model
        self.process_unit: List[GeneratedProcessUnit] = []
        self.generated_forward_func = generated_forward_func
        self.generated_visit_func = generated_visit_func

    def add_processor(self, processor: BaseProcessor, input_datas: Optional[List[Any]] = None, append: bool = True):
        if append:
            self.process_unit.append(GeneratedProcessUnit(processor,
                                                          self.generated_forward_func,
                                                          self.generated_visit_func,
                                                          input_datas)
                                     )
        else:
            self.process_unit.insert(0, GeneratedProcessUnit(processor,
                                                             self.generated_forward_func,
                                                             self.generated_visit_func,
                                                             input_datas
                                                             )
                                     )

    def run(self):
        generated_schedule(self.process_unit)
