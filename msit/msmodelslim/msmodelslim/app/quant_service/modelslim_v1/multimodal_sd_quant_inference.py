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

from abc import ABC, abstractmethod

from msmodelslim.utils.exception import UnsupportedError


class MultimodalSDQuantInference(ABC):
    @abstractmethod
    def run_calib_inference(self):
        raise UnsupportedError(f"You should implement the run_calib_inference method for {self.__class__.__name__}")

    @abstractmethod
    def apply_quantization(self, quant_model_func, quant_config, calib_data):
        """
        应用模型量化的抽象方法，子类需实现具体逻辑

        参数:
            quant_model_func: 量化函数（即api中的quant_model函数）
            quant_config: 量化配置对象
            calib_data: 校准数据
        """
        raise UnsupportedError(f"You should implement the apply_quantization method for {self.__class__.__name__}")

