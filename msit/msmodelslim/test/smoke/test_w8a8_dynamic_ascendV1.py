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

import json
import os

import pytest
import torch
from safetensors.torch import safe_open

from msmodelslim.app.quant_service.modelslim_v1 import process_model
from .test_w8a8_dynamic_only import TestW8A8DynamicQuantization


@pytest.mark.smoke
class TestW8A8DynamicAscendV1Quantization(TestW8A8DynamicQuantization):
    """测试W8A8动态量化功能的单元测试类"""

    def yaml_file_name(self) -> str:
        return "w8a8_dynamic_ascendV1.yaml"

    @pytest.mark.skip
    def test_w8a8_dynamic_quantization_basic(self, test_device, test_dtype):
        pass

    def test_w8a8_dynamic_and_save_ascendv1(self, test_device, test_dtype):
        """测试基本的W8A8动态量化功能"""

        with torch.device(test_device):
            torch.set_default_dtype(test_dtype)
            model = self.create_model()
            process_model(model, self.service_cfg.process + self.service_cfg.save, 'model_wise',
                          self.calib_data)

        # 验证模型文件是否被保存
        assert os.path.exists(os.path.join(self.temp_dir, "quant_model_description.json"))
        assert os.path.exists(os.path.join(self.temp_dir, "quant_model_weights.safetensors"))

        # 检查json内容
        with open(os.path.join(self.temp_dir, "quant_model_description.json"), "r") as f:
            config_data = json.load(f)

        expected_config_data = {
            "0.weight": "W8A8_DYNAMIC",
            "0.weight_scale": "W8A8_DYNAMIC",
            "0.weight_offset": "W8A8_DYNAMIC",
            "0.bias": "W8A8_DYNAMIC",
            "1.weight": "FLOAT",
            "1.bias": "FLOAT",
            "2.weight": "W8A8_DYNAMIC",
            "2.weight_scale": "W8A8_DYNAMIC",
            "2.weight_offset": "W8A8_DYNAMIC",
            "2.bias": "W8A8_DYNAMIC",
            "model_quant_type": "W8A8_DYNAMIC",
            "version": "1.0.0",
        }

        assert config_data.keys() == expected_config_data.keys()

        for key, value in expected_config_data.items():
            assert value == config_data[key]

        dtype_check = {
            "[02].weight": torch.int8,
            "[02].weight_scale": torch.float32,
            "[02].weight_offset": torch.float32,
        }

        shape_check = {
            "0.weight": (self.intermedia_size, self.hidden_size),
            "0.weight_scale": (self.intermedia_size, 1),
            "0.weight_offset": (self.intermedia_size, 1),
            "2.weight": (self.hidden_size, self.intermedia_size),
            "2.weight_scale": (self.hidden_size, 1),
            "2.weight_offset": (self.hidden_size, 1),
            "2.bias": (self.hidden_size,)
        }

        # 检查safetensors中的tensor数据类型
        with safe_open(os.path.join(self.temp_dir, "quant_model_weights.safetensors"),
                       framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                # 根据key的模式匹配对应的dtype
                matched_dtype = None
                for pattern, expected_dtype in dtype_check.items():
                    if key.endswith(pattern.replace("*", "")):
                        matched_dtype = expected_dtype
                        break

                if matched_dtype is not None:
                    assert tensor.dtype == matched_dtype, \
                        f"Tensor {key} has incorrect dtype. Expected {matched_dtype}, got {tensor.dtype}"

                # 根据key的模式匹配对应的shape
                matched_shape = None
                for pattern, expected_shape in shape_check.items():
                    if key.endswith(pattern.replace("*", "")):
                        matched_shape = expected_shape
                        break

                if matched_shape is not None:
                    assert tensor.shape == matched_shape, \
                        f"Tensor {key} has incorrect shape. Expected {matched_shape}, got {tensor.shape}"
