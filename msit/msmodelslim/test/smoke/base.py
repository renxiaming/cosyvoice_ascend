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

import os
import shutil
import tempfile
from abc import abstractmethod
from functools import lru_cache

import pytest
import torch
import yaml

from msmodelslim.app.quant_service.modelslim_v1.quant_config import ModelslimV1ServiceConfig
from msmodelslim.app.quant_service.modelslim_v1.save import AscendV1Config


class SessionTestCaseBase:
    """pytest版本的测试基类，使用fixture模式替代unittest.TestCase"""

    @pytest.fixture(autouse=True)
    def setup_session_test(self):
        """自动设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_dir = os.path.realpath(self.temp_dir)
        assert os.path.exists(self.temp_dir)

        # 创建W8A8量化配置
        self.yaml_cfg_file = os.path.join(os.path.dirname(__file__), "configs", self.yaml_file_name())
        self.service_cfg = ModelslimV1ServiceConfig.model_validate(
            yaml.safe_load(open(self.yaml_cfg_file, "r"))['service_cfg'])

        yield

        # 清理资源
        shutil.rmtree(self.temp_dir)

    @pytest.fixture(autouse=True)
    def setup_ascendv1(self, setup_session_test):
        """设置AscendV1配置"""
        for save_cfg in self.service_cfg.save:
            save_cfg.set_save_directory(self.temp_dir)
            if isinstance(save_cfg, AscendV1Config):
                save_cfg.part_file_size = 0

    @abstractmethod
    def yaml_file_name(self) -> str:
        raise NotImplementedError(f"You should provide a yaml name to init test session")


@lru_cache(maxsize=1)
def is_npu_available():
    try:
        import torch_npu
        return torch.npu.is_available()
    except ImportError:
        return False


@lru_cache(maxsize=1)
def is_cuda_available():
    try:
        return torch.cuda.is_available()
    except ImportError:
        return False
