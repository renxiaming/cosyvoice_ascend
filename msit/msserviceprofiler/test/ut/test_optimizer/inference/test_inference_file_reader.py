# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from msserviceprofiler.modelevalstate.inference.file_reader import FileHanlder


class TestFileHandler:
    @staticmethod
    def test_load_static_data(static_file):
        fh = FileHanlder(static_file)
        fh.load_static_data()
        assert fh.hardware
        assert fh.env_info
        assert fh.mindie_info
        assert fh.model_config_info
        assert fh.model_struct_info
        assert fh.prefill_op_data
        assert fh.decode_op_data
