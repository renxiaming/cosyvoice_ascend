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
import pandas as pd
from msserviceprofiler.modelevalstate.inference.data_format_v1 import BatchField, RequestField
from msserviceprofiler.modelevalstate.inference.dataset import DataProcessor, InputData, CustomLabelEncoder,\
      preset_category_data, CustomOneHotEncoder
from msserviceprofiler.modelevalstate.inference.file_reader import FileHanlder


def test_preprocessor(static_file):
    custom_encoder = CustomLabelEncoder(preset_category_data)
    custom_encoder.fit()
    processor = DataProcessor(custom_encoder)
    fh = FileHanlder(static_file)
    fh.load_static_data()
    input_data = InputData(
        batch_field=BatchField("decode", 20, 20.0, 580.0, 29.0),
        request_field=(
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
        ),
        model_op_field=fh.get_op_field("decode", 29, 29, fh.prefill_op_data, fh.decode_op_data),
        model_struct_field=fh.model_struct_info,
        model_config_field=fh.model_config_info,
        mindie_field=fh.mindie_info,
        env_field=fh.env_info,
        hardware_field=fh.hardware
    )
    result = processor.preprocessor(input_data)
    assert len(result) == 902


def test_preprocessor_no_model_op_field(static_file):
    custom_encoder = CustomLabelEncoder(preset_category_data)
    custom_encoder.fit()
    processor = DataProcessor(custom_encoder)
    fh = FileHanlder(static_file)
    fh.load_static_data()
    input_data = InputData(
        batch_field=BatchField("decode", 20, 20.0, 580.0, 29.0),
        request_field=(
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
            RequestField(29.0, 1, 2),
        )
    )
    result = processor.preprocessor(input_data)
    assert len(result) == 160
