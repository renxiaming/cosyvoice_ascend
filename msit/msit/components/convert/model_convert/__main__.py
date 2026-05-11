# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

from components.utils.parser import BaseCommand
from components.utils.log import logger
from model_convert.aie.bean import ConvertConfig
from model_convert.aie.core.convert import Convert
from model_convert.cmd_utils import add_arguments, gen_convert_cmd, execute_cmd
from components.utils.security_check import get_valid_read_path, get_valid_write_path, MAX_READ_FILE_SIZE_32G


def parse_input_param(model: str,
                      output: str,
                      soc_version: str) -> ConvertConfig:
    return ConvertConfig(
        model=model,
        output=output,
        soc_version=soc_version
    )


class ModelConvertCommand(BaseCommand):
    def __init__(self, backend, *args, **kwargs):
        super(ModelConvertCommand, self).__init__(*args, **kwargs)
        self.conf_args = None
        self.backend = backend

    def add_arguments(self, parser, **kwargs):
        self.conf_args = add_arguments(parser, backend=self.backend)

    def handle(self, args, **kwargs):
        convert_cmd = gen_convert_cmd(self.conf_args, args, backend=self.backend)
        execute_cmd(convert_cmd)


class AieCommand(BaseCommand):
    def add_arguments(self, parser, **kwargs):
        parser.add_argument("-gm",
                            "--golden-model",
                            dest="model",
                            required=True,
                            default=None,
                            help="the path of the onnx model")
        parser.add_argument("-of",
                            "--output-file",
                            dest="output",
                            required=True,
                            default=None,
                            help="Output file path&name(needn\'t .om suffix for ATC, need .om suffix for AIE)")
        parser.add_argument("-soc",
                            "--soc-version",
                            dest='soc_version',
                            required=True,
                            default=None,
                            help="The soc version.")

    def handle(self, args, **kwargs):
        logger.info("AIE start converting now")
        model_path = get_valid_read_path(args.model, size_max=MAX_READ_FILE_SIZE_32G)
        output_path = get_valid_write_path(args.output)
        try:
            config = parse_input_param(
                model_path, output_path, args.soc_version
            )
        except ValueError as e:
            logger.error(str(e))
            return

        converter = Convert(config)
        if converter is None:
            logger.error('The object of \'convert\' create failed.')
            return

        converter.convert_model()
        logger.info("AIE convert success")


def get_cmd_instance():
    aie_cmd = AieCommand("aie", help_info="Convert onnx to om by aie.")
    atc_cmd = ModelConvertCommand(name="atc", help_info="Convert onnx to om by atc.", backend="atc")
    aoe_cmd = ModelConvertCommand(name="aoe", help_info="Convert onnx to om by aoe.", backend="aoe")
    convert_cmd = BaseCommand(
        name="convert",
        help_info="convert tool converts the model from ONNX, TensorFlow, Caffe and MindSpore to OM. \
                   It supports atc, aoe and aie.",
        children=[atc_cmd, aoe_cmd, aie_cmd]
    )
    return convert_cmd
