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

import os
import sys
import subprocess

from model_convert.aie.bean import ConvertConfig
from components.utils.log import logger
from components.utils.util import filter_cmd

MAX_AIE_CONVERT_TIME = 300


class Convert:
    def __init__(self, config: ConvertConfig) -> None:
        self._config = config
        self.python_version = sys.executable or "python3"

    @classmethod
    def execute_command(cls, cmd):
        """
        Function Description:
            run the following command
        Parameter:
            cmd: command
        Return Value:
            command output result
        Exception Description:
            when invalid command throw exception
        """
        cmd = filter_cmd(cmd)
        logger.info('Execute command:%s', cmd)
        process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, _ = process.communicate(timeout=MAX_AIE_CONVERT_TIME)
        if process.returncode != 0:
            logger.error('Failed to execute command:%s', " ".join(cmd))
            logger.error(stdout.decode())
        else:
            logger.debug(stdout.decode())

    def convert_model(self) -> None:
        cur_dir = os.path.dirname(__file__)
        aie_convert = os.path.join(cur_dir, "..", "aie_convert")
        if not self._config.output.endswith(".om"):
            self._config.output += ".om"
        run_cmd = [aie_convert, self._config.model, self._config.output, self._config.soc_version]
        self.execute_command(run_cmd)
        logger.info("AIE model convert finished, the command: %s", run_cmd)
