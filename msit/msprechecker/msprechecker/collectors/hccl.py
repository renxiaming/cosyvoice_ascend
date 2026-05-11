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

import os
import shlex
import shutil
import subprocess
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

from msguard import Rule

from .base import BaseCollector
from ..utils import get_npu_count, is_in_container

    
HCCN_TOOL_CMD = "/usr/local/Ascend/driver/tools/hccn_tool"


class HCCNCollector(BaseCollector):
    CMD_NAME = ""

    def __init__(self, error_handler=None):
        super().__init__(error_handler)
        self.npu_count = get_npu_count()

    @abstractmethod
    def _generate_cmd(self):
        pass

    def _run_cmd(self, cmd: str):
        output = None
        try:
            output = subprocess.check_output(
                shlex.split(cmd), stderr=subprocess.DEVNULL, text=True
            )
        except Exception:
            output = "100% packet loss"

        return output

    def _collect_data(self):
        if not Rule.input_file_exec.is_satisfied_by(HCCN_TOOL_CMD):
            working_place = "宿主机" if not is_in_container() else "容器"
            self.error_handler.add_error(
                filename=__file__,
                function='_collect_data', lineno=55,
                what=f"{working_place}上没有找到 'hccn_tool' 命令或者权限不符合要求",
                reason=f"{working_place}上没有找到 'hccn_tool' 命令或者权限不符合要求"
            )
            return {}

        cmds = list(self._generate_cmd())
        max_workers = min(len(cmds), os.cpu_count() or 1)

        with ThreadPoolExecutor(max_workers) as executor:
            futures = [executor.submit(self._run_cmd, cmd) for cmd in cmds]

        return [future.result() for future in futures]


class VnicCollector(HCCNCollector):
    CMD_NAME = "vnic"

    def _generate_cmd(self):
        for device_id in range(self.npu_count):
            yield f"{HCCN_TOOL_CMD} -i {device_id} -{self.CMD_NAME} -g"


class LinkCollector(HCCNCollector):
    CMD_NAME = "link"

    def _generate_cmd(self):
        for device_id in range(self.npu_count):
            yield f"{HCCN_TOOL_CMD} -i {device_id} -{self.CMD_NAME} -g"


class TlsCollector(HCCNCollector):
    CMD_NAME = "tls"

    def _generate_cmd(self):
        for device_id in range(self.npu_count):
            yield f"{HCCN_TOOL_CMD} -i {device_id} -{self.CMD_NAME} -g"


class HCCLCollector(BaseCollector):
    CMD_NAME = "ping"

    def __init__(self, error_handler=None, *, rank_table=None):
        super().__init__(error_handler)
        self.rank_table = rank_table
        self.option = "-hccs_ping" if getattr(rank_table, 'version', "1.0") == "1.2" else "-ping"
    
    def _run_cmd(self, device_id: int, device_ip: str):
        cmd = f"{HCCN_TOOL_CMD} -i {device_id} {self.option} -g address {device_ip}"

        output = None
        try:
            output = subprocess.check_output(
                shlex.split(cmd), stderr=subprocess.DEVNULL, text=True
            )
        except Exception as e:
            output = "100% packet loss"

        return output

    def _collect_data(self):
        if not shutil.which(HCCN_TOOL_CMD):
            working_place = "宿主机" if not is_in_container() else "容器"
            self.error_handler.add_error(
                filename=__file__,
                function='_collect_data', lineno=63,
                what=f"{working_place}上没有找到 'hccn_tool' 命令",
                reason=f"[Errno 2] No such file or directory: '{HCCN_TOOL_CMD}'"
            )
            return {}

        npu_count = get_npu_count()
        max_workers = min(npu_count * 8, os.cpu_count() or 1) # each device has maximum concurrency 8

        all_devices = (
            device_info 
            for device_info_list in self.rank_table.host_to_devices.values() 
            for device_info in device_info_list
        )

        futures = {}
        with ThreadPoolExecutor(max_workers) as executor:
            futures = {
                executor.submit(self._run_cmd, device_id, device_info.device_ip): (device_id, device_info.rank_id)
                for device_info in all_devices
                for device_id in range(npu_count)
            }

        results = [[] for _ in range(npu_count)]
        for future in as_completed(futures):
            device_id, rank_id = futures[future]
            results[device_id].append(
                {"rank_id": rank_id, "result": future.result()}
            )

        return results
