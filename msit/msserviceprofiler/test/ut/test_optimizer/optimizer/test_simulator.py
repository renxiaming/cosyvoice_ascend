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

import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import msserviceprofiler.modelevalstate
from msserviceprofiler.modelevalstate.config.config import settings
from msserviceprofiler.modelevalstate.optimizer.simulator import Simulator, VllmSimulator, enable_simulate


class TestSimulate(unittest.TestCase):

    def test_set_config_dict(self):
        origin_config = {"a": {"b": {"c": 3}}}
        Simulator.set_config(origin_config, "a.b.c", 4)
        assert origin_config["a"]["b"]["c"] == 4

    def test_set_config_list(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        Simulator.set_config(origin_config, "a.b.0.c", 4)
        assert origin_config["a"]["b"][0]["c"] == 4

    def test_set_config_new_key(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        Simulator.set_config(origin_config, "a.b.0.d", 4)
        assert origin_config["a"]["b"][0]["d"] == 4

    def test_set_config_add_dict_list_dict(self):
        origin_config = {"a": {"b": {"c": 3}}}
        Simulator.set_config(origin_config, "a.d.0.c", 4)
        assert origin_config["a"]["d"][0]["c"] == 4

    def test_set_config_add_dict(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        Simulator.set_config(origin_config, "a.b.1.c", 4)
        assert origin_config["a"]["b"][1]["c"] == 4

    def test_set_config_add_dict_list_dict_dict(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        Simulator.set_config(origin_config, "a.d.0.c.e", 4)
        assert origin_config["a"]["d"][0]["c"]["e"] == 4


class TestVllmSimulator(unittest.TestCase):

    @pytest.fixture
    def pre_simulator(self):
        simulator = VllmSimulator(settings.vllm)
        simulator.mindie_log = "mock_log_file.log"
        simulator.mindie_log_offset = 0
        simulator.process = MagicMock()
        return simulator


def test_enable_simulate_with_simulator(tmpdir, monkeypatch):
    config_path = Path(tmpdir).joinpath("config.json")
    with open(config_path, 'w') as f:
        f.write("""{
    "Version": "1.0.0",
    "ServerConfig": {
        "tlsCaFile": [
            "ca.pem"
        ],
        "tlsCert": "security/certs/server.pem"
    },
    "BackendConfig": {
        "backendName": "mindieservice_llm_engine",
        "ModelDeployConfig": {
            "maxSeqLen": 2560,
            "maxInputTokenLen": 2048,
            "truncation": false,
            "ModelConfig": [
                {
                    "modelInstanceType": "Standard"
                }
            ]
        },
        "ScheduleConfig": {
            "templateType": "Standard"
        }
    }
}""")
    settings.mindie.config_path = config_path
    settings.mindie.config_bak_path = Path(tmpdir).joinpath("config_bak.json")
    simulator = Simulator(settings.mindie)
    monkeypatch.setattr(msserviceprofiler.modelevalstate.optimizer.simulator, "simulate_flag", True)
    with enable_simulate(simulator) as flag:
        with open(config_path, 'r') as f:
            data = json.load(f)
            assert data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0][
                       "plugin_params"] == '{"plugin_type": "simulate"}'
    with open(config_path, 'r') as f:
        data = json.load(f)
        assert "plugin_params" not in data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]


def test_enable_simulate_with_simulator_plugin_params_exists(tmpdir, monkeypatch):
    config_path = Path(tmpdir).joinpath("config.json")
    data = {
        "BackendConfig": {
            "backendName": "mindieservice_llm_engine",
            "ModelDeployConfig": {
                "maxSeqLen": 2560,
                "ModelConfig": [
                    {
                        "modelInstanceType": "Standard",
                        "plugin_params": "{\"plugin_type\":\"tp\"}"
                    }
                ]

            },
            "ScheduleConfig": {
                "templateType": "Standard"
            }
        }
    }
    with open(config_path, 'w') as f:
        json.dump(data, f)
    settings.mindie.config_path = config_path
    settings.mindie.config_bak_path = Path(tmpdir).joinpath("config_bak.json")
    simulator = Simulator(settings.mindie)
    monkeypatch.setattr(msserviceprofiler.modelevalstate.optimizer.simulator, "simulate_flag", True)
    with enable_simulate(simulator) as flag:
        with open(config_path, 'r') as f:
            data = json.load(f)
            assert data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0][
                       "plugin_params"] == '{"plugin_type": "tp,simulate"}'
    with open(config_path, 'r') as f:
        data = json.load(f)
        assert data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["plugin_params"] == '{"plugin_type":"tp"}'
