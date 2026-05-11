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
import shutil
import unittest
import tempfile
import re
import pytest
from unittest.mock import patch, mock_open, MagicMock
import os
import json

import numpy as np
import pandas as pd

from components.utils.log import logger


with patch.dict("sys.modules", {
    "c2lb": MagicMock(),
    "c2lb_dynamic": MagicMock(),
    "speculative_moe": MagicMock(),
    "c2lb_a3": MagicMock(),
}):
    from components.expert_load_balancing.elb.preprocessing import get_csv_dimensions, get_csv_path, AppArgs, \
            check_input, numerical_sort_key, parse_ep_file, convert, generate_json, has_prefill_decode, \
            refresh_dependent_args, get_csv_path, get_csv_dimensions, process_speculative_moe, \
            process_prefill_or_decode, get_dynamic_expert_hot_from_csv, validate_args
    
from components.expert_load_balancing.elb.constant import SUPPORTED_COMBINATIONS
from components.utils.constants import JSON_FILE_MAX_SIZE

class TestGetCSVDimensions(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_csv_file(self, filename, data=None, header=True):
        """create csv file"""
        file_path = os.path.join(self.temp_dir, filename)
        if data is not None:
            df = pd.DataFrame(data)
            df.to_csv(file_path, header=header, index=False)
        else:
            open(file_path, 'w').close()
        return file_path

    def test_valid_csv(self):
        test_data = np.random.rand(3, 5)
        csv_path = self._create_csv_file("valid.csv", test_data)
        with self.assertLogs(logger="msit_logger", level="INFO") as log:
            rows, cols = get_csv_dimensions(csv_path)

        self.assertEqual((rows, cols), (3, 5))
        self.assertIn(f"Successfully read", log.output[0])
        self.assertIn("Number of row=3, Number of columns=5", log.output[1])

    def test_empty_file(self):
        csv_path = self._create_csv_file("empty.csv", data=None)
        with self.assertRaises(ValueError) as cm:
            get_csv_dimensions(csv_path)
        self.assertIn("文件无有效数据", str(cm.exception))

    def test_malformed_csv(self):
        csv_path = os.path.join(self.temp_dir, "malformed.csv")
        # 创建列数不一致的CSV
        with open(csv_path, 'w') as f:
            f.write("col1,col2\n1\n3,4,5")

        with self.assertRaises(ValueError) as cm:
            get_csv_dimensions(csv_path)
        self.assertIn("CSV格式解析失败", str(cm.exception))

    def test_nonexistent_file(self):
        fake_path = os.path.join(self.temp_dir, "non_existent.csv")
        
        with self.assertRaises(RuntimeError) as cm:
            get_csv_dimensions(fake_path)
        self.assertIn("未知错误", str(cm.exception))

    def test_empty_dataframe(self):
        csv_path = self._create_csv_file("headers_only.csv", data=[], header=True)

        with self.assertRaises(ValueError) as cm:
            get_csv_dimensions(csv_path)
        self.assertIn("文件无有效数据", str(cm.exception))
    
    @patch("pandas.read_csv")
    def test_permission_error(self, mock_read):
        """
        Exception example: Simulating file read permission error
        """
        mock_read.side_effect = PermissionError("权限拒绝")
        csv_path = os.path.join(self.temp_dir, "dummy.csv")
        open(csv_path, 'w').close()

        with self.assertRaises(RuntimeError) as cm:
            get_csv_dimensions(csv_path)
        self.assertIn("未知错误", str(cm.exception))


class TestGetCSVPath(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_file(self, filename):
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, "w") as f:
            f.write("mock_data")
        return file_path

    def test_valid_file_path(self):
        file_name = "decode_info.csv"
        expected_path = self._create_file(file_name)
        result_path = get_csv_path(self.temp_dir, file_name)
        self.assertEqual(result_path, expected_path)

        file_name = "prefill_info.csv"
        expected_path = self._create_file(file_name)
        result_path = get_csv_path(self.temp_dir, file_name)
        self.assertEqual(result_path, expected_path)
    
    def test_file_not_found(self):
        file_name = "non_existent.csv"
        full_path = os.path.join(self.temp_dir, file_name)
        
        with self.assertRaises(FileNotFoundError) as cm:
            get_csv_path(self.temp_dir, file_name)
        
        self.assertIn(full_path, str(cm.exception))
        self.assertIn("parameter -o", str(cm.exception))

    def test_path_concatenation(self):
        file_name = "test.csv"
        expected_path = os.path.join(self.temp_dir + os.sep, file_name)
        self._create_file(file_name)
        result_path = get_csv_path(self.temp_dir + os.sep, file_name)
        self.assertEqual(result_path, expected_path)
    
        rel_dir = "."
        file_name = "rel_test.csv"
        expected_path = os.path.join(rel_dir, file_name)
        with open(file_name, "w") as f:
            f.write("data")
        try:
            result_path = get_csv_path(rel_dir, file_name)
            self.assertEqual(result_path, expected_path)
        finally:
            os.remove(file_name)
        
    def test_invalid_filename(self):
        with self.assertRaises(FileNotFoundError):
            get_csv_path(self.temp_dir, "../invalid_path.csv")


def test_app_args_initialization():
    """测试 AppArgs 类参数是否正确初始化"""
    # 定义测试参数
    input_args = {
        "expert_popularity_csv_load_path": "/data/popularity.csv",
        "output_dir": "/output",
        "num_nodes": 4,
        "num_npus": 8,
        "share_expert_devices": 2,
        "num_redundancy_expert": 1,
        "algorithm": "dynamic_c2lb",
        "device_type": "GPU"
    }

    # 初始化对象
    args = AppArgs(**input_args)

    # 验证直接映射参数
    assert args.trace_path == input_args["expert_popularity_csv_load_path"]
    assert args.deploy_fp == input_args["output_dir"]
    assert args.n_nodes == input_args["num_nodes"]
    assert args.n_devices == input_args["num_npus"]
    assert args.n_share_expert_devices == input_args["share_expert_devices"]
    assert args.redundant_experts == input_args["num_redundancy_expert"]
    assert args.algorithm == input_args["algorithm"]
    assert args.device_type == input_args["device_type"]

    # 验证固定默认值
    assert args.selected_layers == [-1, -1]
    assert args.n_layers == 58
    assert args.max_time_in_seconds == 300
    assert args.eplb_map == "./"
    assert args.n_selected_expert == 8
    assert args.collection_interval == 16
    assert args.cpu_per_process == 12
    assert args.num_stages == 8
    assert args.enhanced is False
    assert args.black_box_annealing is False
    assert args.all2all_balance is False

    assert args.n_experts == 0 


def test_valid_input():
    """测试正常参数通过验证"""
    # 创建有效参数对象
    args = AppArgs(
        expert_popularity_csv_load_path="/data",
        output_dir="/output",
        num_nodes=4,
        num_npus=8,
        share_expert_devices=2,
        num_redundancy_expert=16,  # 16是8的倍数
        algorithm="c2lb",
        device_type="GPU"
    )
    args.selected_layers = [0, 10]  # 有效范围
    args.max_time_in_seconds = 300  # 正值
    
    # 应无异常抛出
    check_input(args)


class TestNumericalSortKey:
    """测试文件名的数值排序函数"""
    
    @pytest.mark.parametrize("filename, expected", [
        # 基本数字格式
        ("file1.txt", ["file", 1, ".txt"]),
        ("file123.txt", ["file", 123, ".txt"]),
        ("123file.txt", ["", 123, "file.txt"]),
        ("file00123.txt", ["file", 123, ".txt"]), 
    ])
    def test_sort_key(self, filename, expected):
        """测试不同文件名的排序键生成"""
        result = numerical_sort_key(filename)
        assert result == expected


class TestParseEpFile:
    """测试专家文件解析函数"""

    # 基本专家文件结构
    BASE_EP_FILE = {
        "moe_layer_count": 2,
        "layer_list": [
            {
                "layer_id": 1,
                "device_count": 3,
                "device_list": [
                    {"device_expert": [1, 2]},
                    {"device_expert": [3, 4]},
                    {"device_expert": [5, 6]}
                ]
            },
            {
                "layer_id": 2,
                "device_count": 2,
                "device_list": [
                    {"device_expert": [7, 8]},
                    {"device_expert": [9, 10]}
                ]
            }
        ]
    }

    def test_normal_usage_file_path(self):
        """测试通过文件路径正常解析"""
        # 模拟文件内容
        file_content = json.dumps(self.BASE_EP_FILE)
        
        # 模拟文件打开和读取
        with patch("components.expert_load_balancing.elb.preprocessing.ms_open", mock_open(read_data=file_content)) as mock_open_func:
            result = parse_ep_file("path/to/ep.json")
            
            # 验证文件打开
            mock_open_func.assert_called_once_with("path/to/ep.json", max_size=JSON_FILE_MAX_SIZE)
            
            # 验证解析结果
            assert result == {
                1: [[1, 2], [3, 4], [5, 6]],  # 第一个设备的专家被跳过
                2: [[7, 8], [9, 10]]          # 第一个设备的专家被跳过
            }

    def test_share_expert_devices_out_of_range(self):
        """测试分享专家设备数超出范围"""
        # n_share_expert_devices 大于设备数量
        result = parse_ep_file("dummy_path", ep_file=self.BASE_EP_FILE, n_share_expert_devices=5)
        
        # 应为空列表
        assert result == {1: [], 2: []}


class TestConvertFunction:
    """测试数据转换函数"""

    @pytest.mark.parametrize("input_val, expected", [
        # NumPy 整数类型
        (np.int8(42), 42),
        (np.int16(32767), 32767),
        (np.int32(-100), -100),
        (np.int64(2**40), 2**40),
        (np.uint8(255), 255),
    ])
    def test_valid_conversions(self, input_val, expected):
        """测试有效的转换结果"""
        result = convert(input_val)
        
        # 处理NaN值
        if isinstance(expected, list) and any(np.isnan(x) for x in expected if isinstance(x, float)):
            # 对于数组中的NaN，分别验证每个元素
            for i in range(len(result)):
                if isinstance(expected[i], float) and np.isnan(expected[i]):
                    assert np.isnan(result[i])
                else:
                    assert result[i] == expected[i]
        elif isinstance(result, float) and np.isnan(expected):
            # 对于单个NaN值
            assert np.isnan(result)
        else:
            # 其他情况直接比较
            assert result == expected or np.isclose(result, expected, equal_nan=True)


    @pytest.mark.parametrize("invalid_input", [
        # Python原生类型
        42, 
        3.14,
        "hello",
        [1, 2, 3],
        {"a": 1},
    ])
    def test_invalid_inputs(self, invalid_input):
        """测试无效输入类型"""
        with pytest.raises(TypeError) as excinfo:
            convert(invalid_input)
            
        # 验证错误消息是否包含类型信息
        expected_type = type(invalid_input).__name__
        assert expected_type in str(excinfo.value)
        assert "is not JSON serializable" in str(excinfo.value)


class TestHasPrefillDecode:
    """测试预填充和解码文件检测函数"""
    
    @patch("os.listdir")
    @patch("os.path.isfile")
    def test_only_decode_files(self, mock_isfile, mock_listdir):
        """测试只有解码文件的情况"""
        # 设置目录内容
        mock_listdir.return_value = [
            "decode_1.csv", 
            "decode_2.csv", 
            "model_gen_config.json",
            "other_file.txt"
        ]
        # 设置所有都是文件
        mock_isfile.side_effect = lambda x: True
        
        # 创建模拟参数
        new_args = MagicMock()
        new_args.trace_path = "/trace/folder"
        
        # 执行测试
        has_decode, has_prefill = has_prefill_decode(new_args)
        
        # 验证结果
        assert has_decode is True
        assert has_prefill is False
        # 验证文件路径检查
        assert mock_isfile.call_count == 4
    
    @patch("os.listdir")
    @patch("os.path.isfile")
    def test_only_prefill_files(self, mock_isfile, mock_listdir):
        """测试只有预填充文件的情况"""
        # 设置目录内容
        mock_listdir.return_value = [
            "prefill_001.csv",
            "prefill_002.csv",
            "config.json"
        ]
        # 设置所有都是文件
        mock_isfile.side_effect = lambda x: True
        
        # 创建模拟参数
        new_args = MagicMock()
        new_args.trace_path = "/trace/folder"
        
        # 执行测试
        has_decode, has_prefill = has_prefill_decode(new_args)
        
        # 验证结果
        assert has_decode is False
        assert has_prefill is True
        # 验证文件路径检查
        assert mock_isfile.call_count == 3


    @patch("os.listdir")
    @patch("os.path.isfile")
    def test_both_file_types(self, mock_isfile, mock_listdir):
        """测试同时存在两种文件类型的情况"""
        # 设置目录内容
        mock_listdir.return_value = [
            "decode_1.csv",
            "prefill_1.csv",
            "both_file.txt"
        ]
        # 设置所有都是文件
        mock_isfile.side_effect = lambda x: True
        
        # 创建模拟参数
        new_args = MagicMock()
        new_args.trace_path = "/trace/folder"
        
        # 执行测试并验证异常
        with pytest.raises(ValueError) as excinfo:
            has_prefill_decode(new_args)
        
        # 验证错误信息
        assert "Make sure you only have one of decode and prefill" in str(excinfo.value)


    @patch("os.listdir")
    @patch("os.path.isfile")
    def test_no_file_types(self, mock_isfile, mock_listdir):
        """测试没有任何相关文件的情况"""
        # 设置目录内容
        mock_listdir.return_value = [
            "model_gen_config.json",
            "other_data.dat",
            "notes.txt"
        ]
        # 设置所有都是文件
        mock_isfile.side_effect = lambda x: True
        
        # 创建模拟参数
        new_args = MagicMock()
        new_args.trace_path = "/trace/folder"
        
        # 执行测试并验证异常
        with pytest.raises(ValueError) as excinfo:
            has_prefill_decode(new_args)
        
        # 验证错误信息
        assert "No decode and prefill files" in str(excinfo.value)


class TestGetCSVPath(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_file(self, filename):
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, "w") as f:
            f.write("mock_data")
        return file_path

    def test_valid_file_path(self):
        file_name = "decode_info.csv"
        expected_path = self._create_file(file_name)
        result_path = get_csv_path(self.temp_dir, file_name)
        self.assertEqual(result_path, expected_path)

        file_name = "prefill_info.csv"
        expected_path = self._create_file(file_name)
        result_path = get_csv_path(self.temp_dir, file_name)
        self.assertEqual(result_path, expected_path)
    
    def test_file_not_found(self):
        file_name = "non_existent.csv"
        full_path = os.path.join(self.temp_dir, file_name)
        
        with self.assertRaises(FileNotFoundError) as cm:
            get_csv_path(self.temp_dir, file_name)
        
        self.assertIn(full_path, str(cm.exception))
        self.assertIn("parameter -o", str(cm.exception))


class TestGetCSVDimensions(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_csv_file(self, filename, data=None, header=True):
        """create csv file"""
        file_path = os.path.join(self.temp_dir, filename)
        if data is not None:
            df = pd.DataFrame(data)
            df.to_csv(file_path, header=header, index=False)
        else:
            open(file_path, 'w').close()
        return file_path

    def test_valid_csv(self):
        test_data = np.random.rand(3, 5)
        csv_path = self._create_csv_file("valid.csv", test_data)
        with self.assertLogs(logger="msit_logger", level="INFO") as log:
            rows, cols = get_csv_dimensions(csv_path)

        self.assertEqual((rows, cols), (3, 5))
        self.assertIn(f"Successfully read", log.output[0])
        self.assertIn("Number of row=3, Number of columns=5", log.output[1])

    def test_empty_file(self):
        csv_path = self._create_csv_file("empty.csv", data=None)
        with self.assertRaises(ValueError) as cm:
            get_csv_dimensions(csv_path)
        self.assertIn("文件无有效数据", str(cm.exception))

    def test_malformed_csv(self):
        csv_path = os.path.join(self.temp_dir, "malformed.csv")
        # 创建列数不一致的CSV
        with open(csv_path, 'w') as f:
            f.write("col1,col2\n1\n3,4,5")

        with self.assertRaises(ValueError) as cm:
            get_csv_dimensions(csv_path)
        self.assertIn("CSV格式解析失败", str(cm.exception))


class TestRefreshDependentArgs:
    """测试参数刷新函数"""
    
    @pytest.fixture
    def base_args(self):
        """创建基本的模拟参数对象"""
        args = MagicMock()
        args.trace_path = "/path/to/trace"
        args.n_share_expert_devices = 2
        args.algorithm = "3"
        args.device_type = "a2"
        args.selected_layers = [-1, -1]
        args.dump_vresion = True
        args.enhanced = False
        args.black_box_annealing = False
        args.all2all_balance = False
        return args
    
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.join", side_effect=lambda a, b: f"{a}/{b}")
    @patch("os.path.dirname", return_value="/path/to")
    @patch("os.path.basename", return_value="trace")
    @patch("components.expert_load_balancing.elb.preprocessing.has_prefill_decode", return_value=(True, False))
    @patch("json.load")
    @patch("os.path.exists")
    @patch("components.expert_load_balancing.elb.preprocessing.ms_open")
    def test_new_version_with_eplb(self, mock_ms_open, moch_exist, mock_json, mock_has_decode, 
                                  mock_basename, mock_dirname, mock_join, mock_isfile, base_args):
        base_args.eplb_map = ""
        config_data = {
            "num_moe_layers": 16,
            "num_of_experts": 64,
            "eplb_expert_map_file": "/path/to/expert_map.json",
            "collection_Interval": 100,
            "num_of_selected_expert": [8],
            "enable_dangling_shared_expert": True,
            "num_dangling_shared_experts": 4
        }
        mock_json.return_value = config_data
        
        mock_file = mock_open()
        mock_ms_open.return_value.__enter__.return_value = mock_file.return_value
        moch_exist.return_value = True
        result = refresh_dependent_args(base_args)
    
        assert result.trace_path == "/path/to/trace_selected"
        assert result.dump_vresion is False
        mock_ms_open.assert_called_once_with("/path/to/trace_selected/model_gen_config.json")
        assert result.n_layers == 16
        assert result.n_experts == 64
        assert result.eplb_map == "/path/to/expert_map.json"
        assert result.collection_interval == 100
        assert result.n_selected_expert == 8
        assert result.n_share_expert_devices_files == 4  # 覆盖初始值
        assert result.enhanced is True
        assert result.black_box_annealing is True
        assert result.all2all_balance is True
        assert result.num_stages == 8
        assert result.selected_layers == [0, 15]  # 从0到15层


    @patch("os.path.isfile", return_value=False)
    @patch("components.expert_load_balancing.elb.preprocessing.has_prefill_decode", return_value=(True, False))
    def test_old_version_base(self, mock_has_decode, mock_isfile, base_args):
        """测试旧版本文件的基本配置"""
        result = refresh_dependent_args(base_args)
        assert result.trace_path == "/path/to/trace"
        assert result.dump_vresion is True
        assert result.eplb_map == ""
        assert result.num_stages == 1
        mock_has_decode.assert_called_once_with(result)
        assert result.has_decode is True
        assert result.has_prefill is False
        assert result.enhanced is True  # algorithm=3
        assert result.black_box_annealing is False
        assert result.all2all_balance is False


class TestProcessSpeculativeMoe:
    """测试推测式MOE处理函数"""
    
    @pytest.fixture
    def base_args(self):
        """创建基本的模拟参数对象"""
        args = MagicMock()
        args.expert_popularity_csv_load_path = "/path/to/popularity.csv"
        args.output_dir = "/output/dir"
        args.num_nodes = 8
        args.num_npus = 64
        args.share_expert_devices = 2
        args.num_redundancy_expert = 4
        args.algorithm = "3"
        args.device_type = "a2"
        return args

    @patch("components.expert_load_balancing.elb.preprocessing.refresh_dependent_args")
    @patch("components.expert_load_balancing.elb.preprocessing.check_input")
    @patch("components.expert_load_balancing.elb.preprocessing.logger")
    @patch("components.expert_load_balancing.elb.preprocessing.get_csv_path")
    @patch("components.expert_load_balancing.elb.preprocessing.get_csv_dimensions")
    @patch("components.expert_load_balancing.elb.preprocessing.process_prefill_or_decode")
    def test_new_version_processing(self, mock_process, mock_dimensions, mock_get_csv, 
                                  mock_logger, mock_check, mock_refresh, base_args):
        """测试新版本处理流程"""
        # 配置模拟的刷新后参数
        refreshed_args = MagicMock()
        refreshed_args.dump_vresion = False
        refreshed_args.has_decode = True
        refreshed_args.has_prefill = False
        mock_refresh.return_value = refreshed_args
        process_speculative_moe(base_args)

        mock_refresh.assert_called_once()
        mock_check.assert_called_once_with(refreshed_args)
        mock_logger.info.assert_called_once_with("new version")
        mock_process.assert_called_once_with(refreshed_args)
        
        mock_get_csv.assert_not_called()
        mock_dimensions.assert_not_called()


    @patch("components.expert_load_balancing.elb.preprocessing.refresh_dependent_args")
    @patch("components.expert_load_balancing.elb.preprocessing.check_input")
    @patch("components.expert_load_balancing.elb.preprocessing.logger")
    @patch("components.expert_load_balancing.elb.preprocessing.get_csv_path")
    @patch("components.expert_load_balancing.elb.preprocessing.get_csv_dimensions")
    @patch("components.expert_load_balancing.elb.preprocessing.process_prefill_or_decode")
    def test_old_version_decode_only(self, mock_process, mock_dimensions, mock_get_csv, 
                                  mock_logger, mock_check, mock_refresh, base_args):
        """测试旧版本仅解码路径处理"""
        # 配置模拟的刷新后参数
        refreshed_args = MagicMock()
        refreshed_args.dump_vresion = True
        refreshed_args.has_decode = True
        refreshed_args.has_prefill = False
        mock_refresh.return_value = refreshed_args
        
        mock_get_csv.return_value = "/output/dir/decode_info.csv"
        mock_dimensions.return_value = (12, 48)  # (n_layers, n_experts)
        process_speculative_moe(base_args)
        
        mock_logger.info.assert_any_call("old version")
        mock_logger.info.assert_any_call("has decode")
        
        assert refreshed_args.n_layers == 12
        assert refreshed_args.n_experts == 48
        
        mock_get_csv.assert_called_once_with(refreshed_args.deploy_fp, "decode_info.csv")
        mock_dimensions.assert_called_once_with("/output/dir/decode_info.csv")
    
        mock_process.assert_called_once_with(refreshed_args)


class TestProcessPrefillOrDecode(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.trace_path = os.path.join(self.temp_dir.name, 'test_trace.csv')
        with open(self.trace_path, 'w') as f:
            f.write("layer,expert,hotness\n1,1,0.5\n1,2,0.3\n2,1,0.4\n2,2,0.6")

        self.mock_args = MagicMock()
        self.mock_args.trace_path = self.trace_path
        self.mock_args.n_layers = 2
        self.mock_args.n_experts = 2
        self.mock_args.eplb_map = None
        self.mock_args.n_share_expert_devices_files = 1
        self.mock_args.n_selected_expert = 2
        self.mock_args.collection_interval = 1
        self.mock_args.all2all_balance = False
        self.mock_args.n_devices = 2
        self.mock_args.enhanced = False
        self.mock_args.black_box_annealing = False
        self.mock_args.selected_layers = [1, 2]
        self.mock_args.cpu_per_process = 1
        self.mock_args.output_path = os.path.join(self.temp_dir.name, 'output.json')
        self.patcher1 = patch('components.expert_load_balancing.elb.preprocessing.get_dynamic_expert_hot_from_csv')
        self.mock_get_hotness = self.patcher1.start()
        self.mock_get_hotness.return_value = ({1: {1: 0.5, 2: 0.3}, 2: {1: 0.4, 2: 0.6}}, None)

        self.patcher2 = patch('components.expert_load_balancing.elb.preprocessing.ExpSolver')
        self.mock_exp_solver = self.patcher2.start()
        self.mock_solver_instance = MagicMock()
        self.mock_solver_instance.fit.return_value = ({}, [], 0, 0, {1: {1: 1}, 2: {1: 2}})
        self.mock_exp_solver.return_value = self.mock_solver_instance

        self.patcher3 = patch('components.expert_load_balancing.elb.preprocessing.ExpILPSolver')
        self.mock_ilp_solver = self.patcher3.start()
        self.mock_ilp_instance = MagicMock()
        self.mock_ilp_instance.fit.return_value = ({}, [], {1: {1: 1}, 2: {1: 2}})
        self.mock_ilp_solver.return_value = self.mock_ilp_instance

        self.patcher4 = patch('components.expert_load_balancing.elb.preprocessing.second_optim')
        self.mock_second_optim = self.patcher4.start()
        self.mock_second_optim.return_value = {1: {1: 1}, 2: {1: 2}}

        self.patcher5 = patch('components.expert_load_balancing.elb.preprocessing.all_to_all_algorithm_multi_process')
        self.mock_all2all = self.patcher5.start()
        self.mock_all2all.return_value = {1: {1: 1}, 2: {1: 2}}

        self.patcher6 = patch('components.expert_load_balancing.elb.preprocessing.generate_json')
        self.mock_generate_json = self.patcher6.start()

        self.patcher7 = patch('components.expert_load_balancing.elb.preprocessing.logger')
        self.mock_logger = self.patcher7.start()

    def tearDown(self):
        self.temp_dir.cleanup()
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()
        self.patcher5.stop()
        self.patcher6.stop()
        self.patcher7.stop()

    def test_basic_flow(self):
        """测试基本流程"""
        process_prefill_or_decode(self.mock_args)
        self.mock_get_hotness.assert_called_once()
        self.mock_exp_solver.assert_called_once()
        self.mock_solver_instance.fit.assert_called_once_with(1)
        self.mock_generate_json.assert_called_once()

    def test_enhanced_algorithm(self):
        """测试增强算法"""
        self.mock_args.enhanced = True
        process_prefill_or_decode(self.mock_args)
        
        self.mock_ilp_solver.assert_called_once()
        self.mock_ilp_instance.fit.assert_called_once_with(cpu_per_process=1)

    def test_black_box_annealing(self):
        """测试黑盒退火算法"""
        self.mock_args.black_box_annealing = True
        process_prefill_or_decode(self.mock_args)
        
        self.mock_second_optim.assert_called_once()

    def test_all2all_balance(self):
        """测试all2all平衡算法"""
        self.mock_args.all2all_balance = True
        process_prefill_or_decode(self.mock_args)
        self.assertTrue(True) # 程序没有崩溃则认为测试通过

    def test_generate_json_output(self):
        """测试生成JSON输出"""
        process_prefill_or_decode(self.mock_args)
        self.mock_generate_json.assert_called_once_with({1: {1: 1}, 2: {1: 2}}, self.mock_args)

    def test_progress_bar_updates(self):
        """测试进度条更新"""
        with patch('components.expert_load_balancing.elb.preprocessing.tqdm') as mock_tqdm:
            mock_bar = MagicMock()
            mock_tqdm.return_value = mock_bar
            process_prefill_or_decode(self.mock_args)
            self.assertEqual(mock_bar.update.call_count, 6)
            mock_bar.close.assert_called_once()


class TestGenerateJson(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mock_args = MagicMock()
        self.mock_args.deploy_fp = self.temp_dir
        self.test_data = {"key": "value"}

    def tearDown(self):
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.temp_dir)

    @patch('components.expert_load_balancing.elb.preprocessing.ms_makedirs')
    @patch('components.expert_load_balancing.elb.preprocessing.ms_open')
    @patch('components.expert_load_balancing.elb.preprocessing.logger')
    def test_generate_json_with_decode(self, mock_logger, mock_ms_open, mock_ms_makedirs):
        self.mock_args.has_decode = True
        self.mock_args.has_prefill = False
        generate_json(self.test_data, self.mock_args)
        mock_ms_makedirs.assert_called_once_with(self.mock_args.deploy_fp, exist_ok=True)
        expected_file = os.path.join(self.mock_args.deploy_fp, "decode_global_deployment.json")
        mock_ms_open.assert_called_once_with(expected_file, "w")
        mock_logger.info.assert_called_once_with(f"save in {self.mock_args.deploy_fp}")


def create_temp_csv(folder, filename, data):
    path = os.path.join(folder, filename)
    np.savetxt(path, data, delimiter=",", fmt="%d")
    return path

def test_get_dynamic_expert_hot_from_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        decode_data = np.random.randint(0, 100, (100, 256))
        create_temp_csv(tmpdir, "decode_1.csv", decode_data)
        dynamic_hot, topk = get_dynamic_expert_hot_from_csv(
            root_folder=tmpdir,
            n_layer=2,
            n_expert=256,
            collection_interval=16,
            topk_info=False
        )
        assert dynamic_hot.shape[0] == 2
        assert dynamic_hot.shape[1] == 256
        assert topk is None

    with tempfile.TemporaryDirectory() as tmpdir:
        decode_data = np.random.randint(0, 100, (100, 256))
        topk_data = np.random.rand(100, 8)
        create_temp_csv(tmpdir, "decode_1.csv", decode_data)
        create_temp_csv(tmpdir, "decode_topk_1.csv", topk_data)
        dynamic_hot, topk = get_dynamic_expert_hot_from_csv(
            root_folder=tmpdir,
            n_layer=2,
            n_expert=256,
            collection_interval=16,
            topk_info=True
            )
        assert dynamic_hot.shape[0] == 2
        assert dynamic_hot.shape[1] == 256
        assert topk is not None
        assert topk.shape[0] == 2  # n_layer

    with tempfile.TemporaryDirectory() as tmpdir:
        decode_data = np.random.randint(0, 100, (100, 256))
        create_temp_csv(tmpdir, "decode_1.csv", decode_data)
        eplb_map = {
            0: np.arange(256).reshape(1, 256),
            1: np.arange(256).reshape(1, 256)
        }
        dynamic_hot, _ = get_dynamic_expert_hot_from_csv(
            root_folder=tmpdir,
            n_layer=2,
            n_expert=256,
            eplb_map=eplb_map,
            topk_info=False
        )
        assert dynamic_hot.shape[0] == 2
        assert dynamic_hot.shape[1] == 256


class TestValidateArgs:
    class Args:
        def __init__(self, device_type, algorithm):
            self.device_type = device_type
            self.algorithm = algorithm

    @pytest.fixture
    def supported_device(self):
        """返回一个支持的设备类型"""
        return next(iter(SUPPORTED_COMBINATIONS.keys()))

    @pytest.fixture
    def unsupported_device(self):
        """返回一个不支持的设备类型"""
        return "unsupported_device"

    @pytest.fixture
    def supported_algorithm(self, supported_device):
        """返回支持的算法"""
        return SUPPORTED_COMBINATIONS[supported_device]

    @pytest.fixture
    def unsupported_algorithm(self):
        """返回不支持的算法"""
        return "unsupported_algorithm"

    def test_valid_args(self, supported_device, supported_algorithm):
        """测试支持的设备和算法组合"""
        args = self.Args("a2", "1")
        validate_args(args)

    def test_unsupported_device(self, unsupported_device, supported_algorithm):
        """测试不支持的设备类型"""
        args = self.Args(unsupported_device, supported_algorithm)
        with pytest.raises(ValueError) as excinfo:
            validate_args(args)
        assert f"device '{unsupported_device}' is not supported." in str(excinfo.value)

    def test_unsupported_algorithm(self, supported_device, unsupported_algorithm):
        """测试不支持的算法"""
        args = self.Args(supported_device, unsupported_algorithm)
        with pytest.raises(ValueError) as excinfo:
            validate_args(args)
        assert f"device '{supported_device}' does not support algorithm '{unsupported_algorithm}'." in str(excinfo.value)

