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
import shutil
import unittest
import tempfile
import logging 
import re
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock, call

import torch
import json
import numpy as np
import pandas as pd

from components.utils.file_open_check import OpenException
from components.utils.log import logger
from components.expert_load_balancing.elb.constant import PREFILL, DECODE, DECODE_FILE_NAME, \
                        PREFILL_FILE_NAME, ALGORITHM_C2LB, ALGORITHM_DYNAMIC_C2LB, A2, A3


with patch.dict("sys.modules", {
    "c2lb": MagicMock(),
    "c2lb_dynamic": MagicMock(),
    "speculative_moe": MagicMock(),
    "c2lb_a3": MagicMock(),
}):
    from components.expert_load_balancing.elb.load_balancing import save_matrix_to_json, dump_tables, \
    load_expert_popularity_csv, merge_csv_columns, check_file_type, \
    save_dataframes, process_c2lb, process_dynamic_c2lb, load_balancing, process_c2lb_a3, \
    select_algorithm, load_balancing, is_file_readable, copy_file, copy_files_with_recovery, \
    extract_number, process_files_by_type


class TestSaveMatrixToJson(unittest.TestCase):
    def setUp(self):
        # 创建临时目录
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = self.temp_dir.name
        self.valid_deployment_3d = [
            [[0], [1]],
            [[2], [3]], 
            [[4], [5]] 
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_3d_matrix(self):
        file_name = "valid_output"
        expected_json = {
            "moe_layer_count": 3,
            "layer_list": [
                {
                    "layer_id": 0,
                    "device_count": 2,
                    "device_list": [
                        {"device_id": 0, "device_expert": [0]},
                        {"device_id": 1, "device_expert": [1]}
                    ]
                },
                {
                    "layer_id": 1,
                    "device_count": 2,
                    "device_list": [
                        {"device_id": 0, "device_expert": [2]},
                        {"device_id": 1, "device_expert": [3]}
                    ]
                },
                {
                    "layer_id": 2,
                    "device_count": 2,
                    "device_list": [
                        {"device_id": 0, "device_expert": [4]},
                        {"device_id": 1, "device_expert": [5]}
                    ]
                }
            ]
        }

        save_matrix_to_json(self.output_path, file_name, self.valid_deployment_3d)
        
        output_file = os.path.join(self.output_path, f"{file_name}.json")
        with open(output_file, 'r') as f:
            actual_data = json.load(f)
        
        self.assertEqual(actual_data, expected_json)

    def test_invalid_2d_matrix(self):
        invalid_deployment = [
            [0, 1],
            [2, 3]
        ]
        with self.assertRaises(ValueError) as context:
            save_matrix_to_json(self.output_path, "invalid", invalid_deployment)
        self.assertIn("必须是三维数组", str(context.exception))

    def test_invalid_4d_matrix(self):
        invalid_deployment = [[[[0]]]]  
        with self.assertRaises(ValueError) as context:
            save_matrix_to_json(self.output_path, "invalid", invalid_deployment)
        self.assertIn("必须是三维数组", str(context.exception))

    def test_empty_matrix(self):
        with self.assertRaises(ValueError) as context:
            save_matrix_to_json(self.output_path, "empty", [])
        self.assertIn("三维数组", str(context.exception))


class TestDumpTables(unittest.TestCase):
    def setUp(self):
        self.test_path = "test.json"
        if os.path.exists(self.test_path):
            os.remove(self.test_path)

    def tearDown(self):
        if os.path.exists(self.test_path):
            os.remove(self.test_path)

    @patch('components.utils.file_open_check.ms_open', mock_open())
    def test_dump_tables_given_valid_input_when_processed_then_success(self):
        test_d2e =  [
            np.array([[1, 2], [3, 4]], dtype=np.int32), 
            np.array([[5, 6], [7, 8]], dtype=np.float32) 
        ]
        test_path = self.test_path
        
        dump_tables(test_path, test_d2e, n_devices=2)
        
        expected_json = {
            "moe_layer_count": 2,
            "layer_list": [
                {
                    "layer_id": 0,
                    "device_count": 2,
                    "device_list": [
                        {"device_id": 0, "device_expert": 1},
                        {"device_id": 1, "device_expert": 2}
                    ]
                },
                {
                    "layer_id": 1,
                    "device_count": 2,
                    "device_list": [
                        {"device_id": 0, "device_expert": 7.0},
                        {"device_id": 1, "device_expert": 8.0}
                    ]
                }
            ]
        }
        
        with open(test_path, 'r') as f:
            written_data = json.load(f)
            self.assertEqual(written_data, expected_json)

    @patch('components.utils.file_open_check.ms_open', mock_open())
    def test_dump_tables_given_empty_layers_when_processed_then_success(self):
        dump_tables(self.test_path, [], n_devices=2)
        with open(self.test_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(data["moe_layer_count"], 0)

    def test_dump_tables_given_mismatch_device_count_when_processed_then_fail(self):
        with self.assertRaises(IndexError):
            dump_tables(self.test_path, np.array([[[1]]]), n_devices=2)

    @patch('components.utils.file_open_check.ms_open', mock_open())
    def test_dump_tables_given_single_device_when_processed_then_success(self):
        dump_tables(self.test_path, np.array([[[1,2,3]]]), n_devices=1)
        with open(self.test_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(data["layer_list"][0]["device_count"], 1)

    @patch('components.utils.file_open_check.ms_open', mock_open())
    def test_dump_tables_given_irregular_shape_when_processed_then_success(self):
        test_data = [
            np.array([[1, 2], [3, 4]], dtype=np.int32),
            np.array([[5, 6], [7, 8], [9, 10]], dtype=np.int32)  
        ]
        dump_tables(self.test_path, test_data, n_devices=2)
        with open(self.test_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(len(data["layer_list"][1]["device_list"]), 2)

    def test_dump_tables_given_invalid_layer_index_when_processed_then_fail(self):
        with self.assertRaises(IndexError):
            invalid_data = [MagicMock()]
            invalid_data[0].__getitem__.side_effect = IndexError
            dump_tables(self.test_path, invalid_data, n_devices=1)


class TestLoadExpertPopularityCSV(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_csv_file(self, filename, content):
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, "w") as f:
            f.write(content)
        return file_path

    def test_both_files_exist(self):
        self._create_csv_file("decode_info.csv", "col1,col2\n1,2\n3,4")
        self._create_csv_file("prefill_info.csv", "colA,colB\n5,6\n7,8")

        decode_df, prefill_df = load_expert_popularity_csv(self.temp_dir)
        self.assertIsInstance(decode_df, pd.DataFrame)
        self.assertIsInstance(prefill_df, pd.DataFrame)
        self.assertEqual(len(decode_df), 2)
        self.assertEqual(len(prefill_df), 2)

    def test_only_decode_exists(self):
        self._create_csv_file("decode_info.csv", "col1,col2\n1,2")

        decode_df, prefill_df = load_expert_popularity_csv(self.temp_dir)
        self.assertIsNotNone(decode_df)
        self.assertIsNone(prefill_df)

    def test_only_prefill_exists(self):
        self._create_csv_file("prefill_info.csv", "colA,colB\n5,6")

        decode_df, prefill_df = load_expert_popularity_csv(self.temp_dir)
        self.assertIsNone(decode_df)
        self.assertIsNotNone(prefill_df)

    def test_no_files_exist(self):
        with self.assertRaises(FileNotFoundError):
            load_expert_popularity_csv(self.temp_dir)
    
    def test_empty_file_raises_error(self):
        self._create_csv_file("decode_info.csv", "")  

        with self.assertRaises(ValueError) as cm:
            load_expert_popularity_csv(self.temp_dir)
        self.assertIn("empty", str(cm.exception))
    
    def test_header_only_returns_empty_df(self):
        self._create_csv_file("prefill_info.csv", "colA,colB") 

        with self.assertRaises(FileNotFoundError) as cm:
            load_expert_popularity_csv(self.temp_dir)
        self.assertIn("No decode_info.csv or prefill_info.csv found", str(cm.exception))

    @patch("pandas.read_csv")
    def test_general_exception_handling(self, mock_read_csv):
        mock_read_csv.side_effect = PermissionError("模拟权限错误")
        self._create_csv_file("decode_info.csv", "col1,col2\n1,2")

        with self.assertRaises(RuntimeError) as cm:
            load_expert_popularity_csv(self.temp_dir)
        self.assertIn("Error reading", str(cm.exception))


class TestMergeCSVColumns(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_test_csv(self, filename, columns=3, rows=2):
        file_path = os.path.join(self.temp_dir, filename)
        data = np.random.rand(rows, columns)
        df = pd.DataFrame(data)
        df.to_csv(file_path, header=False, index=False)
        return file_path

    def test_normal_merge(self):
        files = [
            self._create_test_csv("pattern_0.csv", columns=2),
            self._create_test_csv("pattern_1.csv", columns=3),
            self._create_test_csv("pattern_2.csv", columns=1)
        ]
        result_df = merge_csv_columns(self.temp_dir, "pattern")
        self.assertEqual(result_df.shape, (2, 6))  # 2行，2+3+1=6列
        expected_columns = [f"expert_{i}" for i in range(6)]
        self.assertListEqual(list(result_df.columns), expected_columns)
    
    def test_no_files_found(self):
        with self.assertRaises(FileNotFoundError) as cm:
            merge_csv_columns(self.temp_dir, "non_existent")
        self.assertIn("No files found", str(cm.exception))

    def test_empty_file(self):
        empty_file = os.path.join(self.temp_dir, "pattern_0.csv")
        with open(empty_file, "w") as f:
            pass  
        with self.assertRaises(ValueError) as cm:
            merge_csv_columns(self.temp_dir, "pattern")
    
    def test_invalid_filename_format(self):
        self._create_test_csv("pattern_abc.csv")
        self._create_test_csv("pattern_1.csv")   

        result_df = merge_csv_columns(self.temp_dir, "pattern")
        self.assertEqual(result_df.shape[1], 3)

    def test_file_sorting(self):
        files = [
            self._create_test_csv("pattern_10.csv"),
            self._create_test_csv("pattern_2.csv"),
            self._create_test_csv("pattern_1.csv")
        ]
        with self.assertLogs(logger="msit_logger", level="DEBUG") as log:
            merge_csv_columns(self.temp_dir, "pattern")

        logged_files = [record.message for record in log.records 
                       if "Processing file" in record.message]
        expected_order = sorted(files, key=lambda x: int(re.search(r'_(\d+)\.', x).group(1)))
        self.assertIn(os.path.basename(expected_order[0]), logged_files[0])

    def test_column_merging(self):
        file1 = os.path.join(self.temp_dir, "pattern_0.csv")
        pd.DataFrame([[1, 2], [3, 4]]).to_csv(file1, header=False, index=False)
        
        file2 = os.path.join(self.temp_dir, "pattern_1.csv")
        pd.DataFrame([[5], [6]]).to_csv(file2, header=False, index=False)
        result_df = merge_csv_columns(self.temp_dir, "pattern")
        expected_data = [
            [1, 2, 5],
            [3, 4, 6]
        ]
        np.testing.assert_array_equal(result_df.values, expected_data)

    def test_path_handling(self):
        self._create_test_csv("pattern_0.csv")

        abs_path = os.path.abspath(self.temp_dir)
        result_abs = merge_csv_columns(abs_path, "pattern")
        self.assertFalse(result_abs.empty)

        rel_path = os.path.relpath(self.temp_dir)
        result_rel = merge_csv_columns(rel_path, "pattern")
        self.assertFalse(result_rel.empty)


class TestCheckFileType(unittest.TestCase):
    def setUp(self):
        """
        Create a temporary directory
        """
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """
        Clean up temporary directories
        """
        shutil.rmtree(self.temp_dir)

    def _create_files(self, filenames):
        """
        Create a list of specified file names in a temporary directory
        """
        for fname in filenames:
            open(os.path.join(self.temp_dir, fname), 'w').close()

    # Test mixed file types
    def test_both_types(self):
        """
        Normal scenario: both prefill and decode files exist
        """
        self._create_files([
            "prefill_data1.csv",
            "prefill_data2.csv",
            "decode_info1.csv",
            "other_file.txt"
        ])

        types, p_count, d_count = check_file_type(self.temp_dir)
        self.assertEqual(types, {PREFILL, DECODE})
        self.assertEqual(p_count, 2)
        self.assertEqual(d_count, 1)

    # Test prefill files only
    def test_only_prefill(self):
        """
        Boundary scenario: only prefill type
        """
        self._create_files([
            "PREFILL_2023.csv",
            "prefill_backup.csv",
            "image.png"
        ])
        
        types, p_count, d_count = check_file_type(self.temp_dir)
        self.assertEqual(types, {PREFILL})
        self.assertEqual(p_count, 2)
        self.assertEqual(d_count, 0)
    
    # Test decode file only
    def test_only_decode(self):
        """
        Boundary scenario: only decode type
        """
        self._create_files([
            "DECODE_results.csv",
            "my_decode_file.csv",
            "temp.csv"
        ])
        
        types, p_count, d_count = check_file_type(self.temp_dir)
        self.assertEqual(types, {DECODE})
        self.assertEqual(p_count, 0)
        self.assertEqual(d_count, 2)

    # Test Empty folder
    def test_empty_folder(self):
        """
        Boundary scenario: No CSV file
        """
        types, p_count, d_count = check_file_type(self.temp_dir)
        self.assertEqual(types, set())
        self.assertEqual(p_count, 0)
        self.assertEqual(d_count, 0)

    # Test Mixed case test
    def test_case_insensitive(self):
        """
        Verify case insensitivity
        """
        self._create_files([
            "PReFill_test.CSV",
            "DECODE_data.Csv",
            "preFILL_alt.CSV"
        ])
        
        types, p_count, d_count = check_file_type(self.temp_dir)
        self.assertEqual(types, {PREFILL, DECODE})
        self.assertEqual(p_count, 2)
        self.assertEqual(d_count, 1)
    
     # Test Invalid path processing
    def test_invalid_path(self):
        """
        Abnormal scenario: The path does not exist
        """
        with self.assertRaises(FileNotFoundError):
            check_file_type("/non/existent/path")


class TestSaveDataFrames(unittest.TestCase):
    def setUp(self):
        """
        Create a temporary directory
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = self.temp_dir.name

    def tearDown(self):
        """
        Clean up temporary directories
        """
        self.temp_dir.cleanup()

    def _get_file_path(self, filename):
        """
        Get the file path in the temporary directory
        """
        return os.path.join(self.test_dir, filename)

    def _assert_file_exists_with_data(self, filename, expected_df):
        """
        Verify that the file exists and the data matches
        """
        file_path = self._get_file_path(filename)
        self.assertTrue(os.path.exists(file_path), f"文件 {filename} 未生成")
        actual_df = pd.read_csv(file_path)
        pd.testing.assert_frame_equal(actual_df, expected_df)

    def _assert_file_not_exists(self, filename):
        """
        Verify that the file does not exist
        """
        file_path = self._get_file_path(filename)
        self.assertFalse(os.path.exists(file_path), f"意外生成了文件 {filename}")

    # Test Both data frames are valid
    def test_save_both_valid_dataframes(self):
        prefill_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        decode_df = pd.DataFrame({"X": [5, 6], "Y": [7, 8]})

        save_dataframes(prefill_df, decode_df, self.test_dir)
        self._assert_file_exists_with_data("prefill_info.csv", prefill_df)
        self._assert_file_exists_with_data("decode_info.csv", decode_df)

    # Test Save only prefill
    def test_save_prefill_only(self):
        prefill_df = pd.DataFrame({"C": [9]})
        save_dataframes(prefill_df, None, self.test_dir)
        self._assert_file_exists_with_data("prefill_info.csv", prefill_df)
        self._assert_file_not_exists("decode_info.csv")

     # Test save only decode
    def test_save_decode_only(self):
        decode_df = pd.DataFrame({"Z": [10]})
        save_dataframes(None, decode_df, self.test_dir)
        self._assert_file_not_exists("prefill_info.csv")
        self._assert_file_exists_with_data("decode_info.csv", decode_df)

    # Test Empty data frame processing
    def test_prefill_empty_dataframes(self):
        save_dataframes(pd.DataFrame(), pd.DataFrame({"D": [11]}), self.test_dir)
        self._assert_file_not_exists("prefill_info.csv")
        self._assert_file_exists_with_data("decode_info.csv", pd.DataFrame({"D": [11]}))

    def test_decode_empty_dataframes(self):
        save_dataframes(pd.DataFrame({"E": [12]}), pd.DataFrame(), self.test_dir)
        self._assert_file_exists_with_data("prefill_info.csv", pd.DataFrame({"E": [12]}))
        self._assert_file_not_exists("decode_info.csv")

    # Test Both are empty
    def test_both_empty_dataframes(self):
        save_dataframes(pd.DataFrame(), pd.DataFrame(), self.test_dir)
        self._assert_file_not_exists("prefill_info.csv")
        self._assert_file_not_exists("decode_info.csv")
    
    # Test Overwriting existing files
    def test_overwrite_existing_files(self):
        df1 = pd.DataFrame({"H": [15]})
        save_dataframes(df1, None, self.test_dir)
        self._assert_file_exists_with_data("prefill_info.csv", df1)

        df2 = pd.DataFrame({"H": [16, 17]})
        save_dataframes(df2, None, self.test_dir)
        self._assert_file_exists_with_data("prefill_info.csv", df2)


class TestProcessC2LB(unittest.TestCase):
    def setUp(self):
        """
        Initialize the mock object
        """
        self.mock_args = MagicMock()
        self.mock_args.num_redundancy_expert = 3
        self.mock_args.num_npus = 8
        self.output_dir = "/mock/output"

        # log capture
        self.log_capture = []
        self.logger = logging.getLogger("msit_logger")
        self.logger.setLevel(logging.INFO)

        class LogHandler(logging.Handler):
            def emit(handler, record):
                self.log_capture.append(record)

        self.logger.addHandler(LogHandler())

    def tearDown(self):
        """
        Clean up log processor
        """
        self.logger = logging.getLogger("msit_logger")
        self.logger.handlers = []

    # Test only process decoded data
    @patch("components.expert_load_balancing.elb.load_balancing.load_expert_popularity_csv")
    @patch("components.expert_load_balancing.elb.load_balancing.save_matrix_to_json")
    @patch("components.expert_load_balancing.elb.load_balancing.lb_and_intra_layer_affinity_redundancy_deploy")  # 修正补丁路径
    def test_process_decode_only(self, mock_algo, mock_save, mock_load):
        # test data
        mock_df = MagicMock(spec=pd.DataFrame)
        mock_df.shape = (10, 5)
        mock_df.empty = False
        mock_load.return_value = (mock_df, None)
        
        # algorithm return values
        mock_algo.return_value = [[1,2],[3,4]]

        process_c2lb(self.mock_args, self.output_dir)

        # Verify call chain
        mock_algo.assert_called_once_with(
            mock_df.to_numpy(),
            self.mock_args.num_redundancy_expert,
            self.mock_args.num_npus,
            5  
        )
        mock_save.assert_called_once_with(
            self.output_dir,
            "decode_global_deployment", 
            [[1,2], [3,4]]
        )
        # Verify log output
        self.assertIn("C2LB processed decode data", self.log_capture[1].getMessage())

    # Test Processing only prefill data
    @patch("components.expert_load_balancing.elb.load_balancing.load_expert_popularity_csv")
    @patch("components.expert_load_balancing.elb.load_balancing.save_matrix_to_json")
    @patch("components.expert_load_balancing.elb.load_balancing.lb_and_intra_layer_affinity_redundancy_deploy")
    def test_process_prefill_only(self, mock_algo, mock_save, mock_load):
        mock_df = MagicMock(spec=pd.DataFrame)
        mock_df.shape = (8, 3) 
        mock_df.empty = False
        mock_load.return_value = (None, mock_df) 
        
        process_c2lb(self.mock_args, self.output_dir)

        mock_algo.assert_called_once_with(
            mock_df.to_numpy(),
            self.mock_args.num_redundancy_expert,
            self.mock_args.num_npus,
            3
        )
        mock_save.assert_called_once_with(
            self.output_dir,
            "prefill_global_deployment", 
            mock_algo.return_value
        )

        self.assertIn("C2LB processed prefill data", self.log_capture[1].getMessage())

    # Process two data sets simultaneously
    @patch("components.expert_load_balancing.elb.load_balancing.load_expert_popularity_csv")
    @patch("components.expert_load_balancing.elb.load_balancing.save_matrix_to_json")
    @patch("components.expert_load_balancing.elb.load_balancing.lb_and_intra_layer_affinity_redundancy_deploy")
    def test_process_both_datasets(self, mock_algo, mock_save, mock_load):
        decode_df = MagicMock(spec=pd.DataFrame)
        decode_df.shape = (5, 2)
        prefill_df = MagicMock(spec=pd.DataFrame)
        prefill_df.shape = (5, 3)
        mock_load.return_value = (decode_df, prefill_df)
        
        mock_algo.side_effect = [
            [[1,1]],
            [[2,2]] 
        ]

        process_c2lb(self.mock_args, self.output_dir)

        self.assertEqual(mock_algo.call_count, 2)
        calls = [
            call(decode_df.to_numpy(), 3, 8, 2),
            call(prefill_df.to_numpy(), 3, 8, 3)
        ]
        mock_algo.assert_has_calls(calls, any_order=False)

        mock_save.assert_has_calls([
            call(self.output_dir, "decode_global_deployment", [[1,1]]),
            call(self.output_dir, "prefill_global_deployment", [[2,2]])
        ])

    # Test Algorithm execution failed
    @patch("components.expert_load_balancing.elb.load_balancing.load_expert_popularity_csv")
    @patch("components.expert_load_balancing.elb.load_balancing.lb_and_intra_layer_affinity_redundancy_deploy")
    def test_algorithm_failure(self, mock_algo, mock_load):
        mock_df = MagicMock(spec=pd.DataFrame)
        mock_df.shape = (4, 2)
        mock_load.return_value = (mock_df, None)
        
        mock_algo.side_effect = ValueError("Invalid input shape")

        with self.assertRaises(RuntimeError) as cm:
            process_c2lb(self.mock_args, self.output_dir)
        self.assertIn("Failed to process decode data", str(cm.exception))
        self.assertIn("Invalid input shape", str(cm.exception.__cause__))
    
     # Test No valid data file
    @patch("components.expert_load_balancing.elb.load_balancing.load_expert_popularity_csv")
    def test_no_data_files(self, mock_load):
        """验证空数据场景异常抛出"""
        mock_load.return_value = (None, None)
        with self.assertRaises(FileNotFoundError) as cm:
            process_c2lb(self.mock_args, self.output_dir)
        self.assertIn("No valid decode/prefill data found", str(cm.exception))


class TestProcessDynamicC2LB(unittest.TestCase):
    def setUp(self):
        """
        Initialize the mock object
        """
        self.mock_args = MagicMock()
        self.mock_args.num_redundancy_expert = 3
        self.mock_args.num_npus = 8
        self.mock_args.num_nodes = 4
        self.output_dir = "/mock/output"

        # log capture
        self.log_capture = []
        self.logger = logging.getLogger("msit_logger")
        self.logger.setLevel(logging.INFO)

        class LogHandler(logging.Handler):
            def emit(handler, record):
                self.log_capture.append(record)

        self.logger.addHandler(LogHandler())

    def tearDown(self):
        """
        Clean up log processor
        """
        self.logger = logging.getLogger("msit_logger")
        self.logger.handlers = []

    # Test only process decoded data
    @patch("components.expert_load_balancing.elb.load_balancing.load_expert_popularity_csv")
    @patch("components.expert_load_balancing.elb.load_balancing.lb_redundancy_deploy_for_dynamic")  # 修正补丁路径
    @patch("components.expert_load_balancing.elb.load_balancing.save_matrix_to_json")
    def test_process_decode_only(self, mock_save, mock_algo, mock_load):
        # test data
        mock_df = MagicMock(spec=pd.DataFrame)
        mock_df.shape = (10, 5)
        mock_df.empty = False
        mock_load.return_value = (mock_df, None)
        
        # algorithm return values
        mock_algo.return_value = [[1,2],[3,4]]

        process_dynamic_c2lb(self.mock_args, self.output_dir)

        # Verify call chain
        mock_algo.assert_called_once_with(
            mock_df.to_numpy(),
            self.mock_args.num_redundancy_expert,
            self.mock_args.num_nodes,
            self.mock_args.num_npus
        )
        mock_save.assert_called_once_with(
            self.output_dir,
            "decode_global_deployment", 
            [[1,2], [3,4]]
        )
        # Verify log output
        self.assertIn("C2LB processed decode data", self.log_capture[1].getMessage())
    
    # Test Processing only prefill data
    @patch("components.expert_load_balancing.elb.load_balancing.load_expert_popularity_csv")
    @patch("components.expert_load_balancing.elb.load_balancing.lb_redundancy_deploy_for_dynamic") 
    def test_process_prefill_only(self, mock_algo, mock_load):
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.to_numpy.return_value = "mock_prefill_data"
        mock_load.return_value = (None, mock_df)
        
        mock_algo.side_effect = ValueError("模拟算法错误")

        with self.assertRaises(RuntimeError) as cm:
            process_dynamic_c2lb(self.mock_args, self.output_dir)
        self.assertIn("Failed to process decode data", str(cm.exception))
        self.assertIn("模拟算法错误", str(cm.exception.__cause__))

     # Test Processing two datasets simultaneously
    @patch("components.expert_load_balancing.elb.load_balancing.load_expert_popularity_csv")
    @patch("components.expert_load_balancing.elb.load_balancing.lb_redundancy_deploy_for_dynamic")  # 修正补丁路径
    @patch("components.expert_load_balancing.elb.load_balancing.save_matrix_to_json")
    def test_process_both_datasets(self, mock_save, mock_algo, mock_load):
        decode_df = MagicMock(spec=pd.DataFrame)
        prefill_df = MagicMock(spec=pd.DataFrame)
        decode_df.empty = prefill_df.empty = False
        mock_load.return_value = (decode_df, prefill_df)
        
        mock_algo.side_effect = [
            [[1,2]],
            [[3,4]]
        ]

        process_dynamic_c2lb(self.mock_args, self.output_dir)

        self.assertEqual(mock_algo.call_count, 2)
        calls = [
            call(decode_df.to_numpy(), 3, 4, 8),
            call(prefill_df.to_numpy(), 3, 4, 8)
        ]
        mock_algo.assert_has_calls(calls)

        mock_save.assert_has_calls([
            call(self.output_dir, "decode_global_deployment", [[1, 2]]),
            call(self.output_dir, "prefill_global_deployment", [[3, 4]])
        ])

     # Test Empty data file processing
    @patch("components.expert_load_balancing.elb.load_balancing.load_expert_popularity_csv")
    def test_empty_datafiles(self, mock_load):
        mock_load.return_value = (None, None)
        with self.assertRaises(FileNotFoundError) as cm:
            process_dynamic_c2lb(self.mock_args, self.output_dir)
        self.assertIn(f"No valid decode/prefill data found in {self.output_dir}", str(cm.exception))

    # Test Invalid data format processing
    @patch("components.expert_load_balancing.elb.load_balancing.load_expert_popularity_csv")
    def test_invalid_data_format(self, mock_load):
        mock_load.return_value = ("invalid_type", None)
        with self.assertRaises(RuntimeError):
            process_dynamic_c2lb(self.mock_args, self.output_dir)


class TestProcessC2LBA3(unittest.TestCase):
    def setUp(self):
        """
        Initialize the mock object
        """
        self.mock_args = MagicMock()
        self.mock_args.num_redundancy_expert = 3
        self.mock_args.num_npus = 8
        self.output_dir = "/mock/output"

        # log capture
        self.log_capture = []
        self.logger = logging.getLogger("msit_logger")
        self.logger.setLevel(logging.INFO)

        class LogHandler(logging.Handler):
            def emit(handler, record):
                self.log_capture.append(record)

        self.logger.addHandler(LogHandler())

    def tearDown(self):
        """
        Clean up log processor
        """
        self.logger = logging.getLogger("msit_logger")
        self.logger.handlers = []

    # Test only process decoded data
    @patch("components.expert_load_balancing.elb.load_balancing.load_expert_popularity_csv")
    @patch("components.expert_load_balancing.elb.load_balancing.save_matrix_to_json_a3")
    @patch("components.expert_load_balancing.elb.load_balancing.lb_and_intra_layer_affinity_redundancy_deploy_a3")  # 修正补丁路径
    def test_process_decode_only(self, mock_algo, mock_save, mock_load):
        # test data
        mock_df = MagicMock(spec=pd.DataFrame)
        mock_df.shape = (10, 5)
        mock_df.empty = False
        mock_load.return_value = (mock_df, None)
        
        # algorithm return values
        mock_algo.return_value = [[1,2],[3,4]]

        process_c2lb_a3(self.mock_args, self.output_dir)

        # Verify call chain
        mock_algo.assert_called_once_with(
            mock_df.to_numpy(),
            self.mock_args.num_redundancy_expert,
            self.mock_args.num_npus,
            5  
        )
        mock_save.assert_called_once_with(
            self.output_dir,
            "decode_global_deployment", 
            [[1,2], [3,4]]
        )
        # Verify log output
        self.assertIn("C2LB A3 processed decode data", self.log_capture[1].getMessage())

    # Test Processing only prefill data
    @patch("components.expert_load_balancing.elb.load_balancing.load_expert_popularity_csv")
    @patch("components.expert_load_balancing.elb.load_balancing.save_matrix_to_json_a3")
    @patch("components.expert_load_balancing.elb.load_balancing.lb_and_intra_layer_affinity_redundancy_deploy_a3")
    def test_process_prefill_only(self, mock_algo, mock_save, mock_load):
        mock_df = MagicMock(spec=pd.DataFrame)
        mock_df.shape = (8, 3) 
        mock_df.empty = False
        mock_load.return_value = (None, mock_df) 
        
        process_c2lb_a3(self.mock_args, self.output_dir)

        mock_algo.assert_called_once_with(
            mock_df.to_numpy(),
            self.mock_args.num_redundancy_expert,
            self.mock_args.num_npus,
            3
        )
        mock_save.assert_called_once_with(
            self.output_dir,
            "prefill_global_deployment", 
            mock_algo.return_value
        )

        self.assertIn("C2LB A3 processed prefill data", self.log_capture[1].getMessage())

    # Process two data sets simultaneously
    @patch("components.expert_load_balancing.elb.load_balancing.load_expert_popularity_csv")
    @patch("components.expert_load_balancing.elb.load_balancing.save_matrix_to_json_a3")
    @patch("components.expert_load_balancing.elb.load_balancing.lb_and_intra_layer_affinity_redundancy_deploy_a3")
    def test_process_both_datasets(self, mock_algo, mock_save, mock_load):
        decode_df = MagicMock(spec=pd.DataFrame)
        decode_df.shape = (5, 2)
        prefill_df = MagicMock(spec=pd.DataFrame)
        prefill_df.shape = (5, 3)
        mock_load.return_value = (decode_df, prefill_df)
        
        mock_algo.side_effect = [
            [[1,1]],
            [[2,2]] 
        ]

        process_c2lb_a3(self.mock_args, self.output_dir)

        self.assertEqual(mock_algo.call_count, 2)
        calls = [
            call(decode_df.to_numpy(), 3, 8, 2),
            call(prefill_df.to_numpy(), 3, 8, 3)
        ]
        mock_algo.assert_has_calls(calls, any_order=False)

        mock_save.assert_has_calls([
            call(self.output_dir, "decode_global_deployment", [[1,1]]),
            call(self.output_dir, "prefill_global_deployment", [[2,2]])
        ])

    # Test Algorithm execution failed
    @patch("components.expert_load_balancing.elb.load_balancing.load_expert_popularity_csv")
    @patch("components.expert_load_balancing.elb.load_balancing.lb_and_intra_layer_affinity_redundancy_deploy_a3")
    def test_algorithm_failure(self, mock_algo, mock_load):
        mock_df = MagicMock(spec=pd.DataFrame)
        mock_df.shape = (4, 2)
        mock_load.return_value = (mock_df, None)
        
        mock_algo.side_effect = ValueError("Invalid input shape")

        with self.assertRaises(RuntimeError) as cm:
            process_c2lb_a3(self.mock_args, self.output_dir)
        self.assertIn("Failed to process decode data", str(cm.exception))
        self.assertIn("Invalid input shape", str(cm.exception.__cause__))
    
     # Test No valid data file
    @patch("components.expert_load_balancing.elb.load_balancing.load_expert_popularity_csv")
    def test_no_data_files(self, mock_load):
        """验证空数据场景异常抛出"""
        mock_load.return_value = (None, None)
        with self.assertRaises(FileNotFoundError) as cm:
            process_c2lb_a3(self.mock_args, self.output_dir)
        self.assertIn("No valid decode/prefill data found", str(cm.exception))



class TestSelectAlgorithm(unittest.TestCase):
    def setUp(self):
        self.mock_args = MagicMock()
        self.mock_args.output_dir = "/mock/output"
        self.file_names = ["decode_info.csv"]

    @patch("components.expert_load_balancing.elb.load_balancing.process_c2lb")
    def test_a2_c2lb(self, mock_process):
        self.mock_args.algorithm = "0"
        self.mock_args.device_type = "a2"
        
        select_algorithm(self.mock_args)
        mock_process.assert_called_once_with(self.mock_args, output_dir="/mock/output")

    @patch("components.expert_load_balancing.elb.load_balancing.process_speculative_moe")
    def test_a2_speculative_moe(self, mock_process):
        self.mock_args.algorithm = "1"
        self.mock_args.device_type = "a2"
        
        select_algorithm(self.mock_args)
        mock_process.assert_called_once_with(
            self.mock_args, 
        )

    @patch("components.expert_load_balancing.elb.load_balancing.process_dynamic_c2lb")
    def test_a2_dynamic_c2lb(self, mock_process):
        self.mock_args.algorithm = "2"
        self.mock_args.device_type = "a2"
        
        select_algorithm(self.mock_args)
        mock_process.assert_called_once_with(
            self.mock_args, 
            output_dir="/mock/output"
        )

    @patch("components.expert_load_balancing.elb.load_balancing.process_c2lb_a3")
    def test_a3_c2lb_valid(self, mock_process):
        self.mock_args.algorithm = "0"
        self.mock_args.device_type = "a3"
        self.mock_args.share_expert_devices = 0
        
        select_algorithm(self.mock_args)
        mock_process.assert_called_once_with(self.mock_args, output_dir="/mock/output")
    
    def test_a3_c2lb_invalid_share_expert(self):
        self.mock_args.algorithm = "0"
        self.mock_args.device_type = "a3"
        self.mock_args.share_expert_devices = 1
        
        with self.assertRaises(ValueError) as cm:
            select_algorithm(self.mock_args)
        self.assertIn("incorrect share expert devices", str(cm.exception))
    
    
    def test_invalid_combinations(self):
        test_cases = [
            ("Invalid-Algo", "a2", "无效算法类型"),
            ("0", "Invalid-Device", "无效设备类型"),
            ("Dynamic-C2LB", "4", "不支持的算法设备组合")
        ]

        for algo, device, desc in test_cases:
            with self.subTest(algo=algo, device=device, desc=desc):
                self.mock_args.algorithm = algo
                self.mock_args.device_type = device
                
                with self.assertRaises(ValueError) as cm:
                    select_algorithm(self.mock_args)
                self.assertIn("valid parameters", str(cm.exception))


class TestLoadBalancing:

    """测试负载均衡主函数 load_balancing 的类"""

    @pytest.fixture
    def mock_args(self):
        """生成模拟的 args 对象"""
        args = MagicMock()
        args.expert_popularity_csv_load_path = "/fake/path"
        args.output_dir = "/fake/output"
        return args

    @patch('components.expert_load_balancing.elb.load_balancing.check_dump_file_version')
    @patch('components.expert_load_balancing.elb.load_balancing.process_files_by_type')
    @patch('components.expert_load_balancing.elb.load_balancing.select_algorithm')
    def test_new_version_c2lb_algorithm(
        self, mock_select, mock_process, mock_check_dump, mock_args
    ):
        """场景1: 新版本数据 + C2LB算法"""
        mock_check_dump.return_value = "/new/path"
        mock_args.algorithm = ALGORITHM_C2LB

        load_balancing(mock_args)

        mock_check_dump.assert_called_once_with("/fake/path")
        mock_process.assert_has_calls([
            call("/new/path", "decode", "/fake/output"),
            call("/new/path", "prefill", "/fake/output")
        ])
        mock_select.assert_called_once_with(mock_args)

    @patch('components.expert_load_balancing.elb.load_balancing.check_dump_file_version')
    @patch('components.expert_load_balancing.elb.load_balancing.process_files_by_type')
    @patch('components.expert_load_balancing.elb.load_balancing.select_algorithm')
    def test_new_version_non_c2lb_algorithm(
        self, mock_select, mock_process, mock_check_dump, mock_args
    ):
        """场景2: 新版本数据 + 非C2LB算法"""
        mock_check_dump.return_value = "/new/path"
        mock_args.algorithm = "OTHER_ALGO"

        load_balancing(mock_args)

        mock_process.assert_not_called()
        mock_select.assert_called_once_with(mock_args)

    @patch('components.expert_load_balancing.elb.load_balancing.check_dump_file_version')
    @patch('components.expert_load_balancing.elb.load_balancing.check_file_type')
    @patch('components.expert_load_balancing.elb.load_balancing.merge_csv_columns')
    @patch('components.expert_load_balancing.elb.load_balancing.save_dataframes')
    @patch('components.expert_load_balancing.elb.load_balancing.select_algorithm')
    def test_old_version_prefill_only(
        self, mock_select, mock_save, mock_merge, mock_check_file, mock_check_dump, mock_args
    ):
        """场景3: 旧版本数据 + 仅prefill文件"""
        mock_check_dump.return_value = None
        mock_check_file.return_value = ([PREFILL], 3, 0)
        mock_merge.return_value = MagicMock()

        load_balancing(mock_args)

        mock_merge.assert_called_once_with("/fake/path", PREFILL)
        mock_save.assert_called_once_with(mock_merge.return_value, None, "/fake/output")
        mock_select.assert_called_once_with(mock_args)

    @patch('components.expert_load_balancing.elb.load_balancing.check_dump_file_version')
    @patch('components.expert_load_balancing.elb.load_balancing.check_file_type')
    @patch('components.expert_load_balancing.elb.load_balancing.merge_csv_columns')
    @patch('components.expert_load_balancing.elb.load_balancing.save_dataframes')
    @patch('components.expert_load_balancing.elb.load_balancing.select_algorithm')
    def test_old_version_decode_only(
        self, mock_select, mock_save, mock_merge, mock_check_file, mock_check_dump, mock_args
    ):
        """场景4: 旧版本数据 + 仅decode文件"""
        mock_check_dump.return_value = None
        mock_check_file.return_value = ([DECODE], 0, 5)
        mock_merge.return_value = MagicMock()

        load_balancing(mock_args)

        mock_merge.assert_called_once_with("/fake/path", DECODE)
        mock_save.assert_called_once_with(None, mock_merge.return_value, "/fake/output")
        mock_select.assert_called_once_with(mock_args)


    @patch('components.expert_load_balancing.elb.load_balancing.check_dump_file_version')
    @patch('components.expert_load_balancing.elb.load_balancing.check_file_type')
    @patch('components.expert_load_balancing.elb.load_balancing.save_dataframes')
    @patch('components.expert_load_balancing.elb.load_balancing.select_algorithm')
    def test_old_version_no_files(
        self, mock_select, mock_save, mock_check_file, mock_check_dump, mock_args
    ):
        """场景6: 旧版本数据 + 无有效文件"""
        mock_check_dump.return_value = None
        mock_check_file.return_value = ([], 0, 0)

        load_balancing(mock_args)

        mock_save.assert_called_once_with(None, None, "/fake/output")
        mock_select.assert_called_once_with(mock_args)


class TestIsFileReadable:
    """测试文件可读性检测函数的测试套件"""
    
    def test_valid_csv(self, tmp_path):
        """测试可读取的有效CSV文件"""
        # 创建临时CSV文件
        file_path = tmp_path / "valid.csv"
        file_path.write_text("id,name\n1,Alice\n2,Bob")
        
        assert is_file_readable(str(file_path)) is True
    
    @pytest.mark.parametrize("exception,error_msg", [
        (pd.errors.ParserError, "Error tokenizing data"),
        (FileNotFoundError, "[Errno 2] No such file"),
        (PermissionError, "[Errno 13] Permission denied"),
        (MemoryError, "Not enough memory"),
        (Exception, "Unexpected error")
    ])
    def test_read_errors(self, exception, error_msg):
        """测试各种读取错误场景"""
        test_file = "problem.csv"
        
        with patch('pandas.read_csv') as mock_read, \
             patch('components.expert_load_balancing.elb.load_balancing.logger') as mock_logger:  # 替换为实际模块名
            
            # 配置模拟对象
            mock_read.side_effect = exception(error_msg)
            
            # 执行测试
            result = is_file_readable(test_file)
            
            # 验证结果
            assert result is False
            mock_logger.warning.assert_called_once_with(
                f"File {test_file} unable to read: {error_msg}"
            )


class TestCopyFile:
    """测试文件拷贝函数的测试套件"""
    
    # 模拟文件路径
    MAIN_FILE = "/data/main.csv"
    BAK_FILE = "/data/backup.csv"
    TARGET_FILE = "/output/result.csv"
    
    @patch('components.expert_load_balancing.elb.load_balancing.is_file_readable')  
    @patch('components.expert_load_balancing.elb.load_balancing.logger')
    @patch('shutil.copy')
    def test_main_file_readable(self, mock_copy, mock_logger, mock_is_readable):
        """测试主文件可读的情况"""
        # 设置模拟返回值
        mock_is_readable.side_effect = lambda f: f == self.MAIN_FILE
        
        # 执行测试
        copy_file(self.MAIN_FILE, self.BAK_FILE, self.TARGET_FILE)
        
        # 验证行为
        mock_copy.assert_called_once_with(self.MAIN_FILE, self.TARGET_FILE)
        mock_logger.debug.assert_called_once_with(f"copy: {self.MAIN_FILE} -> {self.TARGET_FILE}")

    @patch('components.expert_load_balancing.elb.load_balancing.is_file_readable')  
    @patch('components.expert_load_balancing.elb.load_balancing.logger')
    @patch('shutil.copy')
    @patch('os.path.exists', return_value=True)
    def test_backup_file_used(self, mock_exists, mock_copy, mock_logger, mock_is_readable):
        """测试主文件不可读但备份文件可用的情况"""
        # 主文件不可读，备份文件可读
        mock_is_readable.side_effect = lambda f: f == self.BAK_FILE
        
        # 执行测试
        copy_file(self.MAIN_FILE, self.BAK_FILE, self.TARGET_FILE)
        
        # 验证行为
        mock_copy.assert_called_once_with(self.BAK_FILE, self.TARGET_FILE)
        mock_logger.debug.assert_called_once_with(
            f"The original file is damaged, use the backup file: {self.BAK_FILE} -> {self.TARGET_FILE}"
        )
        mock_exists.assert_called_once_with(self.BAK_FILE)

    @patch('components.expert_load_balancing.elb.load_balancing.is_file_readable')  
    @patch('components.expert_load_balancing.elb.load_balancing.logger')
    @patch('shutil.copy')
    @patch('os.path.exists', return_value=False)
    def test_no_valid_files_exist(self, mock_exists, mock_copy, mock_logger, mock_is_readable):
        """测试主文件和备份文件都不可用的情况"""
        # 两个文件都不可读
        mock_is_readable.return_value = False
        
        # 验证抛出异常
        with pytest.raises(RuntimeError) as excinfo:
            copy_file(self.MAIN_FILE, self.BAK_FILE, self.TARGET_FILE)
        
        # 验证错误信息
        assert "The backup files are corrupted or do not exist!" in str(excinfo.value)
        
        # 验证没有发生复制操作
        mock_copy.assert_not_called()
        mock_exists.assert_called_with(self.BAK_FILE)


class TestCopyFilesWithRecovery:
    """测试文件恢复复制功能的测试套件"""
    
    # 基本文件结构
    BASE_SRC_DIR = "/test/source"
    BASE_DEST_DIR = "/test/destination"
    
    # 测试文件集合
    DECODE_FILES = ["decode_1.csv", "decode_5.csv", "decode_10.csv"]
    PREFILL_FILES = ["prefill_2.csv", "prefill_7.csv"]
    TOPK_FILES = [
        "decode_topk_1.csv", "decode_topk_5.csv", "decode_topk_10.csv",
        "prefill_topk_2.csv", "prefill_topk_7.csv"
    ]
    BAK_FILES = [
        "decode_1_bak.csv", "decode_5_bak.csv", "decode_10_bak.csv",
        "prefill_2_bak.csv", "prefill_7_bak.csv",
        "decode_topk_1_bak.csv", "decode_topk_10_bak.csv",
        "prefill_topk_2_bak.csv", "prefill_topk_7_bak.csv"
    ]
    JSON_FILES = ["config.json", "meta_data.json"]
    
    @pytest.fixture
    def mock_file_system(self):
        """模拟文件系统环境"""
        with patch("os.makedirs") as self.mock_makedirs, \
             patch("os.listdir") as self.mock_listdir, \
             patch("os.path.exists") as self.mock_exists, \
             patch("shutil.copy") as self.mock_shutil_copy, \
             patch("components.expert_load_balancing.elb.load_balancing.copy_file") as self.mock_copy_file, \
             patch("components.expert_load_balancing.elb.load_balancing.logger") as self.mock_logger:
            
            # 默认设置 - 所有文件都存在
            self.mock_exists.side_effect = lambda path: True
            yield

    def setup_files(self, 
                  decode_files=DECODE_FILES, 
                  prefill_files=PREFILL_FILES, 
                  topk_files=TOPK_FILES,
                  bak_files=BAK_FILES,
                  json_files=JSON_FILES):
        """设置模拟的文件列表"""
        all_files = decode_files + prefill_files + topk_files + bak_files + json_files
        self.mock_listdir.return_value = all_files

    @pytest.mark.parametrize("file_type", ["decode", "prefill"])
    def test_file_sorting(self, file_type, mock_file_system):
        """测试文件排序逻辑是否正确"""
        # 乱序文件列表
        files = [f"{file_type}_10.csv", f"{file_type}_2.csv", f"{file_type}_5.csv"]
        self.setup_files(
            decode_files=files if file_type == "decode" else [],
            prefill_files=files if file_type == "prefill" else [],
        )
        
        copy_files_with_recovery(self.BASE_SRC_DIR, self.BASE_DEST_DIR)
        
        # 验证调用顺序是按数字排序的
        calls = self.mock_copy_file.call_args_list
        numbers = [int(re.search(rf"{file_type}_(\d+).csv", call[0][2])).group(1) 
                  for call in calls if call[0][2].startswith(f"{file_type}_")]
        
        assert numbers == sorted(numbers), "文件未按数字顺序处理"

    def test_successful_recovery(self, mock_file_system):
        """测试成功恢复所有文件"""
        self.setup_files()
        copy_files_with_recovery(self.BASE_SRC_DIR, self.BASE_DEST_DIR)
        
        # 验证目录创建
        self.mock_makedirs.assert_called_once_with(self.BASE_DEST_DIR, exist_ok=True)
        
        # 验证文件复制调用次数
        expected_calls = len(self.DECODE_FILES + self.PREFILL_FILES + self.TOPK_FILES)
        assert self.mock_copy_file.call_count == expected_calls
        
        # 验证JSON文件复制
        assert self.mock_shutil_copy.call_count == len(self.JSON_FILES)
        
        # 验证日志输出
        self.mock_logger.debug.assert_any_call("=== dealing decode_* file ===")
        self.mock_logger.debug.assert_any_call("\n=== dealing prefill_* file ===")


    def test_no_json_files(self, mock_file_system):
        """测试没有JSON文件的情况"""
        self.setup_files(json_files=[])
        
        with pytest.raises(FileNotFoundError) as excinfo:
            copy_files_with_recovery(self.BASE_SRC_DIR, self.BASE_DEST_DIR)
        assert "dict has no JSON file" in str(excinfo.value)



class TestExtractNumber:
    """测试提取文件名数字的函数"""

    @pytest.mark.parametrize("filename, expected", [
        # 基本格式测试
        ("decode_123.csv", 123),
        ("prefill_456.csv", 456),
        ("decode_0.csv", 0),  # 边界值0
        
        # 多种数字格式测试
        ("decode_1000000.csv", 1000000),  # 大数字
        ("prefill_00042.csv", 42),  # 前导零
        ("decode_2147483647.csv", 2147483647),  # int32最大正值
        ("prefill_4294967295.csv", 4294967295),  # uint32最大值
        
        # 边界情况测试
        ("", None),  # 空文件名
        ("decode_.csv", None),  # 缺少数字
        ("decode_no_number.csv", None),  # 没有数字
        ("prefill.csv", None),  # 只有前缀
        ("12345.csv", None),  # 只有数字
        ("decode_123.bak.csv", None),  # 中间扩展名
        ("bak_decode_123.csv", None),  # 前缀错误
        
        # 数字边界测试
        ("decode_-123.csv", None),  # 负数
        ("prefill_123.45.csv", None),  # 浮点数
        
        # 国际化测试
        ("decode_123测试.csv", None),  # 文件名包含非ASCII字符
        ("解码_123.csv", None),  # 中文前缀
        ("prefill_абвгд.csv", None),  # 西里尔字母
    ])
    def test_extract_number(self, filename, expected):
        """测试各种文件名格式的数字提取"""
        result = extract_number(filename)
        assert result == expected


class TestProcessFilesByType:

    @patch("os.listdir", return_value=["decode_1.csv", "decode_2.csv", "prefill_1.csv"])
    @patch("os.path.exists", return_value=True)
    @patch("pandas.read_csv")
    @patch("json.load", return_value={"num_moe_layers": 4})
    @patch("pandas.DataFrame.to_csv")
    @patch("components.expert_load_balancing.elb.load_balancing.logger")
    @patch("components.expert_load_balancing.elb.load_balancing.ms_open", create=True)  # 使用 create=True 以防 ms_open 未定义
    def test_normal_processing(self, mock_ms_open, mock_logger, mock_to_csv, mock_json, 
                              mock_read_csv, mock_exists, mock_listdir):
        """测试正常处理流程"""
        # 设置测试环境
        src_dir = "/data/source"
        output_path = "/data/output"
        file_type = "decode"
        
        # 创建实际的DataFrame
        df1 = pd.DataFrame({
            'a': np.arange(20),
            'b': np.arange(20) + 10
        })
        
        df2 = pd.DataFrame({
            'c': np.arange(25),
            'd': np.arange(25) + 20
        })
        
        # 配置read_csv返回不同的DataFrame
        mock_read_csv.side_effect = [df1, df2]
        
        # 模拟ms_open和文件内容
        mock_file = MagicMock()
        mock_ms_open.return_value.__enter__.return_value = mock_file
        
        # 执行测试
        process_files_by_type(src_dir, file_type, output_path)
        
        # 验证JSON文件是否正确打开
        json_path = f"{src_dir}/model_gen_config.json"
        mock_ms_open.assert_called_once_with(json_path, 'r', encoding='utf-8')
        
        # 验证文件读取次数
        assert mock_read_csv.call_count == 2
        mock_logger.info.assert_any_call(f"[{file_type}] num_moe_layers = 4 Minimum number of rows = 20")
        mock_logger.info.assert_any_call(f"Merge completed, results saved to:{output_path}/decode_info.csv")
        
        # 验证输出是否正确保存
        assert mock_to_csv.call_count == 1
        output_arg = mock_to_csv.call_args[0][0]
        assert output_arg == f"{output_path}/decode_info.csv"