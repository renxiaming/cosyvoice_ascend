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
import unittest
import tempfile
import pickle

from msserviceprofiler.msguard import InvalidParameterError
from msserviceprofiler.msguard.security import (
    is_safe_csv_value,
    sanitize_csv_value,
    CSVInjectionError,
    PickleInjectionError,
    pickle_load_s,
    pickle_loads_s,
    open_s
)


class TestCSVInjection(unittest.TestCase):
    def test_is_safe_csv_value_with_numeric_string(self):
        self.assertTrue(is_safe_csv_value("123.45"), 
                       "数字字符串应该被认为是安全的")

    def test_is_safe_csv_value_with_safe_string(self):
        self.assertTrue(is_safe_csv_value("normal text"), 
                       "普通文本应该被认为是安全的")

    def test_is_safe_csv_value_with_unsafe_prefix(self):
        self.assertFalse(is_safe_csv_value("=cmd"), 
                        "以=开头的字符串可能包含CSV注入代码，应该被认为不安全")

    def test_is_safe_csv_value_with_unsafe_infix(self):
        self.assertFalse(is_safe_csv_value("text;=cmd"), 
                        "包含特殊字符的字符串可能包含CSV注入代码，应该被认为不安全")

    def test_is_safe_csv_value_with_non_string(self):
        self.assertTrue(is_safe_csv_value(123), 
                       "非字符串类型应该被认为是安全的")

    def test_sanitize_csv_value_with_safe_string(self):
        self.assertEqual(sanitize_csv_value("safe"), "safe", 
                         "安全字符串应该原样返回")

    def test_sanitize_csv_value_with_unsafe_string_strict(self):
        with self.assertRaises(CSVInjectionError, 
                              msg="replace False 下不安全字符串应该抛出CSVInjectionError"):
            sanitize_csv_value("=cmd", replace=False)

    def test_sanitize_csv_value_with_unsafe_string_replace(self):
        self.assertEqual(sanitize_csv_value("=cmd", replace=True), "'=cmd", 
                         "replace模式下不安全字符串应该被转义")


class TestPickleInjection(unittest.TestCase):
    class DangerPerson:
        def __init__(self):
            self.name = 'echo "This is a malicious command"'

        def __reduce__(self):
            return (os.system, (self.name,))
    
    class NicePerson:
        def __init__(self):
            self.name = 2

        def __reduce__(self):
            return (int, (self.name,))

    def setUp(self):
        # 创建临时测试数据
        self.safe_data = pickle.dumps(self.NicePerson())
        self.unsafe_data = pickle.dumps(self.DangerPerson())

    def test_pickle_loads_s_success_with_default_callback(self):
        """测试使用默认安全回调成功加载安全数据"""
        result = pickle_loads_s(self.safe_data)  # 使用默认回调
        self.assertEqual(result, 2, "应能正确加载基础 int bre类型")

    def test_pickle_loads_s_fail_with_default_callback(self):
        """测试默认安全回调应阻止危险对象"""
        with self.assertRaises(PickleInjectionError, msg="默认回调应阻止os.system"):
            pickle_loads_s(self.unsafe_data)

    def test_pickle_load_s_success_with_custom_callback(self):
        """测试自定义回调允许特定类型"""
        def custom_callback(module, name):
            return module == "builtins" and name == "int"

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.pkl")
            with open_s(test_file, "wb") as f:
                f.write(self.safe_data)

            result = pickle_load_s(test_file, fn=custom_callback)
            self.assertEqual(result, 2, "自定义回调应允许 int 类型")

    def test_pickle_load_s_fail_with_custom_callback(self):
        """测试自定义回调应阻止非允许类型"""
        def custom_callback(module, name):
            return False  # 拒绝所有类型

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.pkl")
            with open_s(test_file, "wb") as f:
                f.write(self.safe_data)

            with self.assertRaises(PickleInjectionError, msg="自定义回调拒绝所有类型时应报错"):
                pickle_load_s(test_file, fn=custom_callback)

    def test_pickle_load_s_file_handle(self):
        """测试直接传递文件句柄"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.pkl")
            with open_s(test_file, "wb") as f:
                f.write(self.safe_data)

            with open_s(test_file, "rb") as f:
                result = pickle_load_s(f)
                self.assertEqual(result, 2, "应能正确处理文件句柄输入")

    def test_pickle_loads_s_invalid_input(self):
        """测试非bytes输入应报TypeError"""
        with self.assertRaises(TypeError, msg="非bytes输入应触发类型错误"):
            pickle_loads_s("not bytes data")  # type: ignore

    def test_pickle_load_s_nonexistent_file(self):
        """测试不存在的文件路径应报FileNotFoundError"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = os.path.join(tmpdir, "nonexistent.pkl")
            with self.assertRaises(InvalidParameterError, msg="不存在的文件应触发错误"):
                pickle_load_s(bad_path)

    def test_pickle_load_s_invalid_callback(self):
        """测试无效回调函数应报TypeError"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.pkl")
            with open_s(test_file, "wb") as f:
                f.write(self.safe_data)

            with self.assertRaises(TypeError, msg="非可调用对象作为回调应触发错误"):
                pickle_load_s(test_file, fn="not callable")  # type: ignore

    def test_pickle_load_s_with_malicious_poc(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "malicious.pickle")
            with open_s(file_path, "wb") as f:
                pickle.dump(self.DangerPerson(), f)

            with self.assertRaises(PickleInjectionError,
                                  msg="应检测到包含系统命令的恶意pickle文件"):
                pickle_load_s(file_path)

    def test_pickle_loads_s_with_malicious_poc(self):
        malicious_data = pickle.dumps(self.DangerPerson())
        with self.assertRaises(PickleInjectionError,
                              msg="应检测到包含系统命令的pickle字节流"):
            pickle_loads_s(malicious_data)
