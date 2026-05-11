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

from msserviceprofiler.msguard.security import update_env_s
from msserviceprofiler.msguard.security import open_s


class TestUpdateEnvS(unittest.TestCase):
    def setUp(self):
        # 备份可能被修改的环境变量
        self.env_backup = os.environ.copy()

    def tearDown(self):
        # 恢复原始环境变量
        os.environ.clear()
        os.environ.update(self.env_backup)

    def test_update_env_s_with_valid_absolute_path(self):
        """测试函数能否正确处理有效的绝对路径"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_var = "TEST_VAR"
            test_path = os.path.join(temp_dir, "test_file")
            open_s(test_path, "a").close()

            update_env_s(test_var, test_path)
            self.assertEqual(os.environ[test_var], test_path,
                            "环境变量应被设置为指定的绝对路径")

    def test_update_env_s_with_non_string_env_var(self):
        """测试函数能否正确处理非字符串环境变量名（应抛出TypeError）"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = os.path.join(temp_dir, "test_file")
            open_s(test_path, "a").close()

            with self.assertRaises(TypeError,
                                 msg="非字符串环境变量名应该引发TypeError异常"):
                update_env_s(123, test_path)

    def test_update_env_s_with_empty_env_var(self):
        """测试函数能否正确处理空环境变量"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_var = "EMPTY_VAR"
            test_path = os.path.join(temp_dir, "test_file")
            open_s(test_path, "a").close()

            if test_var in os.environ:
                del os.environ[test_var]

            update_env_s(test_var, test_path)
            self.assertEqual(os.environ[test_var], test_path,
                            "空环境变量应被设置为指定的路径")

    def test_update_env_s_append_mode(self):
        """测试追加模式（prepend=False）能否正确工作"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_var = "APPEND_TEST"
            test_path = os.path.join(temp_dir, "test_file")
            open_s(test_path, "a").close()
            original_value = "/original/path"

            os.environ[test_var] = original_value
            update_env_s(test_var, test_path, prepend=False)
            expected = f"{original_value}{os.pathsep}{test_path}"
            self.assertEqual(os.environ[test_var], expected,
                            "路径应该被追加到现有值的末尾")

    def test_update_env_s_with_existing_value(self):
        """测试函数能否正确处理已有值的环境变量（默认前置模式）"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_var = "EXISTING_VAR"
            test_path = os.path.join(temp_dir, "test_file")
            open_s(test_path, "a").close()
            original_value = "/existing/path"
            os.environ[test_var] = original_value

            update_env_s(test_var, test_path)
            expected = f"{test_path}{os.pathsep}{original_value}"
            self.assertEqual(os.environ[test_var], expected,
                            "新路径应该被前置到现有值的前面")

    def test_update_env_s_prepend_false_with_existing_value(self):
        """测试追加模式（prepend=False）能否正确处理已有值的环境变量"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_var = "PREPEND_TEST"
            test_path = os.path.join(temp_dir, "test_file")
            open_s(test_path, "a").close()
            original_value = "/original/path"
            os.environ[test_var] = original_value

            update_env_s(test_var, test_path, prepend=False)
            expected = f"{original_value}{os.pathsep}{test_path}"
            self.assertEqual(os.environ[test_var], expected,
                            "新路径应该被追加到现有值的末尾")

    def test_update_env_s_multiple_updates(self):
        """测试多次更新同一个环境变量能否正确工作"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_var = "MULTI_TEST"
            path1 = os.path.join(temp_dir, "path1")
            path2 = os.path.join(temp_dir, "path2")
            open_s(path1, "a").close()
            open_s(path2, "a").close()

            update_env_s(test_var, path1)
            self.assertEqual(os.environ[test_var], path1,
                            "第一次更新后环境变量应等于第一个路径")

            update_env_s(test_var, path2)
            expected = f"{path2}{os.pathsep}{path1}"
            self.assertEqual(os.environ[test_var], expected,
                            "第二次更新后新路径应前置到现有值前面")
