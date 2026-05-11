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

import re
import json

from msguard.security import open_s

from .base import BaseCollector


class ConfigCollector(BaseCollector):
    def __init__(self, error_handler=None, *, config_path: str = None):
        super().__init__(error_handler)
        self.config_path = config_path

    @staticmethod
    def _parse_file_structure(file):
        """解析配置文件结构，记录缩进层级变化"""
        lines = []
        depth_changes = {}
        current_depth = 0

        for line_num, line_content in enumerate(file):
            line_content = line_content.rstrip('\n')
            lines.append(line_content)

            if not line_content.strip():
                continue

            previous_depth = current_depth
            for char in line_content:
                if char in '{[':
                    current_depth += 1
                elif char in '}]':
                    current_depth -= 1

            if previous_depth != current_depth:
                depth_changes[line_num] = current_depth
        
        return lines, depth_changes

    @staticmethod
    def _build_context_hierarchy(lines, depth_changes):
        """构建每行的上下文层级关系"""
        line_count = len(lines)
        context_hierarchy = [[] for _ in range(line_count)]

        active_blocks = []
        for line_num in range(line_count):
            if line_num in depth_changes:
                current_depth = depth_changes[line_num]
                
                if not active_blocks or current_depth > depth_changes[active_blocks[-1]]:
                    active_blocks.append(line_num)
                elif current_depth < depth_changes[active_blocks[-1]]:
                    active_blocks.pop()
                
                context_hierarchy[line_num] = active_blocks.copy()

        # 填充没有直接上下文的行
        previous_context = []
        for i, context in enumerate(context_hierarchy):
            if not context:
                context_hierarchy[i] = previous_context
            else:
                previous_context = context

        return context_hierarchy

    @staticmethod
    def _create_key_mapping(lines, json_data):
        """创建JSON键到文件位置的映射"""
        key_location_map = {}

        def _map_keys(node, current_path, start_line, processed_lines):
            if isinstance(node, dict):
                for key, value in node.items():
                    full_path = key if not current_path else f"{current_path}.{key}"
                    
                    # 在文件中查找键的位置
                    for line_num in range(start_line, len(lines)):
                        line_content = lines[line_num]
                        match = re.search(rf'\s*"{re.escape(key)}"\s*:', line_content)
                        
                        if match:
                            col_start = match.start(0) + len(match.group(0).split('"')[0])
                            key_location_map[full_path] = (line_num, col_start)
                            
                            if line_num not in processed_lines:
                                processed_lines.add(line_num)
                                _map_keys(value, full_path, line_num, processed_lines)
                            break
            elif isinstance(node, list):
                for index, item in enumerate(node):
                    list_path = f"{current_path}[{index}]"
                    _map_keys(item, list_path, start_line, processed_lines)

        processed_lines = set()
        _map_keys(json_data, "", 0, processed_lines)
        return key_location_map

    def _collect_data(self):
        file_lines, depth_info = self._read_file_lines_and_depth()
        if not file_lines:
            return {}, [], {}, []

        json_content = self._parse_json_content(file_lines)
        if not json_content:
            return {}, file_lines, {}, []

        key_mapping = self._create_key_mapping(file_lines, json_content)
        context_hierarchy = self._build_context_hierarchy(file_lines, depth_info)

        return json_content, file_lines, key_mapping, context_hierarchy

    def _read_file_lines_and_depth(self):
        try:
            with open_s(self.config_path, 'r', encoding='utf-8') as config_file:
                return self._parse_file_structure(config_file)
        except Exception as error:
            self.error_handler.add_error(
                filename=__file__,
                function='_read_file_lines_and_depth',
                lineno=131,
                what=f"尝试打开文件失败: {self.config_path!r}",
                reason=str(error)
            )
            return [], {}

    def _parse_json_content(self, file_lines):
        try:
            return json.loads('\n'.join(file_lines)) if file_lines else {}
        except Exception as error:
            self.error_handler.add_error(
                filename=__file__,
                function='_parse_json_content',
                lineno=143,
                what=f"尝试用 Json 格式解析文件失败: {self.config_path!r}",
                reason=str(error)
            )
            return {}


class UserConfigCollector(ConfigCollector):
    def __init__(self, error_handler=None, *, config_path=None):
        super().__init__(error_handler, config_path=config_path)
        self.error_handler.type = "user config"


class MindIEEnvCollector(ConfigCollector):
    def __init__(self, error_handler=None, *, config_path=None):
        super().__init__(error_handler, config_path=config_path)
        self.error_handler.type = "mindie env"


class ModelConfigCollector(ConfigCollector):
    def __init__(self, error_handler=None, *, config_path=None):
        super().__init__(error_handler, config_path=config_path)
        self.error_handler.type = "model config"


class MIESConfigCollector(ConfigCollector):
    def __init__(self, error_handler=None, *, config_path=None):
        super().__init__(error_handler, config_path=config_path)
        self.error_handler.type = "mies config"
