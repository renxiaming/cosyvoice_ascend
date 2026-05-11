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
import sys
import traceback
import importlib
import inspect
import functools
from collections import namedtuple
from abc import ABC, abstractmethod
from typing import Union, Tuple, List, Optional, Callable, Dict, Any
from packaging.version import Version
from .utils import logger

# 全局注册表存储所有hookers
HOOK_REGISTRY = []


def import_object_from_string(import_path: str, module_path: str) -> Any:
    """
    根据点分隔的路径导入对象（模块、类、函数等）
    支持多级嵌套属性，如 "module.submodule.ClassName.method_name"
    """
    if not import_path:
        logger.error(f"Module import_path is empty")
        return None

    try:
        module = importlib.import_module(import_path)
    except ImportError as e:
        logger.error(f"Module import failed for %s: %s", module_path, e)
        return None

    current = module
    for part in module_path.split("."):
        prev = current  # For log
        current = getattr(current, part, None)
        if current is None:
            logger.error(f"Module {prev} doesn't has attribute {part}")
            return None
    return current


class HookHelper:
    """辅助类，用于函数替换操作"""
    def __init__(self, ori_function_define, new_function):
        self.new_function = new_function
        self.ori_function = None
        self.location = None
        self.attr_name = None

        if ori_function_define is None:
            return

        if inspect.isfunction(ori_function_define) or inspect.ismethod(ori_function_define):
            self.ori_function = ori_function_define
            self.location, self.attr_name = self.get_location(self.ori_function)
        elif callable(ori_function_define):
            self.ori_function = ori_function_define.__call__
            self.location, self.attr_name = self.get_location(self.ori_function)

        if not all((self.ori_function, self.location, self.attr_name, self.new_function)):
            warn_msg = f"{ori_function_define} replace failed."
            logger.error(warn_msg)
            raise ValueError(warn_msg)

    @staticmethod
    def get_location(function_ins):
        if not hasattr(function_ins, "__module__"):
            warning_msg = f"function {str(function_ins)} has no __module__."
            logger.error(warning_msg)
            raise ValueError(warning_msg)

        module = importlib.import_module(function_ins.__module__)
        qualified_name = function_ins.__qualname__.split(".")
        classes = qualified_name[:-1]
        attr_name = qualified_name[-1]
        location = module

        for class_name in classes:
            location = getattr(location, class_name, None)
            if location is None:
                break

        if location is None:
            warning_msg = f"{'.'.join(classes)} does not exist"
            logger.error(warning_msg)
            raise ValueError(warning_msg)

        return location, attr_name

    def replace(self):
        if all((self.ori_function, self.location, self.attr_name, self.new_function)):
            setattr(self.location, self.attr_name, self._get_method(self.new_function))

    def recover(self):
        if all((self.ori_function, self.location, self.attr_name, self.new_function)):
            setattr(self.location, self.attr_name, self._get_method(self.ori_function))

    def _get_method(self, func):
        if inspect.isclass(self.location):
            try:
                func_cls_name = inspect.getattr_static(self.location, self.attr_name).__class__.__name__
                if func_cls_name in ("staticmethod", "classmethod"):
                    return staticmethod(func)
            except AttributeError:
                pass
        return func


def get_parents_name(ori_func, index=1):
    """获取调用栈中的父级函数名"""
    gen = traceback.walk_stack(None)
    try:
        for _ in range(index + 1):
            f = next(gen)
        return f[0].f_code.co_name
    except StopIteration:
        return None


class VLLMHookerBase(ABC):
    """Hooker基类，提供核心功能"""
    vllm_version = (None, None)  # (min_version, max_version)
    applied_hook_func_name = ""

    def __init__(self):
        self.hooks = []

    @abstractmethod
    def init(self):
        """初始化hook点，子类必须实现"""
        pass

    def replace_func(self, ori_func, pname, profiler_func):
        """创建替换函数"""
        @functools.wraps(ori_func)
        def wrapper(*args, **kwargs):
            if pname is not None and get_parents_name(ori_func) != pname:
                logger.debug(f"calling {ori_func}")
                return ori_func(*args, **kwargs)
            logger.debug(f"calling profiler_func={self.applied_hook_func_name} for {ori_func}")
            return profiler_func(*args, **kwargs)
        return wrapper

    def do_hook(self, hook_points, profiler_func_maker, pname=None):
        """执行实际的hook操作"""
        for ori_func in hook_points:
            if ori_func is None:
                continue
            profiler_func = profiler_func_maker(ori_func)
            cur_hook = HookHelper(ori_func, self.replace_func(ori_func, pname, profiler_func))
            cur_hook.replace()
            self.hooks.append(cur_hook)
            logger.debug(f"replacing {ori_func} with {self.applied_hook_func_name}")

    def support_version(self, version):
        """检查当前版本是否支持"""
        min_version = self.vllm_version[0]
        max_version = self.vllm_version[1]

        if min_version is not None and Version(min_version) > Version(version):
            logger.debug(f"min_version={min_version} is less than version={version}, skip")
            return False
        if max_version is not None and Version(max_version) < Version(version):
            logger.debug(f"max_version={max_version} is larger than version={version}, skip")
            return False
        return True

    def register(self):
        """注册hooker到全局注册表"""
        HOOK_REGISTRY.append(self)


def vllm_hook(
    hook_points: Union[Tuple[str, str], List[Tuple[str, str]]],
    min_version: Optional[str] = None,
    max_version: Optional[str] = None,
    caller_filter: Optional[str] = None
) -> Callable:
    """
    装饰器工厂函数，用于简化hooker创建
    :param hook_points: hook点列表，格式为(模块名, 属性路径)
    :param min_version: 支持的最小版本
    :param max_version: 支持的最大版本
    :param caller_filter: 调用者过滤条件
    """
    def decorator(hook_func):
        logger.debug(f"Handling {hook_func}")
        
        class AutoHooker(VLLMHookerBase):
            vllm_version = (min_version, max_version)
            applied_hook_func_name = getattr(hook_func, "__name__", str(hook_func))

            def init(self):
                hook_list = [hook_points] if isinstance(hook_points, tuple) else hook_points
                points = [import_object_from_string(import_path, func_path) for import_path, func_path in hook_list]
                self.do_hook(
                    hook_points=points,
                    profiler_func_maker=lambda ori_func: lambda *args, **kwargs: hook_func(ori_func, *args, **kwargs),
                    pname=caller_filter,
                )
        hooker = AutoHooker()
        hooker.register()
        return hook_func

    return decorator


def apply_hooks(version: str = None):
    """应用所有注册的hookers"""
    if version is None:
        import vllm
        version = vllm.__version__
    for hooker in HOOK_REGISTRY:
        if hooker.support_version(version):
            try:
                hooker.init()
            except Exception as e:
                logger.error(f"Failed to apply hooker: {str(e)}")
        else:
            logger.debug(f"Skipping hooker: {hooker.__class__.__name__} for version not matched")
