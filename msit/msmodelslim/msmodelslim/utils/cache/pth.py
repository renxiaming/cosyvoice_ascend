# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import os
from typing import Any, List, Dict, Callable
import functools
import inspect
import contextvars
import torch
import torch.nn as nn

from ascend_utils.common.security import get_valid_read_path
from ascend_utils.common.security import get_valid_write_path
from ascend_utils.common.security.pytorch import safe_torch_load
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import get_logger

MAX_RECURSION_DEPTH = 20
MAX_READ_FILE_SIZE_32G = 34359738368  # 32G, 32 * 1024 * 1024 * 1024


def load_cached_data(pth_file_path, generate_func, model, dump_config):
    """内部缓存加载函数"""
    try:
        # 检查文件是否存在
        if not os.path.exists(pth_file_path):
            raise FileNotFoundError(f"File {repr(pth_file_path)} does not exist")

        # 加载pth文件
        pth_file_path = get_valid_read_path(pth_file_path, size_max=MAX_READ_FILE_SIZE_32G)
        data = safe_torch_load(pth_file_path)
        get_logger().info(f"Successfully loaded data from %r", pth_file_path)
        return data
    except FileNotFoundError as e:
        pth_file_path = get_valid_write_path(pth_file_path)
        get_logger().info(f"Failed to load %r: %s, will regenerate", pth_file_path, e)

        # 触发缓存未命中时的处理逻辑
        # 配置dump参数
        dumper_manager = DumperManager(model, capture_mode=dump_config.capture_mode)

        # 执行推理获取激活值
        generate_func()

        # 返回dump的数据
        dumper_manager.save(pth_file_path)

        data = safe_torch_load(pth_file_path)
        get_logger().info(f"Successfully dumped and loaded data from %r", pth_file_path)
        return data


class InputCapture:
    """Handles capturing and storing function inputs and outputs."""

    _captured_inputs_var = contextvars.ContextVar("captured_inputs", default=[])

    @classmethod
    def reset(cls) -> None:
        """Reset all captured inputs."""
        cls._captured_inputs_var.set([])

    @classmethod
    def get_all(cls) -> List[Dict[str, Any]]:
        """Get all captured inputs."""
        return cls._captured_inputs_var.get()

    @classmethod
    def add_record(cls, record: Dict[str, Any]) -> None:
        """Add a new record to the captured inputs."""
        inputs = cls._captured_inputs_var.get()
        inputs.append(record)
        cls._captured_inputs_var.set(inputs)

    @classmethod
    def capture_forward_inputs(
            cls,
            func: Callable,
            capture_mode: str = 'args',
    ) -> Callable:
        """
        Decorator to capture inputs to a forward function.

        Args:
            func: Forward function to decorate
            capture_mode: 'args'

        Returns:
            Wrapped function
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature and bind arguments
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Handle 'self' for methods
            is_method = 'self' in sig.parameters
            captured_args = list(bound.args[1:]) if is_method else list(bound.args)

            captured_kwargs = bound.arguments.copy()
            if is_method and 'self' in captured_kwargs:
                del captured_kwargs['self']

            # Apply capture mode
            # 可拓展其他模式
            if capture_mode == 'args':
                captured_kwargs = {}
                record = captured_args
            else:
                raise SchemaValidateError(f"Invalid capture_mode: {capture_mode}."
                                          f"Must be 'args' ")

            # Execute original function
            result = func(*args, **kwargs)

            # Store record
            record = to_device(record, device='cpu')
            cls.add_record(record)

            return result

        return wrapper


class DumperManager(nn.Module):
    """Module that listens to and captures forward pass inputs and outputs."""

    def __init__(
            self,
            module: nn.Module,
            capture_mode: str = 'args',
    ):
        """
        Initialize a listener for the given module.

        Args:
            module: Module to listen to
            capture_mode: 'args'
        """
        super().__init__()
        self.module = module
        self.capture_mode = capture_mode
        self.old_forward = None

        if capture_mode not in {'args'}:
            raise SchemaValidateError(f"Invalid capture_mode: {capture_mode}."
                                      f"Must be 'args' ")

        self._add_hook(self.module)

    def save(self, path: str = '__output.pth') -> List[Dict[str, Any]]:
        """Save captured data and restore original forward method."""
        data = InputCapture.get_all()
        torch.save(data, path)

        # Restore original forward method
        if self.old_forward:
            self.module.forward = self.old_forward
            self.old_forward = None

        get_logger().info('Captured data saved to: %r', path)
        return data

    def reset(self) -> None:
        """Reset captured inputs."""
        InputCapture.reset()

    def _add_hook(self, module: nn.Module) -> Callable:
        """Add forward hook to the module."""
        self.old_forward = module.forward
        wrapper = InputCapture.capture_forward_inputs(
            self.old_forward,
            capture_mode=self.capture_mode,
        )
        module.forward = wrapper
        return wrapper


def to_device(data, device, depth=0):
    """ recursive function to move data to the specified device """
    if depth > MAX_RECURSION_DEPTH:
        raise RecursionError(f"Maximum recursion depth {MAX_RECURSION_DEPTH} exceeded")

    if isinstance(data, dict):
        return {k: to_device(v, device, depth=depth + 1) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(item, device, depth=depth + 1) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_device(item, device, depth=depth + 1) for item in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data
