#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.

import traceback
from importlib.metadata import entry_points
from typing import Dict, Type

from msmodelslim.app.quant_service import BaseQuantService
from msmodelslim.utils.exception import EnvError
from msmodelslim.utils.logging import get_logger

_QUANT_SERVICE_PLUGINS: Dict[str, Type[BaseQuantService]] = {}
_FAILED_PLUGINS: Dict[str, str] = {}  # 记录加载失败的插件及其错误信息


def load_plugins() -> Dict[str, Type[BaseQuantService]]:
    for entry in entry_points(group="msmodelslim.quant_service.plugins"):
        try:
            plugin_class = entry.load()
            if issubclass(plugin_class, BaseQuantService):
                _QUANT_SERVICE_PLUGINS[plugin_class.backend_name] = plugin_class
                get_logger().info(f"Load quant service plugin {entry.name} success!")
            else:
                error_msg = f"Plugin {entry.name} is not a subclass of BaseQuantService"
                _FAILED_PLUGINS[entry.name] = error_msg
                get_logger().warning(f"Failed to load plugin {entry.name}: {error_msg}")
        except Exception as e:
            error_msg = f"Exception: {str(e)}\nTraceback: {traceback.format_exc()}"
            _FAILED_PLUGINS[entry.name] = error_msg
            get_logger().warning(f"Failed to load plugin {entry.name}: {e}")
    return _QUANT_SERVICE_PLUGINS


def load_quant_service_cls(backend_name: str) -> Type[BaseQuantService]:
    # 首先检查请求的后端是否在失败列表中
    if backend_name in _FAILED_PLUGINS:
        raise EnvError(
            f"Quant service plugin for backend '{backend_name}' failed to load:\n{_FAILED_PLUGINS[backend_name]}")

    if backend_name not in _QUANT_SERVICE_PLUGINS:
        available_backends = list(_QUANT_SERVICE_PLUGINS.keys())
        raise EnvError(
            f"No quant service plugin found for backend '{backend_name}'. Available backends: {available_backends}")

    return _QUANT_SERVICE_PLUGINS[backend_name]
