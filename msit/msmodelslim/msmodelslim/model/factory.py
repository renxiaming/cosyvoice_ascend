# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Type, TypeVar, Optional

from msmodelslim.utils.exception import ToDoError, UnsupportedError
from msmodelslim.utils.logging import get_logger

model_map = {}

T = TypeVar('T')


class ModelFactory:
    @staticmethod
    def register(model_name: str):
        def decorator(cls) -> Type:
            if model_name in model_map:
                raise ToDoError(f"Model {model_name} already registered",
                                action=f'Please make sure {model_name} not registered')
            if not isinstance(cls, type):
                raise ToDoError(f"object {cls} is not a class",
                                action=f'Please make sure {cls} is a class')

            model_map[model_name] = cls
            return cls

        return decorator

    @staticmethod
    def create(model_name: str, interface: Optional[Type[T]] = None) -> Type[T]:
        original_model_name = model_name
        if model_name not in model_map:
            if 'default' in model_map:
                get_logger().warning(f"Model '{original_model_name}' not found in registered models. "
                              f"Using default model instead. "
                              f"Registered models: {list(model_map.keys())}")
                model_name = 'default'
            else:
                raise UnsupportedError(f"Model {model_name} not found",
                                       action=f"Please choose one in {list(model_map.keys())}")
        cls = model_map[model_name]
        if interface is not None and not issubclass(cls, interface):
            raise UnsupportedError(f"Model {model_name} not implements {interface.__name__}",
                                   action=f'Please change model or implement {interface.__name__}')
        return cls
