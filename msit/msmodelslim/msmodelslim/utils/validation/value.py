# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Any, List
from msmodelslim.utils.exception import SchemaValidateError


def greater_than_zero(v: float) -> float:
    if v <= 0:
        raise SchemaValidateError("value must be greater than 0", 
                                  action="Please check the float value")
    return v


def validate_normalized_value(v: Any) -> float:
    if not isinstance(v, (float, type(None))):
        raise SchemaValidateError("value must be a float or None type",
                                  action="Please provide a float or None value")
    if v is not None and (v <= 0 or v >= 1):
        raise SchemaValidateError("value must be in the range (0, 1)",
                                  action="Please check the float value to ensure it is between 0 and 1")
    return v


def is_boolean(v: Any) -> bool:
    if not isinstance(v, bool):
        raise SchemaValidateError("value must be a boolean type", 
                                  action="Please provide a boolean value (True or False)")
    return v


def is_string_list(v: Any) -> List[str]:
    if not isinstance(v, list):
        raise SchemaValidateError("value must be a list type", 
                                  action="Please provide a list value")
    
    for item in v:
        if not isinstance(item, str):
            raise SchemaValidateError("all elements in the list must be string types", 
                                      action="Please ensure all list elements are strings")
    
    return v
