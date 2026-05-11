#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from typing import Any, Dict, Optional, Tuple, Set, Callable

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle
from transformers import Cache

from msmodelslim.utils.logging import get_logger

CONST_PAST_KEY_VALUES = 'past_key_values'
CONST_PAST_KEY_VALUE = 'past_key_value'


class KVCacheListener(Cache):
    def __init__(self, listen_helper: Callable[[int, torch.Tensor, torch.Tensor], None], cache: Optional[Cache] = None):
        super().__init__()
        self.cache = cache
        self.listen_helper = listen_helper

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.listen_helper(layer_idx, key_states, value_states)

        if self.cache is None:
            return key_states, value_states
        return self.cache.update(key_states, value_states, layer_idx, cache_kwargs)


class KVCacheListenerManager:
    def __init__(self):
        super().__init__()
        self.remove_handlers: Set[RemovableHandle] = set()

    def attach_listener_to_module(self, module: nn.Module,
                                  listen_helper: Callable[[int, torch.Tensor, torch.Tensor], None]) -> None:
        def pre_hook(_: nn.Module, args: Any, kwargs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
            if CONST_PAST_KEY_VALUES in kwargs:
                kwargs[CONST_PAST_KEY_VALUES] = KVCacheListener(listen_helper, cache=kwargs[CONST_PAST_KEY_VALUES])
            elif CONST_PAST_KEY_VALUE in kwargs:
                kwargs[CONST_PAST_KEY_VALUE] = KVCacheListener(listen_helper, cache=kwargs[CONST_PAST_KEY_VALUE])
            else:
                get_logger().warning(f"{CONST_PAST_KEY_VALUES} and {CONST_PAST_KEY_VALUE} both are None")
                kwargs[CONST_PAST_KEY_VALUES] = KVCacheListener(listen_helper)
                kwargs[CONST_PAST_KEY_VALUE] = KVCacheListener(listen_helper)
            return args, kwargs

        remove_handler = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
        self.remove_handlers.add(remove_handler)

    def remove_listeners(self) -> None:
        for handler in self.remove_handlers:
            handler.remove()
        self.remove_handlers.clear()
