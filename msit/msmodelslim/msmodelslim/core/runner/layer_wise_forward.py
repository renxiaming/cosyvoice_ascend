#  -*- coding: utf-8 -*-
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

from typing import List, Tuple, Dict, Optional, Any, Generator

import torch.nn
from torch import nn, distributed as dist

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.utils.exception import InvalidModelError


class _TransformersForwardBreak(Exception):
    """内部使用的异常类，用于中断前向传播并捕获输入"""
    pass


def generated_decoder_layer_visit_func(model: torch.nn.Module,
                                       transformer_blocks: Optional[List[Tuple[str, torch.nn.Module]]] = None,
                                       ) -> Generator[ProcessRequest, Any, None]:
    if transformer_blocks is None:
        transformer_blocks = [
            (name, module)
            for name, module in model.named_modules()
            if "decoderlayer" in module.__class__.__name__.lower()
        ]

    if dist.is_initialized():
        dist.barrier()

    for name, block in transformer_blocks:
        yield ProcessRequest(name, block, [], {})


def transformers_generated_forward_func(model: torch.nn.Module,
                                        inputs: Any,
                                        ) -> Generator[ProcessRequest, Any, None]:
    transformer_blocks = [
        (name, module)
        for name, module in model.named_modules()
        if "decoderlayer" in module.__class__.__name__.lower()
    ]

    # 存储第一个transformer block的输入
    first_block_input = None

    def break_hook(module: nn.Module, hook_args: Tuple[Any, ...], hook_kwargs: Dict[str, Any]):
        nonlocal first_block_input
        first_block_input = (hook_args, hook_kwargs,)
        raise _TransformersForwardBreak()

    hooks = [transformer_blocks[0][1].register_forward_pre_hook(break_hook, with_kwargs=True)]

    # 执行一次前向传播以获取输入
    try:
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            model(*inputs)
        elif isinstance(inputs, dict):
            model(**inputs)
        else:
            model(inputs)
    except _TransformersForwardBreak:
        pass
    except Exception as e:
        raise e
    finally:
        for hook in hooks:
            hook.remove()

    if first_block_input is None:
        raise InvalidModelError("Can't get first block input.", action="Please check the model and input")

    # 循环处理每个transformer block
    current_inputs = first_block_input

    if dist.is_initialized():
        dist.barrier()

    for name, block in transformer_blocks:
        args, kwargs = current_inputs
        outputs = yield ProcessRequest(name, block, args, kwargs)
        hidden_states = outputs[0]
        current_inputs = ((hidden_states,), current_inputs[1])
