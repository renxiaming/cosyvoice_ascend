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


from typing import Optional, Tuple, Union, List

import torch
from pydantic import BaseModel
from torch import distributed as dist, nn

from msmodelslim.utils.exception import SpecError


class MinMaxObserverConfig(BaseModel):
    dim: Union[int, List[int]] = []
    keepdim: bool = False


class MsMinMaxObserver(nn.Module):

    def __init__(self, config: MinMaxObserverConfig):
        super().__init__()
        self.config = config
        self.min_val = None
        self.max_val = None

    def update(self, x: torch.Tensor, sync: bool = False, group: Optional[dist.ProcessGroup] = None):

        if self.min_val is None:
            self.min_val = torch.amin(x, self.config.dim, self.config.keepdim)
        else:
            self.min_val = torch.min(self.min_val, torch.amin(x, self.config.dim, self.config.keepdim))

        if self.max_val is None:
            self.max_val = torch.amax(x, self.config.dim, self.config.keepdim)
        else:
            self.max_val = torch.max(self.max_val, torch.amax(x, self.config.dim, self.config.keepdim))

        if sync and group:
            if not dist.is_initialized():
                raise SpecError("MinMaxStrategy's update_with_group requires distributed enabled",
                                action='Please make sure enable distributed')

            dist.all_reduce(self.min_val, op=dist.ReduceOp.MIN, group=group)
            dist.all_reduce(self.max_val, op=dist.ReduceOp.MAX, group=group)

    def reset(self):
        self.min_val = None
        self.max_val = None

    def get_min_max(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.min_val is None or self.max_val is None:
            raise SpecError(
                "Trying to get stats but no any update_stats invoked,"
                "maybe you are quantifying a moe expert, but this expert has never been activated.",
                action="Please check your model and quant config.")
        return self.min_val, self.max_val
