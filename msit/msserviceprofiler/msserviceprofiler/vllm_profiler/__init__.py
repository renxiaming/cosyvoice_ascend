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
from .utils import logger, set_log_level
from .module_hook import apply_hooks

set_log_level("info")  # Default is info, put here for user changes
VLLM_USE_V1 = os.environ.get('VLLM_USE_V1', '0')
if VLLM_USE_V1 == "0":
    from .vllm_v0 import batch_hookers, kvcache_hookers, model_hookers, request_hookers
    apply_hooks()  # 应用所有hookers
elif VLLM_USE_V1 == "1":
    from .vllm_v1 import batch_hookers, kvcache_hookers, model_hookers, request_hookers
    apply_hooks()  # 应用所有hookers
else:
    logger.error(f"unknown vLLM interface version: VLLM_USE_V1={VLLM_USE_V1}")
