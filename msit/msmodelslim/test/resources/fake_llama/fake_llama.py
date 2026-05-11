# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os

from transformers import AutoTokenizer
from transformers.models.llama import LlamaConfig, LlamaForCausalLM


def get_fake_llama_model_and_tokenizer():
    """
    获取一个随机的、非常小的Llama模型以及其所对应的tokenizer，用于验证工具中某些数值算法的正确性
    """
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = LlamaConfig.from_json_file(config_path)
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(__file__))
    return LlamaForCausalLM(config), tokenizer