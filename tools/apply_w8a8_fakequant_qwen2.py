# Copyright (c) 2025
# Build a W8A8 *fake-quant* Qwen2ForCausalLM from msmodelslim ascendV1 outputs (safetensors + json).
# Ref: msmodelslim/docs/.../FakeQuantizeCalibrator.md  and test/cases/pytorch/msmodelslim/test_fake_quant.py
#
# The fp_model_dir must be the SAME HuggingFace-style directory you passed to quant_qwen.py --model_path
# (e.g. CosyVoice export: tools/export_cosyvoice_qwen_for_msmodelslim.py output).
#
# CosyVoice2 integration: after CosyVoice2(...) loads llm.pt, replace backbone:
#   from tools.apply_w8a8_fakequant_qwen2 import build_fake_quant_qwen2
#   cosyvoice.model.llm.llm.model = build_fake_quant_qwen2(fp_dir, quant_dir, dev_type="npu", dev_id=0)
# Then use COSYVOICE_SKIP_LLM_HALF=1 for infer.py.

import argparse
import json
import os
from typing import Optional

import torch
from safetensors.torch import load_file
from transformers import Qwen2ForCausalLM

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import FakeQuantizeCalibrator

# QuantModelJsonDescription.check_description() requires the *first* key to be "model_quant_type".
_MQT = 'model_quant_type'


def _normalize_quant_description_for_fakequant(description: dict) -> dict:
    if _MQT not in description:
        raise ValueError(
            'quant_model_description.json missing "{}". First keys: {}'.format(
                _MQT, list(description.keys())[:20]
            )
        )
    ordered = {_MQT: description[_MQT]}
    for k in sorted(description.keys()):
        if k == _MQT:
            continue
        ordered[k] = description[k]
    return ordered


def _default_weight_name(quant_dir: str) -> str:
    p = os.path.join(quant_dir, 'quant_model_weight_w8a8.safetensors')
    if os.path.isfile(p):
        return p
    for fn in os.listdir(quant_dir):
        if fn.startswith('quant_model_weight_') and fn.endswith('.safetensors'):
            return os.path.join(quant_dir, fn)
    raise FileNotFoundError('No quant_model_weight_*.safetensors under {}'.format(quant_dir))


def build_fake_quant_qwen2(
    fp_model_dir: str,
    quant_dir: str,
    *,
    dev_type: str = 'npu',
    dev_id: int = 0,
    torch_dtype: str = 'bfloat16',
    weight_file: Optional[str] = None,
    desc_file: Optional[str] = None,
) -> Qwen2ForCausalLM:
    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    weight_file = weight_file or _default_weight_name(quant_dir)
    desc_file = desc_file or os.path.join(quant_dir, 'quant_model_description.json')
    if not os.path.isfile(desc_file):
        raise FileNotFoundError(desc_file)

    model = Qwen2ForCausalLM.from_pretrained(
        fp_model_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    safetensor_dic = load_file(weight_file)
    with open(desc_file, 'r', encoding='utf-8') as f:
        description_dic = _normalize_quant_description_for_fakequant(json.load(f))

    calibrator = FakeQuantizeCalibrator(
        model,
        dev_id if dev_type == 'npu' else None,
        dev_type,
        description_dic,
        safetensor_dic,
    )
    return calibrator.model


def main():
    parser = argparse.ArgumentParser(description='Build fake-quant Qwen2ForCausalLM from W8A8 ascendV1 outputs')
    parser.add_argument('--fp_model_dir', required=True, help='Same as quant_qwen --model_path (HF export)')
    parser.add_argument('--quant_dir', required=True, help='Directory with quant_model_description.json + safetensors')
    parser.add_argument('--dev_type', default='cpu', choices=['cpu', 'npu'])
    parser.add_argument('--dev_id', type=int, default=0)
    parser.add_argument('--torch_dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    args = parser.parse_args()

    m = build_fake_quant_qwen2(
        args.fp_model_dir,
        args.quant_dir,
        dev_type=args.dev_type,
        dev_id=args.dev_id,
        torch_dtype=args.torch_dtype,
    )
    p = next(m.parameters(), None)
    print('ok: fake-quant Qwen2; first param device={} dtype={}'.format(
        p.device if p is not None else 'n/a',
        p.dtype if p is not None else 'n/a',
    ))


if __name__ == '__main__':
    main()
