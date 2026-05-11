# Copyright (c) 2025
# Load msmodelslim ascendV1 / safetensors LLM-PTQ artifacts onto CosyVoice2 Qwen2 backbone.
import glob
import importlib.util
import json
import os


def _pick_quant_description_json(quant_dir):
    preferred = os.path.join(quant_dir, 'quant_model_description.json')
    if os.path.isfile(preferred):
        return preferred
    paths = sorted(glob.glob(os.path.join(quant_dir, 'quant_model_description*.json')))
    if not paths:
        raise FileNotFoundError('No quant_model_description*.json under {}'.format(quant_dir))
    return paths[0]


def _load_merged_safetensors(quant_dir):
    from safetensors.torch import load_file

    paths = sorted(glob.glob(os.path.join(quant_dir, '*.safetensors')))
    if not paths:
        raise FileNotFoundError('No *.safetensors under {}'.format(quant_dir))
    merged = {}
    for p in paths:
        merged.update(load_file(p))
    return merged


def _map_device_type_for_fake_quant(device):
    """FakeQuantizeCalibrator expects dev_type in cpu | npu | gpu."""
    t = device.type
    if t == 'cuda':
        return 'gpu', device.index if device.index is not None else 0
    if t == 'cpu':
        return 'cpu', None
    if t == 'npu':
        return 'npu', device.index if device.index is not None else 0
    raise ValueError('Unsupported device for msmodelslim fake-quant LLM: {}'.format(device))


def attach_msmodelslim_fake_quant_to_qwen2lm(cosyvoice2_torch_model, quant_dir):
    """
    Swap Qwen2ForCausalLM Linears to msmodelslim fake-quant modules (W8A8 / W8A8_DYNAMIC / W8A16 / W4A16).

    Must run after CosyVoice2Model.load() so shapes and llm.pt weights match calibration graph.
    quant_dir: directory containing quant_model_description.json (or *_w8a8.json) and *.safetensors
    """
    quant_dir = os.path.abspath(quant_dir)
    if not os.path.isdir(quant_dir):
        raise FileNotFoundError(quant_dir)

    from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import FakeQuantizeCalibrator

    llm = cosyvoice2_torch_model.llm
    backbone = llm.llm.model
    if next(backbone.parameters()).device.type == 'meta':
        raise RuntimeError('Backbone is on meta device; load llm.pt before attach_msmodelslim_fake_quant_to_qwen2lm')

    if (
        importlib.util.find_spec('torch_npu') is not None
        and next(backbone.parameters()).device.type == 'cpu'
    ):
        import torch

        if torch.npu.is_available():
            backbone.to('npu:0')

    desc_path = _pick_quant_description_json(quant_dir)
    with open(desc_path, 'r', encoding='utf-8') as f:
        description = json.load(f)
    safetensor = _load_merged_safetensors(quant_dir)

    param_dev = next(backbone.parameters()).device
    dev_type, dev_id = _map_device_type_for_fake_quant(param_dev)
    FakeQuantizeCalibrator(backbone, dev_id, dev_type, description, safetensor)
    backbone.eval()


def ensure_llm_fp32_for_quant(cosyvoice2_torch_model):
    """Cast Qwen2LM floating modules to float32 before fake-quant (idempotent for already-fp32)."""
    cosyvoice2_torch_model.llm.float()
