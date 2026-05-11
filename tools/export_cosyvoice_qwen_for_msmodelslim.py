# Copyright (c) 2025
# Export Qwen2ForCausalLM weights from CosyVoice2 llm.pt into a HuggingFace-style
# directory for Ascend msmodelslim quant_qwen.py (W8A8 / INT8 weight+activation).
#
# Typical usage on the Ascend machine (after copying this repo):
#   python3 tools/export_cosyvoice_qwen_for_msmodelslim.py \\
#       --model_dir /path/to/CosyVoice2-0.5B \\
#       --out_dir /path/to/cosyvoice_qwen2_fp32_export
#
# Then (see msit msmodelslim/example/Qwen/README.md):
#   python3 quant_qwen.py --model_path /path/to/cosyvoice_qwen2_fp32_export \\
#       --save_directory /path/to/cosyvoice_qwen2_w8a8 \\
#       --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu
#
# Notes:
# - Replace calib JSONL with CosyVoice-like prompts / dumped activations if BoolQ hurts TTS quality.
# - After quantization, load quantized weights per msmodelslim docs; avoid calling .half() on quantized Linear.

import argparse
import os
import shutil
import sys

import torch


def _pick_state_dict(obj):
    if isinstance(obj, dict) and 'state_dict' in obj:
        return obj['state_dict']
    if isinstance(obj, dict) and 'model' in obj and isinstance(obj['model'], dict):
        return obj['model']
    return obj


def _strip_known_prefixes(key: str, prefixes):
    for p in prefixes:
        if key.startswith(p):
            return key[len(p) :]
    return None


def extract_qwen_causal_lm_keys(state_dict: dict):
    """
    CosyVoice2: Qwen2LM.llm is Qwen2Encoder; its .model is Qwen2ForCausalLM.
    Checkpoint keys are usually 'llm.model.<Qwen2ForCausalLM state key>'.
    """
    candidate_prefixes = (
        'llm.model.',
        'module.llm.model.',
    )
    out = {}
    unmatched = []
    for k, v in state_dict.items():
        if '.text_encoder' in k or k.startswith('text_encoder'):
            continue
        if k.startswith('llm_embedding') or k.startswith('speech_embedding'):
            continue
        if k.startswith('llm_decoder') or k.startswith('spk_embed'):
            continue
        if k.startswith('text_embedding') or k.startswith('text_encoder'):
            continue
        stripped = _strip_known_prefixes(k, candidate_prefixes)
        if stripped is None:
            if k.startswith('llm.') and not k.startswith('llm.model.'):
                unmatched.append(k)
            continue
        # Keep only tensors that belong to Qwen2ForCausalLM (model.* / lm_head.*)
        if stripped.startswith('model.') or stripped.startswith('lm_head.'):
            out[stripped] = v
        else:
            unmatched.append(k)
    return out, unmatched


def copy_blank_hf_assets(blank_dir: str, out_dir: str):
    if not os.path.isdir(blank_dir):
        raise FileNotFoundError('CosyVoice blank Qwen dir not found: {}'.format(blank_dir))
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(blank_dir):
        src = os.path.join(blank_dir, name)
        if os.path.isfile(src) and not name.startswith('.'):
            shutil.copy2(src, os.path.join(out_dir, name))


def main():
    parser = argparse.ArgumentParser(description='Export Qwen2 backbone from CosyVoice2 llm.pt for msmodelslim')
    parser.add_argument('--model_dir', required=True, help='CosyVoice2 model directory (contains llm.pt and CosyVoice-BlankEN/)')
    parser.add_argument('--out_dir', required=True, help='Output HF-style directory')
    parser.add_argument(
        '--blank_subdir',
        default='CosyVoice-BlankEN',
        help='Subdirectory under model_dir with Qwen2 config/tokenizer (default: CosyVoice-BlankEN)',
    )
    parser.add_argument('--llm_pt', default='llm.pt', help='LLM checkpoint filename inside model_dir')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Try loading with transformers.Qwen2ForCausalLM.from_pretrained (requires transformers)',
    )
    args = parser.parse_args()

    llm_path = os.path.join(args.model_dir, args.llm_pt)
    blank_dir = os.path.join(args.model_dir, args.blank_subdir)
    if not os.path.isfile(llm_path):
        print('Missing {}'.format(llm_path), file=sys.stderr)
        sys.exit(1)

    raw = torch.load(llm_path, map_location='cpu')
    state = _pick_state_dict(raw)
    sub, unmatched = extract_qwen_causal_lm_keys(state)
    if not sub:
        print('No Qwen2ForCausalLM tensors found; check key prefixes in checkpoint.', file=sys.stderr)
        sys.exit(1)

    copy_blank_hf_assets(blank_dir, args.out_dir)
    torch.save(sub, os.path.join(args.out_dir, 'pytorch_model.bin'))

    print('Wrote {} tensors to {}'.format(len(sub), args.out_dir))
    if unmatched:
        print('Ignored / unmatched sample keys (first 10):')
        for k in unmatched[:10]:
            print('  ', k)

    if args.validate:
        from transformers import Qwen2ForCausalLM

        m = Qwen2ForCausalLM.from_pretrained(args.out_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        print('validate: loaded Qwen2ForCausalLM ok, num_parameters={}'.format(sum(p.numel() for p in m.parameters())))


if __name__ == '__main__':
    main()
