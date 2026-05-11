#!/usr/bin/env bash
# CosyVoice2 inference with msmodelslim W8A8 fake-quant Qwen2 backbone (replace after llm.pt load).
# Prereq: same HF export as quant_qwen.py --model_path, plus quant output dir (json + safetensors).
#
# Example (use real absolute paths on your machine):
#   export MODEL_PATH=/home/you/weight/CosyVoice2-0.5B
#   export QWEN_FP_DIR=/home/you/weight/CosyVoice2-0.5B-wa8a   # export_cosyvoice_qwen_for_msmodelslim.py
#   export QWEN_QUANT_DIR=/home/you/weight/CosyVoice2-0.5B-w8a8_quant
#   bash run_infer_w8a8.sh --stream
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# run_infer_py.sh 会重写简短 PYTHONPATH 并设 COSYVOICE_SKIP_MSLIM_PYTHONPATH_PREPEND=1，
# 避免继承巨型 PYTHONPATH 导致 execve「Argument list too long」。
if [[ -z "${COSYVOICE_SKIP_MSLIM_PYTHONPATH_PREPEND:-}" ]]; then
  export PYTHONPATH="${ROOT}/msit/msmodelslim:${PYTHONPATH:-}"
fi
export COSYVOICE_SKIP_LLM_HALF="${COSYVOICE_SKIP_LLM_HALF:-1}"

: "${MODEL_PATH:?set MODEL_PATH to CosyVoice2 asset dir (with llm.pt)}"
: "${QWEN_FP_DIR:?set QWEN_FP_DIR to HF export used for quant_qwen}"
: "${QWEN_QUANT_DIR:?set QWEN_QUANT_DIR to W8A8 quant output dir}"

# Delegate to run.sh env if you already use it: source run.sh pieces or set ASCEND paths in your shell.
exec python3 "${ROOT}/infer.py" \
  --model_path "${MODEL_PATH}" \
  --qwen_fp_model_dir "${QWEN_FP_DIR}" \
  --qwen_quant_dir "${QWEN_QUANT_DIR}" \
  "$@"
