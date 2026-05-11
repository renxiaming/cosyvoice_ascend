#!/usr/bin/env bash
set -euo pipefail

# 正确的根目录（必须指向 msmodelslim 这一层）
export MSMODELSLIM_ROOT=/home/ma-user/work/test/model/cosyvoice_w8a8/msit/msmodelslim
export PYTHONPATH="${MSMODELSLIM_ROOT}:${PYTHONPATH}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False 

cd "${MSMODELSLIM_ROOT}"

python3 example/Qwen/quant_qwen.py \
  --model_path /home/ma-user/work/test/model/weight/CosyVoice2-0.5B-wa8a \
  --save_directory /home/ma-user/work/test/model/weight/CosyVoice2-0.5B-w8a8_quant \
  --calib_file example/common/boolq.jsonl \
  --w_bit 8 --a_bit 8 \
  --device_type npu \
  --model_type qwen2 \
  --trust_remote_code True
