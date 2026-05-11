# 指定使用NPU ID，默认为0
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH=third_party/Matcha-TTS:$PYTHONPATH
export PYTHONPATH=transformers/src:$PYTHONPATH
# 终端运行这 2 行，再启动你的代码
# export ASCEND_GEO_W8A16=1
# export DYNAMIC_QUANT=1

# 使能环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 规避找不到ttsfrd
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# 规避找不到cstdint
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0:${CPLUS_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu:${CPLUS_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1/toolkit/toolchain/hcc/aarch64-target-linux-gnu/sys-include:${CPLUS_INCLUDE_PATH}

# 清理modelscope缓存
rm -rf ~/.cache/modelscope/

# 与 weight 目录对齐：浮点包 + msmodelslim 量化目录
# 终端可覆盖，例如：export MODEL_ROOT=/home/ma-user/work/test/model/weight
MODEL_ROOT="${MODEL_ROOT:-../weight}"

# 设 USE_LLM_QUANT=0 可对比浮点 Qwen（不挂 --llm_quant_dir）
USE_LLM_QUANT="${USE_LLM_QUANT:-1}"
QUANT_ARGS=()
if [ "${USE_LLM_QUANT}" != "0" ]; then
  QUANT_ARGS+=(--llm_quant_dir="${MODEL_ROOT}/CosyVoice2-0.5B-w8a8_quant")
fi

python3 infer.py \
  --model_path="${MODEL_ROOT}/CosyVoice2-0.5B" \
  "${QUANT_ARGS[@]}" \
  --stream
