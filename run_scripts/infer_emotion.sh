#!/bin/bash

export WANDB_API_KEY="726eb58e29b9cbfaf940dc4f286ec9b28749963d"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 配置参数
PROMPT_MODE="grpo"  # 可以是 "sft" 或 "grpo"
DATASET="FABA-Test"
NUM_EXAMPLES=20000000
HOME_PATH="/root/paddlejob/workspace/wujiulong"
MODEL_NAME="Qwen2.5-VL-7B-Instruct-emotion-${PROMPT_MODE}"

# 输入输出文件路径
INPUT_FILE="${HOME_PATH}/emotion_dataset/${DATASET}.jsonl"
OUTPUT_FILE="${HOME_PATH}/output/answers/${DATASET}.jsonl"
CONFIG_JSON="${HOME_PATH}/Facial-R1/src/open-r1-multimodal/src/open_r1/configs/config.json"

# 设置工作目录和Python路径
cd $HOME_PATH/Facial-R1/src/eval
export PYTHONPATH=$PYTHONPATH:$HOME_PATH/Facial-R1/src/open-r1-multimodal/src

# 创建输出目录（如果不存在）
mkdir -p $(dirname "${OUTPUT_FILE}")

# 使用torchrun执行分布式推理
torchrun --nproc_per_node=4 --master_port="12346" infer_emotion.py \
  --main_rank 0 \
  --model_path "${HOME_PATH}/output/checkpoints/${MODEL_NAME}" \
  --input_file "${INPUT_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --image_root "${HOME_PATH}/emotion_dataset/${DATASET}" \
  --config_json "${CONFIG_JSON}" \
  --prompt_mode "${PROMPT_MODE}" \
  --bsz 4 \
  --max_new_tokens 256

echo "Inference completed. Results saved to: ${OUTPUT_FILE}"