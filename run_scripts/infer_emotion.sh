#!/bin/bash

export WANDB_API_KEY="726eb58e29b9cbfaf940dc4f286ec9b28749963d"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 配置参数
PROMPT_MODE="grpo"  # 可以是 "sft" 或 "grpo"
DATASET="FER2013-Train-3000"
HOME_PATH="/root/paddlejob/workspace/wujiulong"
IMAGE_ROOT="${HOME_PATH}/emotion_dataset/FER2013-Train"
MODEL_NAME="Qwen2.5-VL-7B-Instruct-emotion-${PROMPT_MODE}-best"

# 输入输出文件路径
INPUT_FILE="${HOME_PATH}/emotion_dataset/${DATASET}.jsonl"
OUTPUT_FILE="${HOME_PATH}/emotion_dataset/${DATASET}-infer1.jsonl"
CONFIG_JSON="${HOME_PATH}/Facial-R1/src/open-r1-multimodal/src/open_r1/configs/config.json"

# 设置工作目录和Python路径
cd $HOME_PATH/Facial-R1/src/eval
export PYTHONPATH=$PYTHONPATH:$HOME_PATH/Facial-R1/src/open-r1-multimodal/src

# 创建输出目录（如果不存在）
mkdir -p $(dirname "${OUTPUT_FILE}")

# 使用torchrun执行分布式推理
torchrun --nproc_per_node=4 --master_port="12345" infer_emotion.py \
  --main_rank 0 \
  --model_path "${HOME_PATH}/output/checkpoints/${MODEL_NAME}" \
  --input_file "${INPUT_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --image_root "${IMAGE_ROOT}" \
  --config_json "${CONFIG_JSON}" \
  --prompt_mode "${PROMPT_MODE}" \
  --bsz 8 \
  --max_new_tokens 1024

echo "Inference completed. Results saved to: ${OUTPUT_FILE}"