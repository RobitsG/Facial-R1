#!/bin/bash
set -e

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export WANDB_API_KEY="726eb58e29b9cbfaf940dc4f286ec9b28749963d"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"
home_dir="/root/paddlejob/workspace/wujiulong"
data_paths="$home_dir/emotion_dataset/train.yaml" # 支持yaml格式
model_path="$home_dir/output/checkpoints/Qwen2.5-VL-7B-Instruct-emotion-grpo-best2"
echo "data_paths: $data_paths"

export EXP_NAME="Qwen2.5-VL-7B-Instruct-emotion-sft"
# ckpt_path="$home_dir/output/checkpoints/${EXP_NAME}/checkpoint-1968"
cd ${REPO_HOME}/src/open-r1-multimodal/src

export DEBUG_MODE="true"
export LOG_PATH="${home_dir}/output/logs/${EXP_NAME}.log"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch \
    --config_file=${REPO_HOME}/src/open-r1-multimodal/configs/zero3.yaml \
    open_r1/sft.py \
    --model_name_or_path $model_path \
    --dataset_name $data_paths \
    --learning_rate 5e-6 \
    --num_train_epochs 1 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --output_dir ${home_dir}/output/checkpoints/${EXP_NAME} \
    --attn_implementation flash_attention_2 \
    --seed 42 \
    --save_strategy epoch \
    --num_train_epochs 2 \
    --eval_strategy no \
    --eval_steps 1000 \
    --remove_unused_columns False \
    --packing

echo "SFT Training completed for ${EXP_NAME}"