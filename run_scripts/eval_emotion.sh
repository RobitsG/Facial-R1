#!/bin/bash

export WANDB_API_KEY="726eb58e29b9cbfaf940dc4f286ec9b28749963d"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

HOME_PATH="/root/paddlejob/workspace/wujiulong"
TEST_MODE="grpo" # grpo or sft
TEST_DATASET="AffectNet-Test"
IMAGE_ROOT="${HOME_PATH}/emotion_dataset/AffectNet-Test"
# MODEL_NAME="Qwen2.5-VL-7B-Instruct"
# MODEL_NAME="InternVL3-8B"
# MODEL_NAME="llava-v1.6-vicuna-7b-hf"
# MODEL_NAME="gpt4o"
# MODEL_PATH="${HOME_PATH}/${MODEL_NAME}"
MODEL_NAME="Qwen2.5-VL-7B-Instruct-emotion-grpo"
MODEL_PATH="${HOME_PATH}/output/checkpoints/${MODEL_NAME}"
CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
OUTPUT_PATH="${HOME_PATH}/output/logs/${MODEL_NAME}_${TEST_DATASET}_${CURRENT_TIME}.json"
CONFIG_PATH="${HOME_PATH}/Facial-R1/src/open-r1-multimodal/src/open_r1/configs/config.json"
cd $HOME_PATH/Facial-R1/src/eval
export PYTHONPATH=$PYTHONPATH:$HOME_PATH/Facial-R1/src/open-r1-multimodal/src

# 根据MODEL_NAME选择不同的推理脚本
if [[ "$MODEL_NAME" == *"qwen"* || "$MODEL_NAME" == *"Qwen"* ]]; then
    EVAL_SCRIPT="eval_qwen.py"
    echo "使用Qwen推理脚本: ${EVAL_SCRIPT}"
elif [[ "$MODEL_NAME" == *"internvl"* || "$MODEL_NAME" == *"InternVL"* ]]; then
    EVAL_SCRIPT="eval_internvl.py"
    echo "使用InternVL推理脚本: ${EVAL_SCRIPT}"
elif [[ "$MODEL_NAME" == *"llava"* || "$MODEL_NAME" == *"LLaVA"* ]]; then
    EVAL_SCRIPT="eval_llava.py"
    echo "使用LLaVA推理脚本: ${EVAL_SCRIPT}"
elif [[ "$MODEL_NAME" == *"gpt"* || "$MODEL_NAME" == *"GPT"* ]]; then
    EVAL_SCRIPT="eval_gpt.py"
    echo "使用GPT推理脚本: ${EVAL_SCRIPT}"
else
    echo "警告: 无法根据模型名称'${MODEL_NAME}'确定推理脚本类型，默认使用eval_qwen.py"
    EVAL_SCRIPT="eval_qwen.py"
fi

torchrun --nproc_per_node=8 --master_port="12345" ${EVAL_SCRIPT} \
  --main_rank 0 \
  --test_mode "${TEST_MODE}" \
  --run_name "${MODEL_NAME}" \
  --home_path "${HOME_PATH}" \
  --bsz 8 \
  --test_datasets "${TEST_DATASET}" \
  --data_root "${HOME_PATH}/emotion_dataset" \
  --image_root "${IMAGE_ROOT}" \
  --config_path "${CONFIG_PATH}" \
  --model_path "${MODEL_PATH}" \
  --output_path "${OUTPUT_PATH}"
