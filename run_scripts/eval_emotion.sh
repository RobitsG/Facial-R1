export WANDB_API_KEY="726eb58e29b9cbfaf940dc4f286ec9b28749963d"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

HOME_PATH="/root/paddlejob/workspace/wujiulong"
TEST_MODE="grpo"
TEST_DATASET="FABA-Test"
IMAGE_ROOT="${HOME_PATH}/emotion_dataset/FABA-Test"
MODEL_NAME="Qwen2.5-VL-7B-Instruct-emotion-sft"
CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
OUTPUT_PATH="${HOME_PATH}/output/logs/${MODEL_NAME}_${TEST_DATASET}_${CURRENT_TIME}.json"
CONFIG_PATH="${HOME_PATH}/Facial-R1/src/open-r1-multimodal/src/open_r1/configs/config.json"
cd $HOME_PATH/Facial-R1/src/eval
export PYTHONPATH=$PYTHONPATH:$HOME_PATH/Facial-R1/src/open-r1-multimodal/src

torchrun --nproc_per_node=4 --master_port="12345" eval_emotion.py \
  --main_rank 0 \
  --test_mode "${TEST_MODE}" \
  --run_name "${MODEL_NAME}" \
  --home_path "${HOME_PATH}" \
  --bsz 8 \
  --test_datasets "${TEST_DATASET}" \
  --data_root "${HOME_PATH}/emotion_dataset" \
  --image_root "${IMAGE_ROOT}" \
  --config_path "${CONFIG_PATH}" \
  --model_path "${HOME_PATH}/output/checkpoints/${MODEL_NAME}" \
  --output_path "${OUTPUT_PATH}"