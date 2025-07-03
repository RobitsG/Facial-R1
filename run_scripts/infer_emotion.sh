export WANDB_API_KEY="726eb58e29b9cbfaf940dc4f286ec9b28749963d"
export CUDA_VISIBLE_DEVICES=0,1,2,3

TEST_MODE="grpo"
TEST_DATASET="RAF-DB-Test"
TEST_EXAMPLES=2000
HOME_PATH="/root/paddlejob/workspace/wujiulong"
RUN_NAME="Qwen2.5-VL-7B-Instruct-emotion-${TEST_MODE}"
cd $HOME_PATH/VLM-R1/src/eval

torchrun --nproc_per_node=4 --master_port="12345" test_emotion_r1.py \
  --main_rank 0 \
  --test_mode "${TEST_MODE}" \
  --run_name "${RUN_NAME}" \
  --home_path "${HOME_PATH}" \
  --bsz 4 \
  --test_datasets "${TEST_DATASET}" \
  --data_root "${HOME_PATH}/emotion_dataset" \
  --image_root "${HOME_PATH}/emotion_dataset/${TEST_DATASET}" \
  --num_samples "${TEST_EXAMPLES}" \
  --model_path "${HOME_PATH}/output/checkpoints/${RUN_NAME}" \
  --output_path "${HOME_PATH}/output/logs/${RUN_NAME}.json"