export WANDB_API_KEY="726eb58e29b9cbfaf940dc4f286ec9b28749963d"
export CUDA_VISIBLE_DEVICES=0,1,2,3

TEST_MODE="grpo"
TEST_DATASET="FABA-Test"
TEST_EXAMPLES=2000
HOME_PATH="/root/paddlejob/workspace/wujiulong"
MODEL_NAME="Qwen2.5-VL-7B-Instruct-emotion-${TEST_MODE}-faba"
cd $HOME_PATH/Facial-R1/src/eval
export PYTHONPATH=$PYTHONPATH:$HOME_PATH/Facial-R1/src/open-r1-multimodal/src

torchrun --nproc_per_node=4 --master_port="12345" test_emotion_r1.py \
  --main_rank 0 \
  --test_mode "${TEST_MODE}" \
  --run_name "${MODEL_NAME}" \
  --home_path "${HOME_PATH}" \
  --bsz 4 \
  --test_datasets "${TEST_DATASET}" \
  --data_root "${HOME_PATH}/emotion_dataset" \
  --image_root "${HOME_PATH}/emotion_dataset/${TEST_DATASET}" \
  --num_samples "${TEST_EXAMPLES}" \
  --model_path "${HOME_PATH}/output/checkpoints/${MODEL_NAME}" \
  --output_path "${HOME_PATH}/output/logs/${MODEL_NAME}_${TEST_DATASET}.json"