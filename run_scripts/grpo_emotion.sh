PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export WANDB_API_KEY="726eb58e29b9cbfaf940dc4f286ec9b28749963d"
export REPO_HOME="${PROJECT_ROOT}"
export EXP_NAME="Qwen2.5-VL-7B-Instruct-emotion-grpo"

echo "REPO_HOME: $REPO_HOME"
TASK_TYPE="emo"
home_dir="/root/paddlejob/workspace/wujiulong"
data_paths="$home_dir/emotion_dataset/RAF-DB-Train.jsonl" 
image_folders="$home_dir/emotion_dataset/RAF-DB-Train"
model_path="$home_dir/Qwen2.5-VL-7B-Instruct"
# model_path="$home_dir/output/checkpoints/Qwen2.5-VL-7B-Instruct-emotion-sft"
output_path="${home_dir}/output/checkpoints/${EXP_NAME}"
is_reward_customized_from_vlm_module=False
echo "data_paths: $data_paths"
echo "image_folders: $image_folders"
cd ${REPO_HOME}/src/open-r1-multimodal

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="${home_dir}/output/logs/${EXP_NAME}.log"
# MAX_STEPS=1200 # TODO: change this to your own max steps


# export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
  src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir $output_path \
    --resume_from_checkpoint True \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 1 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_strategy no \
    --save_steps 1000000000000000 \
    --max_steps 100 \
    --num_generations 8 \
    --max_completion_length 2048 \
    --reward_funcs accuracy format \
    --beta 0.04 \
    --report_to wandb \
    --dataset-name this_is_not_used \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero3.json \

echo "Training completed for ${EXP_NAME}"
