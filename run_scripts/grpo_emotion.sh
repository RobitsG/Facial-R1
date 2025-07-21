PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export WANDB_API_KEY="726eb58e29b9cbfaf940dc4f286ec9b28749963d"
export REPO_HOME="${PROJECT_ROOT}"
export EXP_NAME="Qwen2.5-VL-7B-Instruct-emotion-grpo"

echo "REPO_HOME: $REPO_HOME"
TASK_TYPE="emo"
home_dir="/root/paddlejob/workspace/wujiulong"
data_path1="$home_dir/emotion_dataset/FABA-Train-gpt-2000.jsonl"
data_path2="$home_dir/emotion_dataset/DISFA-Train-3.jsonl"
data_path3="$home_dir/emotion_dataset/RAF-AU-Train.jsonl"
data_path4="$home_dir/emotion_dataset/FER2013-Train-1000.jsonl"
data_path5="$home_dir/emotion_dataset/AffectNet-Train-1000.jsonl"
data_path6="$home_dir/emotion_dataset/RAF-DB-Train-1000.jsonl"
data_paths="${data_path1}:${data_path2}:${data_path3}:${data_path4}:${data_path5}:${data_path6}"
image_folder1="$home_dir/emotion_dataset/FABA-Train"
image_folder2="$home_dir/emotion_dataset/DISFA-Train"
image_folder3="$home_dir/emotion_dataset/RAF-AU-Train"
image_folder4="$home_dir/emotion_dataset/FER2013-Train"
image_folder5="$home_dir/emotion_dataset/AffectNet-Train"
image_folder6="$home_dir/emotion_dataset/RAF-DB-Train"
image_folders="${image_folder1}:${image_folder2}:${image_folder3}:${image_folder4}:${image_folder5}:${image_folder6}"
# model_path="$home_dir/Qwen2.5-VL-7B-Instruct"
model_path="$home_dir/output/checkpoints/Qwen2.5-VL-7B-Instruct-emotion-sft"
output_path="${home_dir}/output/checkpoints/${EXP_NAME}"
is_reward_customized_from_vlm_module=False
echo "data_paths: $data_paths"
echo "image_folders: $image_folders"
cd ${REPO_HOME}/src/open-r1-multimodal/src

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="${home_dir}/output/logs/${EXP_NAME}.log"
# MAX_STEPS=1200 # TODO: change this to your own max steps


# export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
  open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir $output_path \
    --resume_from_checkpoint False \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_strategy epoch \
    --num_train_epochs 2 \
    --max_completion_length 2048 \
    --reward_funcs accuracy format au \
    --beta 0.04 \
    --report_to wandb \
    --dataset-name this_is_not_used \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero3.json \

echo "Training completed for ${EXP_NAME}"
