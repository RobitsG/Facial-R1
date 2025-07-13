from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import os
import argparse
import torch.distributed as dist
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

from open_r1.prompts.emotion_prompt import INFER_PROMPT, GRPO_PROMPT

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, dist.get_world_size(), dist.get_rank()

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL inference: only replace 'description' field")
    parser.add_argument("--main_rank", type=int, default=0)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--prompt_mode", type=str, default="sft", choices=["sft", "grpo"])
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args()

def main():
    set_seed(42)
    try:
        local_rank, world_size, rank = setup_distributed()
    except:
        # 单卡单进程环境
        local_rank, world_size, rank = 0, 1, 0
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    args = parse_args()

    with open(args.config_json, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # inference数据集的emotions确定（需满足接口通用性，可根据你的评估数据key自行微调）
    input_basename = os.path.basename(args.input_file)
    emotions = config.get(input_basename, {}).get("emotions", [])

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": local_rank} if torch.cuda.is_available() else None,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    # 读取输入jsonl
    data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    per_rank = len(data) // world_size
    start_idx = rank * per_rank
    end_idx = start_idx + per_rank if rank < world_size - 1 else len(data)
    rank_data = data[start_idx:end_idx]

    messages = []
    for item in rank_data:
        img_path = os.path.join(args.image_root, item['image']) if 'image' in item else None
        # prompt中的question和emotions和你的评估脚本保持一致，如果有question字段用，否则默认模板
        question = item['question'].replace('<image>','').strip() if 'question' in item and item['question'] else "What is the emotion of this face?"
        if img_path:
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{img_path}"},
                        {"type": "text", "text": INFER_PROMPT.format(
                            Question=question, 
                            Emotions=emotions.keys(),
                            true_aus=item['AUs'],
                            true_emotion=item['labels'],
                        )}
                    ]
                }
            ])
        else:
            messages.append(None)

    # 推理
    rank_outputs = []
    for i in tqdm(range(0, len(messages), args.bsz), desc=f"Rank {rank} inference", disable=rank!=0):
        batch = [m for m in messages[i:i + args.bsz] if m is not None]
        if not batch:
            rank_outputs.extend([None] * min(args.bsz, len(messages[i:i + args.bsz])))
            continue
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in batch]
        image_inputs, video_inputs = process_vision_info(batch)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        ).to(device)
        gen = model.generate(**inputs, use_cache=True, max_new_tokens=args.max_new_tokens, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen)]
        texts = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # 保证和输入数据顺序对应
        batch_idx = 0
        for j in range(min(args.bsz, len(messages[i:i + args.bsz]))):
            if messages[i + j] is not None:
                rank_outputs.append(texts[batch_idx])
                batch_idx += 1
            else:
                rank_outputs.append(None)

    all_outputs = [None] * len(data)
    rank_res = [(start_idx + i, out) for i, out in enumerate(rank_outputs)]
    try:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, rank_res)
    except:
        gathered = [rank_res]
    if rank == args.main_rank:
        for part in gathered:
            for idx, out in part:
                all_outputs[idx] = out

        # 写入输出jsonl
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            for i, item in enumerate(data):
                output_item = item.copy()
                output_item["description"] = all_outputs[i]
                f.write(json.dumps(output_item, ensure_ascii=False) + "\n")
        print(f"Results saved to {args.output_file}")

    try:
        dist.barrier()
    except:
        pass

if __name__ == "__main__":
    main()