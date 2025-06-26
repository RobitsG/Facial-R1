from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor  
from qwen_vl_utils import process_vision_info  
import torch  
import json  
from tqdm import tqdm  
import re  
import os  
import random  
import argparse

import torch.distributed as dist  
import warnings  
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")  

from open_r1.prompts.emotion_prompt import GRPO_PROMPT, SFT_PROMPT

def setup_distributed():  
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  
    torch.cuda.set_device(local_rank)  
    dist.init_process_group(backend="nccl")  
    return local_rank, dist.get_world_size(), dist.get_rank()  

local_rank, world_size, rank = setup_distributed()  
device = f"cuda:{local_rank}"  
print(f"Process {rank} using {device}")  

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL-7B-Instruct-emotion-sft参数设置")

    parser.add_argument("--main_rank", type=int, default=0, help="主进程rank号")
    parser.add_argument("--test_mode", type=str, default="sft", help="测试模式，例如sft")
    parser.add_argument("--run_name", type=str, default="Qwen2.5-VL-7B-Instruct-emotion-sft", help="运行名称")
    parser.add_argument("--home_path", type=str, default="/root/paddlejob/workspace/wujiulong", help="主目录")
    parser.add_argument("--bsz", type=int, default=4, help="batch size")
    parser.add_argument("--test_datasets", nargs='+', default=["PrivateTest_images_annotations"], help="测试数据集列表")
    
    # 数据相关
    parser.add_argument("--data_root", type=str, default="/root/paddlejob/workspace/wujiulong/emotion_data/FER2013", help="数据根目录")
    parser.add_argument("--image_root", type=str, default="/root/paddlejob/workspace/wujiulong/emotion_data/FER2013", help="图片根目录")
    parser.add_argument("--num_samples", type=int, default=100, help="测试样本数量")

    # 下面两个不用用户手动传，默认由home_path、run_name推断
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")
    parser.add_argument("--output_path", type=str, default=None, help="输出日志路径")

    args = parser.parse_args()

    # 根据home_path和run_name动态设置model_path和output_path
    if args.model_path is None:
        args.model_path = f"{args.home_path}/output/checkpoints/{args.run_name}"
    if args.output_path is None:
        args.output_path = f"{args.home_path}/output/logs/emotion_{args.run_name}.json"

    return args

args = parse_args()

# 加载模型与处理器  
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(  
    args.model_path,  
    torch_dtype=torch.bfloat16,  
    attn_implementation="flash_attention_2",  
    device_map={"": local_rank},  
)  
processor = AutoProcessor.from_pretrained(args.model_path)  

# 从 <answer> 标签里提取文本标签  
def extract_class_answer(content):  
    m = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)  
    return m.group(1).strip() if m else None  
 
for ds in args.test_datasets:  
    if rank == args.main_rank:  
        print(f"\nProcessing {ds}...")  
    ds_path = os.path.join(args.data_root, f"{ds}.jsonl")  
    data = []
    with open(ds_path, "r") as file:
        for line in file:
            if line.strip():
                data.append(json.loads(line))
    random.seed(42)  
    random.shuffle(data)  
    data = data[:args.num_samples]  


    if args.test_mode == "grpo":
        QUESTION_PROMPT = GRPO_PROMPT
    elif args.test_mode == "sft":
        QUESTION_PROMPT = SFT_PROMPT
    else:
        raise ValueError(f"Unknown test mode: {args.test_mode}")

    # 划分给各 rank  
    per_rank = len(data) // world_size  
    start_idx = rank * per_rank  
    end_idx = start_idx + per_rank if rank < world_size - 1 else len(data)  
    rank_data = data[start_idx:end_idx]  

    # 构造消息  
    messages = []  
    for x in rank_data:  
        img = os.path.join(args.image_root, x['image'])  
        if x['question']:
            question = x['question'].relace('<image>', '').strip()
        else:
            question = "What is the emotion of this face?"
        messages.append([  
            {  
                "role": "user",  
                "content": [  
                    {"type": "image", "image": f"file://{img}"},  
                    {"type": "text", "text": QUESTION_PROMPT.format(Question=question)}  
                ]  
            }  
        ])  

    # 批次推理  
    rank_outputs = []  
    for i in tqdm(range(0, len(messages), args.bsz), disable=rank!=args.main_rank):  
        batch = messages[i:i+args.bsz]  
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

        gen = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)  
        # 去掉 prompt 部分  
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen)]  
        texts = processor.batch_decode(  
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False  
        )  
        rank_outputs.extend(texts)  

    print(f"Rank {rank} finished {len(rank_outputs)} items")  

    # 汇总所有 rank 的输出  
    all_outputs = [None] * len(data)  
    rank_res = [(start_idx + i, out) for i, out in enumerate(rank_outputs)]  
    gathered = [None] * world_size  
    dist.all_gather_object(gathered, rank_res)  

    if rank == args.main_rank:  
        for part in gathered:  
            for idx, out in part:  
                all_outputs[idx] = out  

        # 计算分类准确率  
        correct_cnt = 0  
        final_results = []  
        for inp, out in zip(data, all_outputs):  
            gt = inp['labels'][0]     # 例如 "happy"、"sad" 等  
            if args.test_mode == "grpo":
                pred = extract_class_answer(out)  
            elif args.test_mode == "sft":
                pred = out
            else:
                raise ValueError(f"Unknown test mode: {args.test_mode}")
            correct = int(pred == gt)  
            correct_cnt += correct  
            final_results.append({  
                'image': inp['image'],  
                'question': inp['question'],  
                'ground_truth': gt,  
                'model_output': out,  
                'predicted_label': pred,  
                'correct': correct  
            })  

        acc = correct_cnt / len(data) * 100  
        print(f"Classification accuracy on {ds}: {acc:.2f}%")  

        # 保存  
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)  
        with open(args.output_path, "w") as f:  
            json.dump({'accuracy': acc, 'results': final_results}, f, indent=2)  
        print("Results saved to", args.output_path)  

    dist.barrier()  