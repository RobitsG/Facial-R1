from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor  
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
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
from open_r1.utils.eval_utils import *


def set_seed(seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(42)

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
    parser.add_argument("--data_root", type=str, default="/root/paddlejob/workspace/wujiulong/emotion_data/FER2013", help="数据根目录")
    parser.add_argument("--image_root", type=str, default="/root/paddlejob/workspace/wujiulong/emotion_data/FER2013", help="图片根目录")
    parser.add_argument("--num_samples", type=int, default=-1, help="测试样本数量")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")
    parser.add_argument("--output_path", type=str, default=None, help="输出日志路径")
    parser.add_argument("--config_path", type=str, default=None)
    # ===== GPT评估参数 =====
    parser.add_argument("--use_gpt", action='store_true', default=False, help="是否调用GPT评价推理与真值描述是否一致")
    parser.add_argument("--openai_key", type=str, default=None, help="OpenAI API KEY，如需用GPT功能")
    parser.add_argument("--base_url", type=str, default=None, help="OpenAI API URL，如需用GPT功能")
    args = parser.parse_args()
    if args.model_path is None:
        args.model_path = f"{args.home_path}/output/checkpoints/{args.run_name}"
    if args.output_path is None:
        args.output_path = f"{args.home_path}/output/logs/emotion_{args.run_name}.json"
    return args

args = parse_args()

with open(args.config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# 加载LLaVA模型和处理器
processor = LlavaNextProcessor.from_pretrained(
    args.model_path,
)
model = LlavaNextForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map={"": local_rank},
)

############# 主流程 #############

for ds in args.test_datasets:
    if rank == args.main_rank:
        print(f"\nProcessing {ds}...")
    ds_path = os.path.join(args.data_root, f"{ds}.jsonl")
    emotions = config[os.path.basename(ds_path)]["emotions"]
    data = []
    with open(ds_path, "r") as file:
        for line in file:
            if line.strip():
                data.append(json.loads(line))
    random.shuffle(data)
    if args.num_samples > 0:
        data = data[:args.num_samples]

    if args.test_mode == "grpo":
        QUESTION_PROMPT = GRPO_PROMPT
    elif args.test_mode == "sft":
        QUESTION_PROMPT = SFT_PROMPT
    else:
        raise ValueError(f"Unknown test mode: {args.test_mode}")

    per_rank = len(data) // world_size
    start_idx = rank * per_rank
    end_idx = start_idx + per_rank if rank < world_size - 1 else len(data)
    rank_data = data[start_idx:end_idx]

    # 准备LLaVA格式的消息
    conversations = []
    for x in rank_data:
        img_path = os.path.join(args.image_root, x['image'])
        question = x['question'].replace('<image>', '').strip() if x['question'] else "What is the emotion of this face?"
        
        # 创建LLaVA格式的对话
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": QUESTION_PROMPT.format(Question=question, Emotions=emotions.keys())},
                    {"type": "image"},
                ]
            }
        ]
        conversations.append((conversation, img_path))

    rank_outputs = []
    for i in tqdm(range(0, len(conversations), args.bsz), disable=rank!=args.main_rank):
        batch = conversations[i:i+args.bsz]
        
        # 准备批量输入
        prompts = []
        images = []
        
        for conv, img_path in batch:
            # 应用对话模板
            prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
            prompts.append(prompt)
            
            # 加载图片
            from PIL import Image
            image = Image.open(img_path)
            images.append(image)
        
        # 批量处理输入
        inputs = processor(
            images=images, 
            text=prompts, 
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # 生成输出
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        
        # 解码输出
        for j, output in enumerate(outputs):
            response = processor.decode(output, skip_special_tokens=True)
            # 从完整响应中提取模型生成的部分（去除提示部分）
            # 这里我们假设LLaVA的输出包含了完整对话历史，需要提取出最后的回复部分
            # 由于不同模型的输出格式可能不同，这里可能需要针对具体模型进行调整
            response_only = response.split("ASSISTANT:")[-1].strip()
            rank_outputs.append(response_only)

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

        # 组装推理明细
        inference_results = []
        all_gt_labels, all_pred_labels = [], []
        all_gt_aus, all_pred_aus = [], []
        label_set, au_set = set(), set()

        for inp, out in zip(data, all_outputs):
            gt_labels = inp.get('labels', [])
            pred_labels = extract_label_pred_answer(out)
            all_gt_labels.append(gt_labels)
            all_pred_labels.append(pred_labels)
            label_set.update(gt_labels)
            # label_set.update(pred_labels)
            gt_aus = inp.get('AUs', [])
            pred_aus = extract_au_pred_think(out)
            all_gt_aus.append(gt_aus)
            all_pred_aus.append(pred_aus)
            au_set.update(gt_aus)
            # au_set.update(pred_aus)

            inference_results.append({
                'image': inp.get('image', ''),
                'question': inp.get('question', ''),
                'gt_AUs': gt_aus,
                'gt_labels': gt_labels,
                'pred_AUs': pred_aus,
                'pred_labels': pred_labels,
                'description': out,
                'gt_description': inp.get('description', ''),
            })

        label_list = sorted(label_set)
        au_list = sorted(au_set, key=lambda x: int(x.replace('AU','')) if re.match('AU\d+', x) else 999)
        if label_list:
            label_metrics = multilabel_metrics(all_gt_labels, all_pred_labels, label_list, mode='label')
            pretty_print_metrics("Label", label_metrics, mode='label')
        else:
            label_metrics = None
        if au_list:
            au_metrics = multilabel_metrics(all_gt_aus, all_pred_aus, au_list, mode='au')
            pretty_print_metrics("AU", au_metrics, sort_key=lambda x: int(x[0][2:]) if x[0].startswith("AU") and x[0][2:].isdigit() else 999, mode='au')
        else:
            au_metrics = None

        # =========== 只保留真值和模型输出的文本，计算rouge并可选GPT评估 ==========
        desc_list = [r['description'] for r in inference_results]
        gt_desc_list = [r['gt_description'] for r in inference_results]
        rouge_mean, rouge_per_row = get_rouge_scores(desc_list, gt_desc_list)
        for i, row in enumerate(inference_results):
            row['gt_description'] = gt_desc_list[i]
            row['rouge'] = rouge_per_row[i]

        print("\n=== ROUGE between Description and GT Description ===")
        print("  ROUGE-1: {:.3f} | ROUGE-2: {:.3f} | ROUGE-L: {:.3f}".format(
            rouge_mean['rouge1'], rouge_mean['rouge2'], rouge_mean['rougeL']))

        mean_gpt_score = None
        if args.use_gpt:
            think_list = [extract_think_text(desc) for desc in desc_list]
            gpt_results = gpt_judge_alignment(
                think_list, gt_desc_list, openai_key=args.openai_key, base_url=args.base_url
            )
            gpt_scores = [r['score'] for r in gpt_results]
            for i, r in enumerate(gpt_results):
                inference_results[i]['gpt_score'] = int(r['score'])
                inference_results[i]['gpt_reason'] = r['reason']
            mean_gpt_score = float(np.mean(gpt_scores)) if gpt_scores else 0.0
            print("\n*** GPT Reasoning-Description Alignment (均分统计) ***")
            print("  GPT Score: {:.3f}".format(mean_gpt_score))

        save_results = {
            'label_metrics': label_metrics,
            'au_metrics': au_metrics,
            'rouge': rouge_mean,
            'gpt_score': mean_gpt_score,
            'inference_results': inference_results
        }
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        print("Results saved to", args.output_path)

    dist.barrier()