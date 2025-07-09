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

from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(42)

# -------- AU评估函数 --------

def sort_au_key(au):
    return int(au.replace('AU', ''))

def AU2Labels(AU_string, AU2indices):
    pattern = r"AU(\d+)"
    matches = re.findall(pattern, AU_string)
    AU_labels = [0] * len(AU2indices)
    for num in matches:
        try:
            idx = AU2indices[num]
            AU_labels[idx] = 1
        except:
            continue
    return AU_labels

def evaluate_au_f1(data, target_aus=None):
    if target_aus is None or not target_aus:
        all_aus = set()
        for item in data:
            if "AUs" in item:
                all_aus.update(item["AUs"])
        all_aus = sorted(list(all_aus), key=sort_au_key)
    else:
        all_aus = sorted(list(set(target_aus)), key=sort_au_key)

    AU2indices = {str(i).replace('AU', ''): idx for idx, i in enumerate(all_aus)}
    true_labels = defaultdict(list)
    pred_labels = defaultdict(list)

    for item in data:
        for au in all_aus:
            true_labels[au].append(1 if au in item.get("AUs", []) else 0)
        pred_au_string = item.get('prediction', '')
        au_labels = AU2Labels(pred_au_string, AU2indices)
        for idx, au in enumerate(all_aus):
            pred_labels[au].append(au_labels[idx])

    results = {}
    for au in all_aus:
        if sum(true_labels[au]) + sum(pred_labels[au]) == 0:
            f1 = 1.0
        else:
            f1 = f1_score(true_labels[au], pred_labels[au])
        results[au] = f1
    macro_f1 = sum(results.values()) / len(results) if results else 0
    return results, macro_f1

def print_results(au_f1_scores, macro_f1):
    print("the F1 score of each AU:")
    for au in sorted(au_f1_scores.keys(), key=sort_au_key):
        print(f"{au}: {au_f1_scores[au]:.4f}")
    print(f"\naverage AU F1 score: {macro_f1:.4f}")

# -------- 多标签label辅助函数 --------

def extract_pred_labels(pred_output, test_mode):
    if test_mode == "grpo":
        m = re.search(r'<answer>(.*?)</answer>', pred_output, re.DOTALL)
        ans = m.group(1).strip() if m else pred_output
    else:
        ans = pred_output
    labels = [x.strip() for x in re.split(',|;|，|；', ans) if x.strip()]
    return labels

# -------- 分布式相关 --------

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, dist.get_world_size(), dist.get_rank()

local_rank, world_size, rank = setup_distributed()
device = f"cuda:{local_rank}"
print(f"Process {rank} using {device}")

# -------- 参数 --------

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
    parser.add_argument("--num_samples", type=int, default=100, help="测试样本数量")
    parser.add_argument("--eval_targets", nargs='+', default=["label", "au"], choices=["label", "au"], help="选择要评估的目标，可为 label（情感分类准确率）、au（AU F1分数），可以多选")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")
    parser.add_argument("--output_path", type=str, default=None, help="输出日志路径")
    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args()
    if args.model_path is None:
        args.model_path = f"{args.home_path}/output/checkpoints/{args.run_name}"
    if args.output_path is None:
        args.output_path = f"{args.home_path}/output/logs/emotion_{args.run_name}.json"
    return args

args = parse_args()

# -------- 模型与处理器 --------
with open(args.config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": local_rank},
)
processor = AutoProcessor.from_pretrained(args.model_path)

# -------- 主评估循环 --------

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
    random.seed(42)
    random.shuffle(data)
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

    messages = []
    for x in rank_data:
        img = os.path.join(args.image_root, x['image'])
        question = x['question'].replace('<image>', '').strip() if x['question'] else "What is the emotion of this face?"
        messages.append([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img}"},
                    {"type": "text", "text": QUESTION_PROMPT.format(Question=question, Emotions=emotions)}
                ]
            }
        ])

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

        final_results = []
        all_gt_labels = []
        all_pred_labels = []
        label_set = set()

        for inp, out in zip(data, all_outputs):
            # 真值labels与预测labels均可能为多标签
            gt_labels = inp.get('labels', [])
            label_set.update(gt_labels)
            pred_labels = extract_pred_labels(out, args.test_mode)
            label_set.update(pred_labels)

            # AU真值和预测
            gt_AUs = inp.get("AUs", [])
            au_pattern = r"AU(\d+)"
            pred_AUs = sorted(list(set(f"AU{num}" for num in re.findall(au_pattern, out))),
                              key=lambda x: int(x.replace('AU',''))) if out else []

            final_results.append({
                "image": inp['image'],
                "question": inp['question'],
                "model_output": out,
                "gt_labels": gt_labels,
                "pred_labels": pred_labels,
                "gt_AUs": gt_AUs,
                "pred_AUs": pred_AUs
            })
            all_gt_labels.append(gt_labels)
            all_pred_labels.append(pred_labels)

        save_results = {'results': final_results}

        if "label" in args.eval_targets:
            label_classes = sorted(list(label_set))
            label2idx = {l: i for i, l in enumerate(label_classes)}
            def binarize(labels_list):
                y = np.zeros(len(label_classes), dtype=int)
                for l in labels_list:
                    if l in label2idx:
                        y[label2idx[l]] = 1
                return y
            Y_true = np.array([binarize(lbls) for lbls in all_gt_labels])
            Y_pred = np.array([binarize(lbls) for lbls in all_pred_labels])
            micro_f1 = f1_score(Y_true, Y_pred, average='micro', zero_division=1)
            macro_f1 = f1_score(Y_true, Y_pred, average='macro', zero_division=1)
            subset_acc = accuracy_score(Y_true, Y_pred)
            print(f"Label multi-label eval: micro-F1={micro_f1:.4f}, macro-F1={macro_f1:.4f}, subset-acc={subset_acc:.4f}")
            save_results["micro_label_f1"] = micro_f1
            save_results["macro_label_f1"] = macro_f1
            save_results["label_subset_acc"] = subset_acc

        if "au" in args.eval_targets:
            au_eval_input = []
            for item in final_results:
                au_eval_input.append({
                    "AUs": item["gt_AUs"],
                    "prediction": " ".join(item["pred_AUs"])
                })
            au_f1_scores, macro_f1 = evaluate_au_f1(au_eval_input, target_aus=None)
            print_results(au_f1_scores, macro_f1)
            save_results['AU_f1'] = au_f1_scores
            save_results['AU_macro_f1'] = macro_f1

        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(save_results, f, indent=2)
        print("Results saved to", args.output_path)

    dist.barrier()