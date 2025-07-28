from transformers import AutoModel, AutoTokenizer
import torch  
import json  
import time
from tqdm import tqdm  
import re  
import os  
import random  
import argparse
import openai
from sklearn.metrics import f1_score, recall_score, accuracy_score
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

import torch.distributed as dist  
import warnings  
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")  

from open_r1.prompts.emotion_prompt import GRPO_PROMPT, SFT_PROMPT
from open_r1.utils.eval_utils import *

# ===== 新增：依赖 =====
from rouge_score import rouge_scorer
try:
    import openai
except:
    openai = None

# ===== InternVL3图像处理函数 =====
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # 计算已存在的图像宽高比
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # 找到最接近目标的宽高比
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # 计算目标宽度和高度
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # 调整图像大小
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # 分割图像
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

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

model = AutoModel.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map={"": local_rank},
)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

# 确保设置了pad_token，避免警告
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

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

    # InternVL3-8B推理部分
    rank_outputs = []
    
    # 使用批处理方式优化推理效率
    batch_size = min(args.bsz, len(rank_data))  # 确保batch_size不大于数据量
    
    # 配置生成参数
    generation_config = {
        "max_new_tokens": 1024, 
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id  # 显式指定pad_token_id
    }
    
    # 批处理推理
    for i in tqdm(range(0, len(rank_data), batch_size), disable=rank!=args.main_rank):
        batch_data = rank_data[i:i+batch_size]
        batch_responses = []

        # 检查是否支持批处理
        if hasattr(model, "batch_chat") and callable(model.batch_chat):
            try:
                # 尝试第一种批处理方式 - 使用单个批处理tensor
                batch_pixel_tensors = []
                batch_questions = []
                
                for x in batch_data:
                    img_path = os.path.join(args.image_root, x['image'])
                    question = x['question'].replace('<image>', '').strip() if x['question'] else "What is the emotion of this face?"
                    
                    # 加载和处理图像
                    pixel_values = load_image(img_path, max_num=12)
                    
                    if torch.cuda.is_available():
                        pixel_values = pixel_values.to(device)
                        if pixel_values.dtype != torch.bfloat16:
                            pixel_values = pixel_values.to(torch.bfloat16)
                    
                    # 格式化问题
                    formatted_question = f"<image>\n{QUESTION_PROMPT.format(Question=question, Emotions=emotions.keys())}"
                    
                    batch_pixel_tensors.append(pixel_values)
                    batch_questions.append(formatted_question)
                
                # 为batch_chat准备单个批处理tensor - 关键修改
                # 需要将所有图像拼接成一个大batch
                all_images = torch.cat([tensor for tensor in batch_pixel_tensors], dim=0)
                num_patches_list = [tensor.shape[0] for tensor in batch_pixel_tensors]
                
                # 批量推理
                batch_responses = model.batch_chat(
                    tokenizer, 
                    all_images,  # 注意：这里是所有图像拼接后的单个tensor
                    batch_questions, 
                    generation_config,
                    num_patches_list=num_patches_list
                )
            except Exception as e:
                print(f"批处理失败，错误: {e}")
                print("退回到单样本处理...")
                # 如果批处理失败，退回到单样本处理
                batch_responses = []
                for x in batch_data:
                    img_path = os.path.join(args.image_root, x['image'])
                    question = x['question'].replace('<image>', '').strip() if x['question'] else "What is the emotion of this face?"
                    
                    pixel_values = load_image(img_path, max_num=12)
                    if torch.cuda.is_available():
                        pixel_values = pixel_values.to(device)
                        if pixel_values.dtype != torch.bfloat16:
                            pixel_values = pixel_values.to(torch.bfloat16)
                    
                    formatted_question = f"<image>\n{QUESTION_PROMPT.format(Question=question, Emotions=emotions.keys())}"
                    response = model.chat(tokenizer, pixel_values, formatted_question, **generation_config)
                    batch_responses.append(response)
        else:
            # 不支持批处理，单样本处理
            for x in batch_data:
                img_path = os.path.join(args.image_root, x['image'])
                question = x['question'].replace('<image>', '').strip() if x['question'] else "What is the emotion of this face?"
                
                pixel_values = load_image(img_path, max_num=12)
                if torch.cuda.is_available():
                    pixel_values = pixel_values.to(device)
                    if pixel_values.dtype != torch.bfloat16:
                        pixel_values = pixel_values.to(torch.bfloat16)
                
                formatted_question = f"<image>\n{QUESTION_PROMPT.format(Question=question, Emotions=emotions.keys())}"
                try:
                    response = model.chat(tokenizer, pixel_values, formatted_question, **generation_config)
                except TypeError:
                    response = model.chat(tokenizer, pixel_values, formatted_question, {
                        "max_new_tokens": generation_config["max_new_tokens"],
                        "do_sample": generation_config["do_sample"]
                    })
                batch_responses.append(response)

        # 收集批处理结果
        rank_outputs.extend(batch_responses)

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