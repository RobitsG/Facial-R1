
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

############# 辅助函数 ##############

import numpy as np

def extract_au_pred_think(model_output):
    m = re.search(r'<think>(.*?)</think>', model_output, re.DOTALL)
    in_think = m.group(1) if m else model_output
    aus = re.findall(r'AU\d+', in_think)
    return sorted(set(aus), key=lambda x: int(x.replace('AU','')))

def extract_label_pred_answer(model_output):
    m = re.search(r'<answer>(.*?)</answer>', model_output, re.DOTALL)
    ans = m.group(1).strip() if m else model_output
    return [x.strip() for x in re.split(r'[,;，；\s]+', ans) if x.strip()]

def extract_think_text(model_output):
    m = re.search(r'<think>(.*?)</think>', model_output, re.DOTALL)
    return m.group(1).strip() if m else ""

from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import numpy as np

def multilabel_metrics(gt_list, pred_list, class_names):
    n_class = len(class_names)
    class2idx = {c: i for i, c in enumerate(class_names)}

    def encode_batch(batch):
        y = np.zeros((len(batch), n_class), dtype=int)
        for i, labels in enumerate(batch):
            for l in labels:
                if l in class2idx:
                    y[i, class2idx[l]] = 1
        return y

    Y_true = encode_batch(gt_list)
    Y_pred = encode_batch(pred_list)

    # 统计哪些类别有正样本
    valid_mask = (Y_true.sum(axis=0) > 0)
    valid_class_idxs = np.where(valid_mask)[0]
    valid_class_names = [class_names[i] for i in valid_class_idxs]

    # micro 统计——全局
    micro_f1 = f1_score(Y_true, Y_pred, average='micro', zero_division=0)
    micro_recall = recall_score(Y_true, Y_pred, average='micro', zero_division=0)
    micro_acc = accuracy_score(Y_true, Y_pred)

    # macro 只对有正例类别做平均
    f1_each = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    recall_each = recall_score(Y_true, Y_pred, average=None, zero_division=0)
    acc_each = (Y_true == Y_pred).mean(axis=0)
    macro_f1 = f1_each[valid_mask].mean() if valid_class_idxs.size > 0 else 0
    macro_recall = recall_each[valid_mask].mean() if valid_class_idxs.size > 0 else 0
    macro_acc = acc_each[valid_mask].mean() if valid_class_idxs.size > 0 else 0

    # 用于per-class输出，如果该类别无正例，则加`None`或0或其他特殊标记
    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            'acc': float(acc_each[i]),
            'recall': float(recall_each[i]) if valid_mask[i] else None,
            'f1': float(f1_each[i]) if valid_mask[i] else None,
            'has_positive': bool(valid_mask[i])
        }

    return {
        'macro': {
            'accuracy': float(macro_acc),
            'recall': float(macro_recall),
            'f1': float(macro_f1)
        },
        'micro': {
            'accuracy': float(micro_acc),
            'recall': float(micro_recall),
            'f1': float(micro_f1)
        },
        'per_class': per_class
    }

def pretty_print_metrics(name, metrics, sort_key=None):
    print(f"\n=== {name} (多标签) 指标 ===")
    print("  Macro指标:  Accuracy: {:.3f} | Recall: {:.3f} | F1: {:.3f}".format(
        metrics['macro']['accuracy'], metrics['macro']['recall'], metrics['macro']['f1']))
    print("  Micro指标:  Accuracy: {:.3f} | Recall: {:.3f} | F1: {:.3f}".format(
        metrics['micro']['accuracy'], metrics['micro']['recall'], metrics['micro']['f1']))
    print("\n  Per-Class Results:")
    headers = f"{'Class':<14} {'Acc':>7} {'Recall':>7} {'F1':>7} {'Info':>10}"
    print("  " + headers)
    print("  " + "-" * len(headers))
    items = list(metrics['per_class'].items())
    if sort_key is not None:
        items = sorted(items, key=sort_key)
    else:
        items = sorted(items)
    for k, v in items:
        v_acc = v['acc']
        v_recall = v['recall']
        v_f1 = v['f1']
        info = ""
        if not v['has_positive']:
            info = "(无正例)"
        print(f"  {k:<14} {v_acc:7.3f} {str(v_recall)[:7]:>7} {str(v_f1)[:7]:>7} {info:>10}")

# ===== ROUGE辅助函数 =====
def get_rouge_scores(hyps, refs):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    result_rows = []
    r1s, r2s, rls = [], [], []
    for h, r in zip(hyps, refs):
        rv = scorer.score(r, h)
        r1s.append(rv['rouge1'].fmeasure)
        r2s.append(rv['rouge2'].fmeasure)
        rls.append(rv['rougeL'].fmeasure)
        result_rows.append({'rouge1':rv['rouge1'].fmeasure, 'rouge2':rv['rouge2'].fmeasure, 'rougeL':rv['rougeL'].fmeasure})
    mean = {'rouge1': float(np.mean(r1s)), 'rouge2': float(np.mean(r2s)), 'rougeL': float(np.mean(rls))}
    return mean, result_rows

# ===== GPT辅助函数 =====
def gpt_judge_alignment(
        think_list, desc_list, openai_key=None, base_url=None, max_retry=3, sleep_time=2
    ):
    if openai is None:
        raise ImportError("需要 pip install openai")
    if openai_key is not None:
        client = openai.OpenAI(api_key=openai_key, base_url=base_url) if base_url else openai.OpenAI(api_key=openai_key)
    elif os.environ.get("OPENAI_API_KEY"):
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=base_url) if base_url else openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    else:
        raise RuntimeError('No OPENAI_API_KEY provided')

    results = []
    for idx, (think, desc) in enumerate(tqdm(list(zip(think_list, desc_list)), desc='GPT评估', total=len(think_list))):
        prompt = (
            "请判断下面“模型推理内容”与“标准描述”之间的一致程度。\n"
            "请返回1-10的分数，10为高度一致，1为完全不一致。\n"
            "请严格只用如下标准Json完整返回：\n"
            '{"score":分数, "reason":"简要理由"}\n'
            "模型推理内容：%s\n"
            "标准描述：%s\n" % (think, desc)
        )
        retry_cnt = 0
        while retry_cnt < max_retry:
            try:
                response = client.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=128, temperature=0.0,
                )
                output = response.choices[0].message.content.strip()
                # 尝试直接json解析
                try:
                    data = json.loads(output)
                except Exception:
                    # 尝试用正则提取json字符串片段再解析
                    json_match = re.search(r'(\{.*?\})', output, re.DOTALL)
                    if json_match:
                        try:
                            data = json.loads(json_match.group(1))
                        except Exception:
                            data = None
                    else:
                        data = None

                if data and "score" in data and "reason" in data:
                    try:
                        score = int(data["score"])
                        reason = str(data["reason"])
                        if 1 <= score <= 10:
                            results.append({'score': score, 'reason': reason})
                            break
                        else:
                            raise ValueError("score 超出范围")
                    except Exception as e:
                        # 解析json但score有问题，则重试
                        retry_cnt += 1
                        if retry_cnt >= max_retry:
                            results.append({'score': 0, 'reason': f"得分异常/解析异常:{output}"})
                            break
                        time.sleep(sleep_time)
                        continue
                else:
                    retry_cnt += 1
                    if retry_cnt >= max_retry:
                        results.append({'score': 0, 'reason': f"Json解析失败:{output}"})
                        break
                    time.sleep(sleep_time)
            except Exception as e:
                retry_cnt += 1
                if retry_cnt >= max_retry:
                    results.append({'score': 0, 'reason': f"请求失败:{e}"})
                    break
                time.sleep(sleep_time)
    return results

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
    generation_config = dict(max_new_tokens=256, do_sample=False)
    
    for x in tqdm(rank_data, disable=rank!=args.main_rank):
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
        
        # 使用模型推理
        response = model.chat(tokenizer, pixel_values, formatted_question, generation_config)
        rank_outputs.append(response)

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
            label_metrics = multilabel_metrics(all_gt_labels, all_pred_labels, label_list)
            pretty_print_metrics("Label", label_metrics)
        else:
            label_metrics = None
        if au_list:
            au_metrics = multilabel_metrics(all_gt_aus, all_pred_aus, au_list)
            pretty_print_metrics("AU", au_metrics, sort_key=lambda x: int(x[0][2:]) if x[0].startswith("AU") and x[0][2:].isdigit() else 999)
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
