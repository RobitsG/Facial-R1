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
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

import torch.distributed as dist  
import warnings  
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")  

# ===== 新增：依赖 =====
from rouge_score import rouge_scorer
try:
    import openai
except:
    openai = None

############# 辅助函数 ##############

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

def multilabel_metrics(gt_list, pred_list, class_names, mode='au'):
    n_class = len(class_names)
    class2idx = {c: i for i, c in enumerate(class_names)}

    def encode_batch(batch):
        y = np.zeros((len(batch), n_class), dtype=int)
        for i, labels in enumerate(batch):
            for l in labels:
                if l in class2idx:
                    y[i, class2idx[l]] = 1
        return y

    # 单标签模式的计算
    if mode == 'label':
        total = len(gt_list)
        correct = 0
        
        # 初始化每类的统计
        class_correct = np.zeros(n_class, dtype=int)
        class_total = np.zeros(n_class, dtype=int)
        
        for gt_labels, pred_labels in zip(gt_list, pred_list):
            gt_class = gt_labels[0]
            pred_class = pred_labels[0]
            
            if gt_class in class2idx:
                class_total[class2idx[gt_class]] += 1
                
                if gt_class == pred_class and pred_class in class2idx:
                    correct += 1
                    class_correct[class2idx[gt_class]] += 1
        
        # 整体准确率
        accuracy = correct / total if total > 0 else 0.0
        
        # 计算per_class指标（单标签模式）
        per_class = {}
        for i, name in enumerate(class_names):
            class_acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
            per_class[name] = {
                'acc': float(class_acc),  # 每类的准确率
                'recall': None,  # 单标签模式下recall和f1不适用
                'f1': None,
                'has_positive': bool(class_total[i] > 0)  # 是否有该类的样本
            }
        
        return {
            'accuracy': float(accuracy),
            'per_class': per_class
        }
    
    # 多标签模式的计算（保持不变）
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

    # 计算per_class指标（多标签模式）
    f1_each = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    recall_each = recall_score(Y_true, Y_pred, average=None, zero_division=0)
    acc_each = (Y_true == Y_pred).mean(axis=0)
    
    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            'acc': float(acc_each[i]),
            'recall': float(recall_each[i]) if valid_mask[i] else None,
            'f1': float(f1_each[i]) if valid_mask[i] else None,
            'has_positive': bool(valid_mask[i])
        }

    # 计算macro指标
    macro_f1 = f1_each[valid_mask].mean() if valid_class_idxs.size > 0 else 0
    macro_recall = recall_each[valid_mask].mean() if valid_class_idxs.size > 0 else 0
    macro_acc = acc_each[valid_mask].mean() if valid_class_idxs.size > 0 else 0

    # 如果只需要micro指标
    if mode == 'au':
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

def pretty_print_metrics(name, metrics, mode='au', sort_key=None):
    # 根据模式选择标题
    if mode == 'label':
        print(f"\n=== {name} (单标签) 指标 ===")
        print("  整体准确率: {:.3f}".format(metrics['accuracy']))
    elif mode == 'au':
        print(f"\n=== {name} (多标签) 指标 ===")
        print("  Macro指标:  Accuracy: {:.3f} | Recall: {:.3f} | F1: {:.3f}".format(
            metrics['macro']['accuracy'], metrics['macro']['recall'], metrics['macro']['f1']))
        print("  Micro指标:  Accuracy: {:.3f} | Recall: {:.3f} | F1: {:.3f}".format(
            metrics['micro']['accuracy'], metrics['micro']['recall'], metrics['micro']['f1']))
    
    # 打印per-class结果
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
        
        # 处理单标签模式下recall和f1为None的情况
        recall_str = str(v_recall)[:7] if v_recall is not None else "None"
        f1_str = str(v_f1)[:7] if v_f1 is not None else "None"
        
        print(f"  {k:<14} {v_acc:7.3f} {recall_str:>7} {f1_str:>7} {info:>10}")


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

