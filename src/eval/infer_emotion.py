from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import os
import argparse
import torch.distributed as dist
import warnings
import re

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

def extract_aus(description):
    """用正则表达式提取AU编号，如AU6、AU12等，返回去重列表"""
    return sorted(set(re.findall(r"AU\d+", description)))

def extract_labels(description):
    """从<answer>标签中提取label，多个用逗号分隔并去除两侧空白"""
    answer_match = re.search(r"<answer>(.*?)</answer>", description, flags=re.DOTALL)
    if not answer_match:
        return []
    labels_text = answer_match.group(1)
    # 用 , 、 ，分割，去除前后空白
    return sorted(set(label.strip() for label in re.split(r"[,\u3001，]", labels_text) if label.strip()))

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
    parser.add_argument("--max_retry", type=int, default=5, help="Maximum retry times for incorrect predictions")
    return parser.parse_args()


def get_error_messages(gold_aus, gold_labels, pred_aus, pred_labels, output):
    """获取错误信息列表"""
    errors = []
    
    # 检查AU错误
    if gold_aus:  # 只在有gold_aus时检查
        if gold_aus != pred_aus:
            errors.append(f"The predicted AUs must be: {gold_aus}")
    
    # 检查label错误
    if gold_labels:  # 只在有gold_labels时检查
        if gold_labels != pred_labels:
            errors.append(f"The predicted labels must be: {gold_labels}")

    # 检查输出格式
    think_match = re.search(r"<think>(.*?)</think>", output, flags=re.DOTALL)
    if not think_match or not think_match.group(1).strip():
        errors.append("Missing or empty <think>...</think> tag.")
    answer_match = re.search(r"<answer>(.*?)</answer>", output, flags=re.DOTALL)
    if not answer_match or not answer_match.group(1).strip():
        errors.append("Missing or empty <answer>...</answer> tag.")

    return errors


def print_retry_info(item_id, image, retry_num, pred_aus, pred_labels, match_aus, match_labels, errors, output, rank):
    """打印单次重试信息"""
    if rank == 0:  # 只在主进程打印
        print(f"\n===== 样本ID: {item_id}, 图片: {image}, 尝试 #{retry_num} =====")
        print(f"预测AUs: {pred_aus}")
        print(f"预测labels: {pred_labels}")
        print(f"AU匹配: {match_aus}, label匹配: {match_labels}")
        if errors:
            print(f"错误: {', '.join(errors)}")
        print(f"输出: {output}")
        print("="*60)

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

    # 定义重试提示词模板
    retry_template = """
### Previous Response Issues
Here is your previous response:
{prev_response}

Your previous response had the following issues:
{issues}

Please correct these issues in your new response. Make sure to:
1. Include all required Action Units in your analysis
2. Provide the correct emotion in your answer
3. Use proper <think>...</think> and <answer>...</answer> tags
4. Avoid negative expressions like "no", "not", "without"
5. Be concise and precise
""".strip()

    # 初始推理和重试
    rank_outputs = []
    for idx, item in enumerate(tqdm(rank_data, desc=f"Rank {rank} inference", disable=rank!=0)):
        item_id = item.get("id", f"item_{idx}")
        img_name = item.get("image", "")
        
        img_path = os.path.join(args.image_root, img_name) if 'image' in item else None
        question = item['question'].replace('<image>','').strip() if 'question' in item and item['question'] else "What is the emotion of this face?"
        
        if not img_path:
            rank_outputs.append(None)
            continue
            
        # 获取真实标签
        gold_aus = sorted(set(item.get("AUs", [])))
        gold_labels = sorted(set(item.get("labels", [])))
        
        # 使用GRPO_PROMPT作为基础提示词
        curr_prompt = GRPO_PROMPT.format(
            Question=question, 
            Emotions=emotions.keys(),
        )
        
        # 根据是否有AUs和labels动态添加Ground Truth部分
        gt_parts = []
        if gold_aus:
            gt_parts.append(f"Your analysis MUST identify these specific Action Units: {gold_aus}")
        if gold_labels:
            gt_parts.append(f"And your final answer MUST be this exact emotion: {gold_labels}")
            
        # 如果有Ground Truth信息，添加到提示词
        if gt_parts:
            gt_section = "### Ground Truth\n" + "\n".join(gt_parts)
            curr_prompt += "\n\n" + gt_section
        
        # 最多重试次数（包括初始尝试）
        best_output = None
        is_correct = False
        
        for retry in range(args.max_retry + 1):  # +1 是因为包括初始尝试
            # 准备单条数据的消息
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{img_path}"},
                        {"type": "text", "text": curr_prompt}
                    ]
                }
            ]
            
            # 单条数据的处理
            text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info([message])
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)
            
            gen = model.generate(**inputs, use_cache=True, max_new_tokens=args.max_new_tokens, do_sample=False)
            trimmed = gen[0][len(inputs.input_ids[0]):]
            output = processor.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            # 第一次尝试的结果保存为best_output，以防后续都失败
            if retry == 0:
                best_output = output
            
            # 验证结果
            pred_aus = extract_aus(output)
            pred_labels = extract_labels(output)
            
            # 验证逻辑 - 只验证有真实值的部分
            match_aus = True if not gold_aus else gold_aus == pred_aus
            match_labels = True if not gold_labels else gold_labels == pred_labels
            # if match_labels == False:
            #     print('pred_labels', pred_labels)
            #     print('gold_labels', gold_labels)
            #     print(output)
            #     exit()
            
            # 获取错误信息
            errors = get_error_messages(gold_aus, gold_labels, pred_aus, pred_labels, output)
            
            # 即时打印重试信息（如果有错误）
            if retry > 0 or not (match_aus and match_labels):
                print_retry_info(item_id, img_name, retry, pred_aus, pred_labels, 
                                match_aus, match_labels, errors, output, rank)
            
            # 如果匹配成功，保存该结果
            if match_aus and match_labels:
                best_output = output
                is_correct = True
                if retry > 0:  # 如果是重试成功，打印成功信息
                    if rank == 0:
                        print(f"\n✅ 样本 {item_id} 在第 {retry} 次重试后成功修正！")
                break
                
            # 如果不是最后一次尝试，则准备重试
            if retry < args.max_retry and errors:
                # 构建重试提示词
                curr_prompt = curr_prompt + '\n\n' + retry_template.format(
                    prev_response=output,
                    issues=", ".join(errors)
                )
                
                if rank == 0:
                    print(f"🔄 样本 {item_id} 准备第 {retry+1} 次重试...")
                
        # 如果所有重试都失败，打印最终失败信息
        if not is_correct and rank == 0:
            print(f"\n❌ 样本 {item_id} 经过 {args.max_retry} 次重试后仍未成功。")
            
        # 保存最好的结果（可能是正确的，也可能是所有尝试都失败后的初始结果）
        rank_outputs.append((best_output, is_correct))
    
    # 收集所有结果
    all_outputs = [(None, False)] * len(data)
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

        # 统计和写入结果
        total = 0
        correct = 0
        filtered_data = []
        for i, item in enumerate(data):
            if all_outputs[i] is not None:  # 确保有输出
                total += 1
                output, is_correct = all_outputs[i]
                
                if is_correct:
                    correct += 1
                    output_item = item.copy()
                    output_item["description"] = output
                    filtered_data.append(output_item)
        
        # 写入仅包含正确预测的输出jsonl
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            for item in filtered_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        # 打印统计信息
        print(f"\n===== 最终统计 =====")
        print(f"总样本数: {total}")
        print(f"AUs和labels均正确的样本数: {correct}")
        if total > 0:
            print(f"准确率: {correct/total:.4f}")
        else:
            print("无有效样本")
            
        print(f"Results saved to {args.output_file}")

    try:
        dist.barrier()
    except:
        pass

if __name__ == "__main__":
    main()