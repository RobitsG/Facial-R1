# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models.
"""

import logging
import os
import sys
import copy

import datasets
import torch
from torch.utils.data import Dataset
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
from open_r1.configs import SFTConfig
from open_r1.utils.callbacks import get_callbacks
from open_r1.prompts.emotion_prompt import SFT_PROMPT
import yaml
import json
import math
import random
from PIL import Image

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(42)

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from dataclasses import field
from qwen_vl_utils import process_vision_info
logger = logging.getLogger(__name__)
from dataclasses import dataclass

@dataclass
class SFTScriptArguments(ScriptArguments):
    image_root: str = field(default=None, metadata={"help": "The root directory of the image."})


processor = None

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: ScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999

                for data in datasets:
                    json_path = data.get("json_path")
                    json_path = data.get("json_path")
                    emotions = config[os.path.basename(json_path)]["emotions"]
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                if item.get("is_valid", True) == False:
                                    continue
                                item = json.loads(line.strip())
                                item['emotions'] = emotions
                                cur_data_dict.append(item)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        # Format into conversation
        def make_conversation_image(example):
            image_root = self.script_args.image_root
            image_path = os.path.join(image_root, example['image'])
            if example['question']:
                question = example['question'].relace('<image>', '').strip()
            else:
                question = "What is the emotion of this face?"

            label = example['labels'][0]
            return  [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": SFT_PROMPT.format(Question=question, Emotions=example['emotions'])},
                    ],
                },
                {
                    "role": "assistant",
                    "content": label,
                }
            ]
        example = self.list_data_dict[i]
        example["messages"] = make_conversation_image(example)
        return example



def collate_fn(examples):
    texts = [
        processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
        for example in examples
    ]
    image_inputs = []
    for example in examples:
        imgs, vids = process_vision_info(example["messages"])
        image_inputs.append(imgs)
    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################

    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)

    ################
    # Load tokenizer
    ################
    global processor
    if "vl" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        logger.info("Using AutoProcessor for vision-language model.")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
        )
        logger.info("Using AutoTokenizer for text-only model.")
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # training_args.model_init_kwargs = model_kwargs
    from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
    if "Qwen2-VL" in model_args.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    elif "Qwen2.5-VL" in model_args.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    else:
        raise ValueError(f"Unsupported model: {model_args.model_name_or_path}")
    ############################
    # Initialize the SFT Trainer
    ############################
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    training_args.remove_unused_columns = False
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    if trainer.accelerator.is_main_process:
        # 先用默认方法生成
        trainer.create_model_card()
        # 追加自定义内容
        model_card_path = os.path.join(training_args.output_dir, "README.md")
        with open(model_card_path, "a") as f:
            f.write(
                f"\n\n## Finetuned from\n{model_args.model_name_or_path}\n"
                f"\n## Dataset\n{str(script_args.dataset_name)}\n"
                f"\n## Tags\nopen-r1\n"
            )




if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    with open('open_r1/configs/config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    print(script_args)
    main(script_args, training_args, model_args)