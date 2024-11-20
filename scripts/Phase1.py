# Phase 1 of PL-SEFT
# Involves translation from X -> en

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"
os.environ["WANDB_DISABLED"] = "true" 

from dataclasses import dataclass, field
from typing import Optional
from datasets.arrow_dataset import Dataset
import torch
import random
from datasets import load_dataset, concatenate_datasets, config
from peft import LoraConfig
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    set_seed as hf_set_seed,
)
import tensorflow_datasets as tfds
import numpy as np
import random
import json 
import trl
from trl import SFTTrainer
from huggingface_hub import login
import sys

login(token = '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')
config.HF_DATASETS_CACHE = './tmp_for_hf_datasets_cache' # Temporary cache for HF datasets
hf_set_seed(42)

MODEL_NAME = 'ai-forever/mGPT'
MODEL_TYPE = 'mgpt'
TARGET_LANG = 'bn'
TASK_TYPE = 'flores-phase1'

TRANSLATION_DATASET_PATH = f"../datasets/flores_in/flores_{TARGET_LANG}_en_test.json"
BOS_TOKEN, EOS_TOKEN = "<s>", "</s>" # Need not change this, gets changed automatically

# Do change save_strategy -- "steps" or "epoch" in training arguments
STAGE = "RA"
NUM_TRAIN_EPOCHS = 10
DATASET_NAME = "NULL"
USE_4BIT = False
USE_NESTED_QUANT = False
BNB_4BIT_COMPUTE_DTYPE = "float16"
BNB_4BIT_QUANT_TYPE = "nf4"
FP16 = False
BF16 = True
PACKING = False
GRADIENT_CHECKPOINTING = True
OPTIM = "paged_adamw_32bit"
LR_SCHEDULER_TYPE = "constant"
OPTIM_STEPS = -1
WARMUP_RATIO = 0.03
GROUP_BY_LENGTH = True
SAVE_STEPS = 60
LOGGING_STEPS = 10
MERGE_AND_PUSH = False
SEED = 42
OUTPUT_DIR = f"../models/{MODEL_TYPE}/{TASK_TYPE}/{TARGET_LANG}"

os.system(f"mkdir -p {OUTPUT_DIR}")

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})

    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=1024)
    model_name: Optional[str] = field(
        default=MODEL_NAME,
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )
    model_type: Optional[str] = field(
        default=MODEL_TYPE,
        metadata={"help": "The type of model to train."},
    )
    translation_dataset_path : Optional[str] = field(
        default=TRANSLATION_DATASET_PATH,
        metadata={"help": "The list of paths to the translation dataset."},
    )
    dataset_name: Optional[str] = field(
        default=DATASET_NAME,
        metadata={"help": "The preference dataset to use."},
    )
    stage: Optional[str] = field(
        default=STAGE,
        metadata={"help": "The stage of the training."},
    )
    use_4bit: Optional[bool] = field(
        default=USE_4BIT,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=USE_NESTED_QUANT,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default=BNB_4BIT_COMPUTE_DTYPE,
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default=BNB_4BIT_QUANT_TYPE,
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=NUM_TRAIN_EPOCHS,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=FP16,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=BF16,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=PACKING,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=GRADIENT_CHECKPOINTING,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default=OPTIM,
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default=LR_SCHEDULER_TYPE,
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=OPTIM_STEPS, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=WARMUP_RATIO, metadata={"help": "Fraction of steps to do a warmup for"})
    group_by_length: bool = field(
        default=GROUP_BY_LENGTH,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: int = field(default=SAVE_STEPS, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=LOGGING_STEPS, metadata={"help": "Log every X updates steps."})
    merge_and_push: Optional[bool] = field(
        default=MERGE_AND_PUSH,
        metadata={"help": "Merge and push weights after training"},
    )
    output_dir: str = field(
        default=OUTPUT_DIR,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

lang_map = {
    "te" : "Telugu",
    "hi" : "Hindi",
    "bn" : "Bengali",
    "mr" : "Marathi",
    "ta" : "Tamil",
    "ur" : "Urdu",
}

def gen_batches_train():
    print("Loading train data for Phase 1 translation from")
    print(script_args.translation_dataset_path)
    translation_data = json.load(open(script_args.translation_dataset_path))["examples"]
    trans_data = []
    for data in translation_data:
            trans_data.append([data["source"], data["target"]])

    total_samples = len(trans_data)
    train_limit = total_samples
    counter = 0
    # Yielding Question Translation Instructions
    for sample in iter(trans_data):
        if counter >= train_limit:
            break 
        new_text_format = f'{BOS_TOKEN}{sample[0]}\n\nQ: Translate the above text from {lang_map[TARGET_LANG]} to English.\nA: {sample[1]}{EOS_TOKEN}'
        if counter == 0:
            print(new_text_format)
            print("BOS token is", BOS_TOKEN)
            print("EOS token is", EOS_TOKEN)
        yield {'text': new_text_format}
        counter += 1

# This entire function doesn't matter
def gen_batches_val():
    print("Loading validation data for Phase 1 translation from")
    print(script_args.translation_dataset_path)
    translation_data = json.load(open(script_args.translation_dataset_path))["examples"]
    trans_data = []
    for data in translation_data:
            trans_data.append([data["source"], data["target"]])

    total_samples = len(trans_data)
    train_limit = 10
    counter = 0
    # Yielding Question Translation Instructions
    for sample in iter(trans_data):
        if counter >= train_limit:
            break 
        new_text_format = f'{BOS_TOKEN}{sample[0]}\n\nQ: Translate the above text from {lang_map[TARGET_LANG]} to English.\nA: {sample[1]}{EOS_TOKEN}'
        yield {'text': new_text_format}
        counter += 1

def create_and_prepare_model(args):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)

    device_map = {"": 0}
    # device_map = "auto" # multi-gpu

    model = None
    if MODEL_TYPE in ["mgpt"]:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, 
            device_map=device_map, 
            token='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',
        )
    else:
        raise ValueError("Invalid MODEL_TYPE")

    def get_target_modules(model_type):
        if model_type == "mgpt":
            return [
                'c_proj', # present in both SdpaAttention and MLP
                'c_attn', # present only in SdpaAttention
                'c_fc' # present only in MLP
            ]
        else: return NotImplementedError
    
    model.config.pretraining_tp = 1 
    peft_config = LoraConfig(
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM", 
        target_modules=get_target_modules(script_args.model_type),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name, 
        trust_remote_code=True, 
        token='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')
    tokenizer.pad_token = tokenizer.eos_token
    return model, peft_config, tokenizer

training_arguments = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    eval_strategy="epoch",
    save_strategy="epoch",
    # save_strategy="steps",
    num_train_epochs=script_args.num_train_epochs,
    max_grad_norm=script_args.max_grad_norm,
    max_steps=script_args.max_steps,
    warmup_ratio=script_args.warmup_ratio,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    report_to=[],
)

model, peft_config, tokenizer = create_and_prepare_model(script_args)
model.config.use_cache = False
if script_args.gradient_checkpointing:
    model.gradient_checkpointing_enable()
    print("Gradient Checkpointing Enabled")

BOS_TOKEN = tokenizer.bos_token
EOS_TOKEN = tokenizer.eos_token

print("Model and Tokenizer loaded")
print("BOS token is", BOS_TOKEN)
print("EOS token is", EOS_TOKEN)

train_gen = Dataset.from_generator(gen_batches_train)
val_gen = Dataset.from_generator(gen_batches_val)

tokenizer.padding_side = "right"
trainer = SFTTrainer(
    model=model,
    train_dataset=train_gen,
    eval_dataset=val_gen, 
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=script_args.packing,
)

trainer.train()