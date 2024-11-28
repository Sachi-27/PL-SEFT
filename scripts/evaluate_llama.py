# Works for Llama-3.2 models

import os 

os.environ['CUDA_VISIBLE_DEVICES'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['HF_HOME'] = '../hf_home'

import torch
import json
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from evaluate import load 
import tensorflow_datasets as tfds
from tqdm import tqdm
# set seeds
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device_map = {"": 0} 

model_id = "meta-llama/Llama-3.1-8B-Instruct"
test_dataset_path = "../datasets/mgsm_in/corrected/mgsm_en.json"
TEMPLATE = "alpaca"
results_file_path = f"../results/mgsm_en_llama3.1-8b-instruct-{TEMPLATE}.json"
NUM_SAMPLES_TO_EVAL = 50
model_is_trained_by_me = False

#### SCRIPT ASSUMES EMPTY INPUT FIELD IN TEST DATA
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


with open(test_dataset_path, 'r') as f:
    test_data = json.load(f)

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def generate_prompt(instruction, template):
    if template == "alpaca":
        return ALPACA_PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
    if template == "raw": # also for llama3.2 no instruct
        return f"{instruction}\nAnswer:"
    elif template == "llama3.2_instruct_orig":
        return f"<|start_header_id|>user<|end_header_id|>\n\nBelow is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Question :\n{instruction}\nYour response should end with \"The final answer is [the_answer_value]\" where the [the_answer_value] is an integer.<|eot_id|> <|start_header_id|>assistant<|end_header_id|>\n\nThe final answer is"
    else: return ValueError("Invalid template")

model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        token="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"
    )
tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    token="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",
)

if model_is_trained_by_me:
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16, 
    device_map=device_map,
)

results = []
for i in tqdm(range(NUM_SAMPLES_TO_EVAL)):
    torch.cuda.empty_cache()
    # script assume empty input
    prompt = generate_prompt(test_data[i]['instruction'], template=TEMPLATE)
    # Tokenize the prompt to get its length
    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_token_length = prompt_tokens.shape[1]
    max_new_tokens = prompt_token_length + 500
    output = pipe(prompt, max_new_tokens=max_new_tokens)[0]['generated_text'].strip().split(prompt)[1].strip()

    results.append({
        "question": test_data[i]['instruction'],
        "output": output,
        "answer": test_data[i]['output']
    })
    print(output)
    print("#####################################")


with open(results_file_path, 'w') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)