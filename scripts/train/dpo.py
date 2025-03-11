import torch
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import os
import pdb
from dataclasses import dataclass, field
from typing import Dict, Optional
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from videollava.model import *

os.environ['WANDB_MODE'] = "offline"
model_path = "checkpoints/homiebot-7b-sft"
cache_dir = "cache_dir"
output_dir="checkpoints/homiebot-7b-dpo-lora"
learning_rate=6e-6
r=8

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
    
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right", use_fast=False)
model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, quantization_config=bnb_config, device_map="auto")
model_ref = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, quantization_config=bnb_config, device_map="auto")

tokenizer.pad_token = tokenizer.unk_token

sys = """You are a powerful housework assistant, I will give you following information for you to make a decision toward the final task.
(1) Observation images: Four first-person perspective images of the current environment, in the order of front, left, back, and right.
(2) Task: Your final goal.
(3) Inventory: Your current assets, remember that you are a one-hand agent, which means you can't open or pick when your Inventory is not None, and you can't put if your Inventory is None, this is very important.
(4) Historical Execution: Subtasks that were already fulfilled in the history, and the execution status of each subtask(success or fail). You need to make decisions based on historical actions, current circumstances and your final task.
(5) Feedback: Feedback will provide error information of the last execution, it will be None if the last execution ends successfully.

You should output with following formats:
Subtask: [action, target], choose your action from the action list [Go to, Pick, Put, Open, Close, End], and the target can be a place or a object from your observation. If you choose Put as your action, output in format [Put, object, place] which means put the object to the place. If the final task is done and no more action is needed, just output [End].
Model: Choose one most suitable model in the model list [NoMaD, PixNav, octo, RT-1-X]. NoMaD can go to a spot like living room, PixNav focuses on object navigation and can go to a object, octo can handle with open and close, RT-1-X is good at picking and putting.

You need to focus on the consistency with previous subtasks. You should pay attention to current Inventory and avoid conflicts.
Remember you can only go to the place and interact with the objects you observe in your sight.
Remember the logic between outputs, it is recommended to open the receptacle before you pick something because you can't open while holding, and it's recommended to arrive the object place before you interact with it.
Remember you just need to output the next subtask to be fulfilled and don't output a whole plan, this is very important. Remember you should output strictly with the response template.
Now, I will send the message so that you can make planning accordingly.\n"""

def dataset_process(samples):
    return {
        "prompt": [
            f"{sys} USER: \n{prompt}\nASSISTANT: "
            for prompt in samples["prompt"]
        ],
        "chosen": samples["chosen"],
        "rejected": samples["rejected"],
    }

dataset = load_dataset(
        "json", 
        data_files="/data1/tct_data/Video-LLaVA/HomieBot/dpo_train.json", 
        split=["train[:90%]", "train[90%:]"]
    )

train_dataset = dataset[0].map(
        dataset_process,
        batched=True,
        remove_columns=dataset[0].column_names
    )
eval_dataset = dataset[1].map(
        dataset_process,
        batched=True,
        remove_columns=dataset[1].column_names
    )

training_args = DPOConfig(
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    num_train_epochs=1, 
    save_steps=1000,
    learning_rate=learning_rate,
    bf16=True,
    save_total_limit=1,
    logging_steps=10,
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    remove_unused_columns=False
)

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

peft_config = LoraConfig(
    r=r,
    lora_alpha=r,
    target_modules=find_all_linear_names(model),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_prompt_length=2048,
    max_length=2048,
)

dpo_trainer.train()

dpo_trainer.save_model(output_dir)
