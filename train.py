import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model
import wandb

base_model_path = "meta-llama/Llama-3.2-1B-Instruct"
epochs = 5
learning_rate = 5e-5
print("Loading base model from", base_model_path)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": int(os.getenv("LOCAL_RANK", 0))
                },  # Assign to correct GPU
    use_cache=False,  # Disable KV cache when using gradient checkpointing
)

# Enable gradient checkpointing for memory efficiency
base_model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(
    base_model_path, trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.model_max_length = 3072  # Match with max_seq_length
base_model.config.pad_token_id = base_model.config.eos_token_id
base_model.config.use_cache = False  # Ensure config is consistent

# Ensure dataset is properly tokenized


def preprocess_function(examples):
    return tokenizer(
        examples["line"],
        truncation=True,
        max_length=3072,
        padding="max_length",
        return_tensors=None,
    )


dataset = load_dataset("./data/",
                       data_files=["phish_mails_formatted.csv"])

# Preprocess the dataset
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)

peft_model = get_peft_model(base_model, lora_config)


def formatting_prompts_func(example):
    if isinstance(example['line'], list):
        return example['line'][0] if example['line'] else ""
    return example['line']


response_template = "<|start_header_id|>assistant<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
    mlm=False,
)

training_args = SFTConfig(
    max_seq_length=3072,
    output_dir="./model/lora_model",
    save_steps=250,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    logging_steps=5,
    learning_rate=learning_rate,
    num_train_epochs=epochs,
    # wandb
    report_to="none",
    run_name="llama_sft",
    # DeepSpeed configuration
    deepspeed="ds_config.json",
    local_rank=int(os.getenv("LOCAL_RANK", 0)),
    fp16=True,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    warmup_steps=100,
    weight_decay=0.01,
    # Distributed training settings
    ddp_find_unused_parameters=False,
    ddp_bucket_cap_mb=25,
    dataloader_pin_memory=False,  # Disable pinned memory for distributed training
    # Optimization settings
    optim="adamw_torch",
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
)

trainer = SFTTrainer(
    peft_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.train()

if trainer.is_world_process_zero():
    trainer.save_model("./model/lora_model")
