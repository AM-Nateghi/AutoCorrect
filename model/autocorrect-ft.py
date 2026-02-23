from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
from trl.trainer.sft_trainer import SFTTrainer
import torch
import huggingface_hub
import os

# Load environment variables from .env next to the project root (if present)
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

model_id = "CohereLabs/tiny-aya-earth"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

if os.environ.get("HF_TOKEN"):
    huggingface_hub.login(token=os.environ.get("HF_TOKEN"))

model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=2, dtype=torch.bfloat16, device_map="auto"
)

# Quantize / LoRa
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
)

model = get_peft_model(model, peft_config)

dataset = load_from_disk(dataset_path="./dataset")


def preprocess(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=512
    )


tokenized_dataset = dataset.map(preprocess, batched=True)

args = TrainingArguments(  # type: ignore[call-arg]
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,  # بسته به GPU
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,  # یا bf16
    save_strategy="steps",
    save_steps=100,
    logging_steps=100,
)


trainer = SFTTrainer(  # type: ignore[call-arg]  # یا Trainer معمولی
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,  # type: ignore[call-arg]
    peft_config=peft_config,
)

trainer.train()

trainer.save_model("./results/final-model")
tokenizer.save_pretrained("./results/final-model")
