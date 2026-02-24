from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig
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
    print("logged in!")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

dataset = load_from_disk(dataset_path="./dataset")


def preprocess(examples):
    encodings = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    # برای مدل causal، برچسب‌ها همان توکن‌های ورودی هستند
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings


tokenized_dataset = dataset.map(preprocess, batched=True)  # type: ignore[assignment]

args = TrainingArguments(  # type: ignore[call-arg]
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=4,  # بسته به GPU
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    eval_strategy="epoch",
    per_device_eval_batch_size=4,
    save_strategy="steps",
    save_steps=100,
    logging_steps=100,
)


trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],  # type: ignore[index]
    eval_dataset=tokenized_dataset["test"],  # type: ignore[index]
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()

eval_metrics = trainer.evaluate(tokenized_dataset["test"])  # type: ignore[index]
print("Evaluation metrics on test set:", eval_metrics)

trainer.save_model("./results/final-model")
tokenizer.save_pretrained("./results/final-model")
