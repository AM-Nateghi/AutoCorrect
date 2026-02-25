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

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TOKEN_LOG_PATH = LOG_DIR / "token_ids.log"

model_id = "CohereLabs/tiny-aya-earth"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"

MAX_SEQ_LEN = 724

SYSTEM_PROMPT = (
    "You are an automatic grader for short-answer questions. "
    "Given the question, the ideal true response, and a student's answer, "
    "decide if the student's answer should be marked correct. "
    "You must respond with exactly one token: YES (1) if it is correct, "
    "otherwise NO (0). Do not output any other words."
)

# بررسی و لاگ کردن توکن‌های YES و NO
yes_token_ids = tokenizer(" YES", add_special_tokens=False).input_ids
no_token_ids = tokenizer(" NO", add_special_tokens=False).input_ids

print(f"YES token ids: {yes_token_ids}")
print(f"NO token ids : {no_token_ids}")

with TOKEN_LOG_PATH.open("a", encoding="utf-8") as f:
    f.write(f"YES token ids: {yes_token_ids}\n")
    f.write(f"NO token ids : {no_token_ids}\n")

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


def build_messages(text: str):
    """
    متن خام (Question + True + Student) را به user message تبدیل می‌کند و
    system instruction را هم اضافه می‌کند تا مدل فقط یک توکن YES/NO پیش‌بینی کند.
    """
    user_content = (
        str(text)
        + "\n\n"
        + "Decide if the student's answer should be marked correct.\n"
        + "Answer with a single token: YES if it is correct, otherwise NO."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def preprocess(examples):
    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []

    texts = examples["text"]
    labels = examples["label"]

    for text, label in zip(texts, labels):
        messages = build_messages(text)

        chat_encoding = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )

        prompt_ids = chat_encoding["input_ids"]

        # پاسخ هدف، فقط یک توکن YES یا NO
        answer_str = " YES" if int(label) == 1 else " NO"
        answer_ids = tokenizer(answer_str, add_special_tokens=False)["input_ids"]

        ids = prompt_ids + answer_ids
        lbls = [-100] * len(prompt_ids) + answer_ids

        # برش از سمت چپ تا حداکثر طول مجاز
        if len(ids) > MAX_SEQ_LEN:
            ids = ids[-MAX_SEQ_LEN:]
            lbls = lbls[-MAX_SEQ_LEN:]

        att = [1] * len(ids)

        # پد کردن تا MAX_SEQ_LEN
        pad_len = MAX_SEQ_LEN - len(ids)
        if pad_len > 0:
            pad_id = tokenizer.pad_token_id
            ids = [pad_id] * pad_len + ids
            att = [0] * pad_len + att
            lbls = [-100] * pad_len + lbls

        input_ids_batch.append(ids)
        attention_mask_batch.append(att)
        labels_batch.append(lbls)

    return {
        "input_ids": input_ids_batch,
        "attention_mask": attention_mask_batch,
        "labels": labels_batch,
    }


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
