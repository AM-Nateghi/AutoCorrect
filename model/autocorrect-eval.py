import json
import os
from typing import Any, Dict, Iterable
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

MAX_QUESTIONS: int = 100
base_model_id = "CohereLabs/tiny-aya-earth"
adapter_path = "./results/final-model"
dataset_path = "../qa_eval_dataset.jsonl"
DEBUG_LOG_PATH = "../logs/autocorrect_eval_debug.jsonl"
TOKEN_LOG_PATH = "../logs/token_ids_eval.log"

device = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = (
    "You are an automatic grader for short-answer questions. "
    "Given the question, the ideal true response, and a student's answer, "
    "decide if the student's answer should be marked correct. "
    "You must respond with exactly one token: YES (1) if it is correct, "
    "otherwise NO (0). Do not output any other words."
)

MAX_SEQ_LEN = 724
THRESHOLD = 0.6

# tokenizer را از همون دایرکتوری نهایی که ذخیره کردی بگیر
tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"

# مدل پایه
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float16,
    device_map="auto" if device == "cuda" else None,
)

# لود کردن LoRA
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# بررسی و لاگ کردن توکن‌های YES و NO
yes_token_ids = tokenizer(" YES", add_special_tokens=False).input_ids
no_token_ids = tokenizer(" NO", add_special_tokens=False).input_ids

print(f"[EVAL] YES token ids: {yes_token_ids}")
print(f"[EVAL] NO token ids : {no_token_ids}")

os.makedirs(os.path.dirname(TOKEN_LOG_PATH), exist_ok=True)
with open(TOKEN_LOG_PATH, "a", encoding="utf-8") as f_tok:
    f_tok.write(f"YES token ids: {yes_token_ids}\n")
    f_tok.write(f"NO  token ids: {no_token_ids}\n")

# Count total questions for better progress bar (capped by MAX_QUESTIONS)
try:
    with open(dataset_path, "r", encoding="utf-8") as f_in:
        total_in_file = sum(1 for _ in f_in)
except FileNotFoundError:
    print("File not Found")


def iter_questions(filepath: str) -> Iterable[Dict[str, Any]]:
    """
    Stream the JSONL file line by line; each line is one question object.
    Does not load the whole file into memory.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def build_messages(text: str):
    """
    متن دیتاست eval را (که شامل Question/True/Student است) به user message تبدیل می‌کند
    و system instruction را اضافه می‌کند تا مدل فقط یک توکن YES/NO پیش‌بینی کند.
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


def predict_is_correct(prompt: str):
    """
    به‌جای generate، از logits آخرین توکن استفاده می‌کنیم و
    توزیع احتمال روی YES/NO را حساب می‌کنیم.
    """
    messages = build_messages(prompt)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # logits توکن بعدی

    # logits مربوط به YES/NO
    yes_id = yes_token_ids[0]
    no_id = no_token_ids[0]
    sub_logits = torch.stack([logits[yes_id], logits[no_id]])
    probs = torch.softmax(sub_logits, dim=-1)

    p_yes = float(probs[0])
    p_no = float(probs[1])

    # اگر اطمینان کافی نباشد، به‌عنوان نامشخص برگردان
    max_p = max(p_yes, p_no)
    if max_p < THRESHOLD:
        return None, {"p_yes": p_yes, "p_no": p_no}

    pred_label = 1 if p_yes >= p_no else 0
    return pred_label, {"p_yes": p_yes, "p_no": p_no}


# Stats
processed_questions = 0
success_predict_count = 0
fail_predict_count = 0
unknown_predict_count = 0

# Confusion matrix counters (فقط وقتی مدل YES/NO واضح می‌دهد)
tp = tn = fp = fn = 0

# اطمینان از وجود دایرکتوری لاگ
log_dir = os.path.dirname(DEBUG_LOG_PATH) or "."
os.makedirs(log_dir, exist_ok=True)

# فایل لاگ دقیق برای هر نمونه
log_f = open(DEBUG_LOG_PATH, "w", encoding="utf-8")

with tqdm(
    total=min(MAX_QUESTIONS, total_in_file),
    desc="🧪 Evaluating Model",
    unit="question",
    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
) as pbar:
    for question in iter_questions(dataset_path):
        if processed_questions >= MAX_QUESTIONS:
            break

        text: str = str(question.get("text", ""))
        label: int = int(question.get("label", -1))
        ratio: float = float(question.get("ratio", -1))

        _l, _t = predict_is_correct(text)

        # لاگ ردیف به‌صورت JSONL
        log_record = {
            "index": processed_questions,
            "text": text,
            "label": label,
            "ratio": ratio,
            "pred_label": _l,
            "pred_info": _t,
        }
        log_f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

        # چند نمونه‌ی اول را روی ترمینال هم چاپ کن
        if processed_questions < 5:
            preview_text = text[:120].replace("\n", " ")
            print("\n---- DEBUG SAMPLE ----")
            print(f"idx      : {processed_questions}")
            print(f"label    : {label}")
            print(f"pred_lbl : {_l}")
            print(f"pred_raw : {repr(_t)}")
            print(f"text     : {preview_text}")
            print("----------------------")

        if _l is None:
            unknown_predict_count += 1
            fail_predict_count += 1
        else:
            if _l == label:
                success_predict_count += 1
            else:
                fail_predict_count += 1

            # Confusion matrix (label: 1=YES درست است، 0=NO درست است)
            if label == 1 and _l == 1:
                tp += 1
            elif label == 0 and _l == 0:
                tn += 1
            elif label == 0 and _l == 1:
                fp += 1
            elif label == 1 and _l == 0:
                fn += 1

        processed_questions += 1
        pbar.update(1)

log_f.close()

# ---- Summary report ----
print("\n========== EVALUATION SUMMARY ==========")
print(f"Total questions evaluated       : {processed_questions}")
print(f"Correct predictions (YES/NO ok) : {success_predict_count}")
print(f"Incorrect predictions           : {fail_predict_count}")
print(f"Unclear (non-YES/NO) outputs    : {unknown_predict_count}")

if processed_questions > 0:
    accuracy = success_predict_count / processed_questions
    print(f"Overall accuracy                : {accuracy * 100:.2f}%")

    evaluated_with_clear_pred = processed_questions - unknown_predict_count
    if evaluated_with_clear_pred > 0:
        clear_acc = (tp + tn) / evaluated_with_clear_pred
        print(
            f"Accuracy on clear YES/NO only   : {clear_acc * 100:.2f}% "
            f"(n={evaluated_with_clear_pred})"
        )

    # Precision / recall برای کلاس YES (label=1)
    if (tp + fp) > 0:
        precision_yes = tp / (tp + fp)
        print(f"Precision (YES)                : {precision_yes * 100:.2f}%")
    if (tp + fn) > 0:
        recall_yes = tp / (tp + fn)
        print(f"Recall (YES)                   : {recall_yes * 100:.2f}%")

    # Precision / recall برای کلاس NO (label=0)
    if (tn + fn) > 0:
        precision_no = tn / (tn + fn)
        print(f"Precision (NO)                 : {precision_no * 100:.2f}%")
    if (tn + fp) > 0:
        recall_no = tn / (tn + fp)
        print(f"Recall (NO)                    : {recall_no * 100:.2f}%")

print("========================================")
