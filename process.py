"""
پردازش خط‌به‌خط (سوال به سوال) فایل results.jsonl:
- فقط پاسخ‌های با base_score >= 0.2 (قبل از هر پردازش)
- حذف پاسخ‌های بی‌ارزش (طول < 10 یا > 5000 کاراکتر)، تمیز کردن، نرمال با hazm
- پردازش موازی پاسخ‌ها (هر سوال به ترتیب؛ داخل هر سوال پاسخ‌ها موازی)، حداکثر 250 پاسخ معتبر
- حذف تگ‌های HTML و کاراکترهای خاص از سوال/پاسخ؛ نرمال سوال با hazm
- حذف سوالاتی که کمتر از 20 پاسخ معتبر دارند
خروجی: result.v2.jsonl | لاگ: process.py.log
"""

import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from hazm import Normalizer
from tqdm import tqdm

# ======================== Settings ========================
INPUT_FILE = "results.jsonl"
OUTPUT_FILE = "result.v2.jsonl"
LOG_FILE = "process.py.log"

MIN_ANSWER_LEN = 10
MAX_ANSWER_LEN = 5000
MIN_VALID_RESPONSES_PER_QUESTION = 20
MAX_RESPONSES_PER_QUESTION = 250
MIN_ACHIVED_SCORE_RATION = 0.2

# موازی: تعداد نخ برای پردازش همزمان پاسخ‌ها
N_WORKERS = 16
# حداکثر زمان برای پردازش یک پاسخ (ثانیه)
RESPONSE_PROCESS_TIMEOUT = 10

BUFFER_SIZE = 64 * 1024  # 64 KB برای خواندن/نوشتن

# ======================== Logging ========================
# هم به فایل هم به کنسول
log_format = "%(asctime)s | %(levelname)-7s | %(message)s"
date_fmt = "%Y-%m-%d %H:%M:%S"

file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(log_format, date_fmt))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(log_format, date_fmt))

logger = logging.getLogger("process")
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ======================== Helpers ========================
# یک نمونه Normalizer برای کل برنامه (فقط یک بار ساخته می‌شود)
NORMALIZER = Normalizer()
_normalizer_lock = threading.Lock()


# حذف تگ HTML؛ بین تگ‌ها فاصله بذار تا کلمات به هم نچسبند
RE_HTML_TAG = re.compile(r"<[^>]+>")

# کاراکترهای کنترلی و مخصوص ورد (مثل smart quotes، dash های خاص و ...)
# فاصله‌های چندتایی را هم یکی می‌کنیم
RE_MULTI_SPACE = re.compile(r"\s+")
# حذف کاراکترهای کنترلی (C0, C1) و برخی نمادهای خاص یونیکد
RE_CONTROL_AND_SPECIAL = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200d\ufeff]+"
)


def clean_question_text(raw: str) -> str:
    """تگ‌های HTML و کاراکترهای خاص را حذف می‌کند؛ فاصله بین کلمات حفظ می‌شود."""
    return clean_text_only(raw) if raw and isinstance(raw, str) else ""


def clean_text_only(raw: str) -> str:
    """فقط تمیز کردن (HTML، کاراکترهای کنترلی، فاصله‌ها) بدون hazm."""
    if not raw or not isinstance(raw, str):
        return ""
    text = RE_HTML_TAG.sub(" ", raw)
    text = RE_CONTROL_AND_SPECIAL.sub(" ", text)
    text = RE_MULTI_SPACE.sub(" ", text)
    return text.strip()


def normalize_text(text: str) -> str:
    """نرمال‌سازی با hazm (با قفل برای استفاده امن در چندنخی)."""
    if not text or not isinstance(text, str):
        return ""
    try:
        with _normalizer_lock:
            return NORMALIZER.normalize(text)
    except Exception:
        return text


def count_lines(filepath: str) -> int:
    """شمارش خطوط بدون بارگذاری کل فایل."""
    count = 0
    with open(filepath, "r", encoding="utf-8", buffering=BUFFER_SIZE) as f:
        for _ in f:
            count += 1
            if count % 500_000 == 0:
                logger.debug("Counted %s lines so far", f"{count:,}")
    return count


def iter_questions(filepath: str, show_progress: bool = True):
    """ژنراتور: خط به خط خواندن؛ هر خط = یک سوال (با پاسخ‌ها)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    total_lines = None
    if show_progress:
        logger.info("📊 Counting lines in %s ...", filepath)
        total_lines = count_lines(filepath)
        logger.info("📈 Total lines (questions): %s", f"{total_lines:,}")

    with open(filepath, "r", encoding="utf-8", buffering=BUFFER_SIZE) as f:
        iterator = f
        if show_progress and total_lines:
            iterator = tqdm(
                iterator,
                total=total_lines,
                desc="📥 Reading questions",
                unit="question",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )

        for line in iterator:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("JSON error: %s | Line preview: %s", e, line[:200])
                continue


def _process_one_response(r: dict) -> dict | None:
    """
    پردازش یک پاسخ در نخ جدا: چک طول، تمیز کردن، نرمال با hazm.
    فقط پاسخ‌هایی که base_score >= MIN_BASE_SCORE دارند اینجا صدا زده می‌شوند.
    """
    value = r.get("value")
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    length = len(value)
    if length < MIN_ANSWER_LEN or length > MAX_ANSWER_LEN:
        return None
    cleaned_ans = clean_text_only(value)
    if not cleaned_ans:
        return None
    normalized_value = normalize_text(cleaned_ans)
    if not normalized_value:
        return None
    return {
        "score": r.get("score"),
        "value": normalized_value,
        "base_score": r.get("base_score"),
    }


def process_one(question_obj: dict, executor: ThreadPoolExecutor) -> dict | None:
    """
    یک سوال را پردازش می‌کند:
    - فقط پاسخ‌های با base_score >= MIN_BASE_SCORE وارد فاز پردازش می‌شوند.
    - تمیز و نرمال سوال (یک بار، با hazm).
    - پردازش موازی پاسخ‌ها با timeout؛ حداکثر MAX_RESPONSES_PER_QUESTION پاسخ معتبر.
    - اگر کمتر از MIN_VALID_RESPONSES_PER_QUESTION پاسخ معتبر ماند، None.
    """

    # فیلتر اول: فقط پاسخ‌هایی با base_score >= 0.2 (قبل از هر چک/تمیز دیگر)
    def _base_score_ok(resp: dict) -> bool:
        bs = resp.get("base_score")
        sc = resp.get("score")
        if bs is None or sc is None:
            return False
        try:
            return float(float(sc) / float(bs)) >= MIN_ACHIVED_SCORE_RATION
        except (TypeError, ValueError):
            return False

    responses = question_obj.get("responses") or []
    candidates = [r for r in responses if _base_score_ok(r)]
    if not candidates:
        return None

    # سوال: تمیز + نرمال با hazm
    raw_text = question_obj.get("text") or ""
    cleaned_q = clean_question_text(raw_text)
    if not cleaned_q.strip():
        logger.debug("Empty question after clean: _id=%s", question_obj.get("_id"))
    question_text = normalize_text(cleaned_q)

    valid_responses = []
    need = MAX_RESPONSES_PER_QUESTION
    # همهٔ کاندیدها را به نخ‌ها می‌سپاریم؛ به محض 250 تا معتبر، بقیه را کنسل می‌کنیم
    futures = {executor.submit(_process_one_response, r): r for r in candidates}

    # حداکثر انتظار برای جمع‌آوری تا 250 پاسخ (هر کدام تا RESPONSE_PROCESS_TIMEOUT)
    overall_timeout = 60 + min(len(futures), need) * RESPONSE_PROCESS_TIMEOUT
    try:
        for fut in as_completed(futures, timeout=overall_timeout):
            if len(valid_responses) >= need:
                for f in futures:
                    f.cancel()
                break
            try:
                result = fut.result(timeout=RESPONSE_PROCESS_TIMEOUT)
                if result is not None:
                    valid_responses.append(result)
                    if len(valid_responses) >= need:
                        for f in futures:
                            f.cancel()
                        break
            except Exception:
                pass
    except Exception:
        for f in futures:
            f.cancel()

    if len(valid_responses) < MIN_VALID_RESPONSES_PER_QUESTION:
        return None

    # نهایتاً فقط 250 تا
    valid_responses = valid_responses[:MAX_RESPONSES_PER_QUESTION]

    return {
        "_id": question_obj.get("_id"),
        "text": question_text,
        "info": question_obj.get("info"),
        "score": question_obj.get("score"),
        "responses_count": len(valid_responses),
        "responses": valid_responses,
    }


def run(
    input_path: str = INPUT_FILE,
    output_path: str = OUTPUT_FILE,
    dry_run: bool = False,
):
    """خواندن خط‌به‌خط، پردازش، نوشتن سطر به سطر. dry_run فقط پردازش بدون نوشتن."""
    logger.info("═══════════════════════════════════════════════════════")
    logger.info("🚀 Process started at %s", datetime.now().strftime(date_fmt))
    logger.info("   Input:  %s", input_path)
    logger.info("   Output: %s", output_path if not dry_run else "(dry run, no output)")
    logger.info("   Min answer length: %s | Max: %s", MIN_ANSWER_LEN, MAX_ANSWER_LEN)
    logger.info(
        "   Min valid responses per question: %s", MIN_VALID_RESPONSES_PER_QUESTION
    )
    logger.info("   Max responses per question: %s", MAX_RESPONSES_PER_QUESTION)
    logger.info(
        "   Min achieved score ratio for responses: %s", MIN_ACHIVED_SCORE_RATION
    )
    logger.info(
        "   Parallel workers: %s | Timeout per response: %ss",
        N_WORKERS,
        RESPONSE_PROCESS_TIMEOUT,
    )
    logger.info("═══════════════════════════════════════════════════════")

    start_total = time.time()
    processed = 0
    written = 0
    skipped_quality = 0
    errors = 0
    total_read = 0

    out_file = None
    if output_path and not dry_run:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out_file = open(output_path, "w", encoding="utf-8", buffering=BUFFER_SIZE)
        logger.info("📂 Output file opened: %s", output_path)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        try:
            for question in iter_questions(input_path, show_progress=True):
                total_read += 1
                try:
                    result = process_one(question, executor)
                    processed += 1
                    if result is None:
                        skipped_quality += 1
                        logger.debug(
                            "Skipped (few valid answers): _id=%s", question.get("_id")
                        )
                        continue
                    if out_file:
                        out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                        written += 1
                except Exception as e:
                    errors += 1
                    logger.exception(
                        "Process error for _id=%s: %s", question.get("_id"), e
                    )
        finally:
            if out_file:
                out_file.close()
                logger.info("📂 Output file closed.")

    elapsed = time.time() - start_total
    summary = f"""
╔════════════════════════════════════════════════════════╗
║              📊 Process Summary (process.py)           ║
╠════════════════════════════════════════════════════════╣
║ 📥 Total lines read:        {total_read:>7} items    ║
║ ✅ Questions processed:     {processed:>7} items    ║
║ ❌ Skipped (low quality):    {skipped_quality:>7} items    ║
║ 💾 Written to output:       {written:>7} items    ║
║ ⚠️  Errors:                  {errors:>7} items    ║
║ ⏱️  Total time:              {elapsed:>7.1f}s        ║
║ 💾 Output file:             {output_path if not dry_run else "N/A":<14} ║
╚════════════════════════════════════════════════════════╝
"""
    logger.info(summary)
    # ذخیره خلاصه در همان فایل لاگ
    with open(LOG_FILE, "a", encoding="utf-8") as lf:
        lf.write(f"\nLast updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lf.write(summary)

    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process results.jsonl → result.v2.jsonl"
    )
    parser.add_argument("--input", "-i", default=INPUT_FILE, help="Input JSONL path")
    parser.add_argument("--output", "-o", default=OUTPUT_FILE, help="Output JSONL path")
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not write output file"
    )
    args = parser.parse_args()

    run(input_path=args.input, output_path=args.output, dry_run=args.dry_run)
