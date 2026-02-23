import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm


# ======================== Settings (adjust as needed) ========================

# Input and output paths (relative to project root by default)
INPUT_FILE = "result.v2.jsonl"
OUTPUT_FILE = "qa_dataset.csv"

# Maximum number of answers per question to include in the dataset
MAX_RESPONSES_PER_QUESTION = 30

# Score ratio threshold for considering an answer "correct"
# ratio = score / base_score
CORRECTNESS_THRESHOLD = 0.7

# Maximum number of questions to process
MAX_QUESTIONS = 50

# Google GenAI configuration
MODEL_NAME = "gemma-3-27b-it"
API_KEY_ENV_VAR = "GAS_API_KEY"

# Logging configuration
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "dataset_maker.py.log")


# ======================== Logging ========================

Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# Load environment variables from .env next to the project root (if present)
PROJECT_ROOT = Path(__file__).resolve().parent
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

log_format = "%(asctime)s | %(levelname)-7s | %(message)s"
date_fmt = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("dataset_maker")
logger.setLevel(logging.INFO)
logger.handlers.clear()

file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(log_format, date_fmt))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(log_format, date_fmt))

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ======================== Core helpers ========================

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
            except json.JSONDecodeError as e:
                logger.warning("JSON decode error: %s | line preview: %s", e, line[:200])
                continue


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_ratio(score: Any, base_score: Any) -> Optional[float]:
    """Compute score/base_score, returning None when invalid."""
    s = _safe_float(score)
    b = _safe_float(base_score)
    if s is None or b is None or b == 0:
        return None
    return s / b


def select_responses_for_question(
    question: Dict[str, Any],
) -> Tuple[str, Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Prepare data for one question.

    Returns:
        question_text: cleaned question text (as-is from result.v2.jsonl)
        info: the optional info/metadata dict
        responses: list of dicts with keys:
            - value: student answer text
            - ratio: float score/base_score
            - is_correct: bool based on CORRECTNESS_THRESHOLD
    """
    question_text = (question.get("text") or "").strip()
    info = question.get("info")

    raw_responses = question.get("responses") or []
    prepared: List[Dict[str, Any]] = []

    for r in raw_responses:
        value = r.get("value")
        if value is None:
            continue
        ratio = compute_ratio(r.get("score"), r.get("base_score"))
        # Ignore responses without a valid ratio (including missing base_score)
        if ratio is None:
            continue
        value_str = str(value).strip()
        if not value_str:
            continue
        prepared.append(
            {
                "value": value_str,
                "ratio": ratio,
                "is_correct": ratio >= CORRECTNESS_THRESHOLD,
            }
        )

    if not prepared:
        return question_text, info, []

    # Sort by ratio descending so we always pick the highest-quality answers first
    prepared.sort(key=lambda x: x["ratio"], reverse=True)
    # Limit the number of responses considered for this question
    prepared = prepared[:MAX_RESPONSES_PER_QUESTION]

    return question_text, info, prepared


# ======================== Google GenAI (reference answer) ========================

def build_reference_answer(
    client: genai.Client,
    question_text: str,
    info: Optional[Dict[str, Any]],
    correct_answers: List[str],
) -> Optional[str]:
    """
    Use Google GenAI (Gemma 3 27B) to synthesize a single reference/ideal answer
    based on the question text, metadata (info), and high-scoring student answers.

    Returns the reference answer string, or None on failure.
    Retries the model call once if an exception occurs.
    """
    if not question_text or not correct_answers:
        return None

    info_json = ""
    if info is not None:
        try:
            info_json = json.dumps(info, ensure_ascii=False, indent=2)
        except TypeError:
            # Fall back to string representation
            info_json = str(info)

    # Build a prompt in English explaining the task, but keep the actual content in Persian.
    # The model should output structured JSON with a single "reference_answer" field.
    answers_block = "\n".join(
        f"{idx + 1}. {ans}" for idx, ans in enumerate(correct_answers)
    )

    prompt = f"""
We are building a question answering dataset for short-answer exam questions.

The question text and student answers are in Persian (Farsi).

Your task:
- Read the exam question, its metadata, and a set of high-scoring student answers.
- Synthesize ONE canonical Persian reference answer that would receive full credit.
- The answer must be concise, coherent, and directly address the question.
- Do NOT mention students, scores, grading, or that this is an aggregation.
- Do NOT number or bullet the answer; return a single plain-text answer.

Return your result as JSON with this exact schema:
{{
  "reference_answer": "string, the ideal Persian reference answer"
}}

Question (Persian):
{question_text}

Question metadata (may be empty, JSON):
{info_json}

High-scoring student answers (Persian):
{answers_block}
""".strip()

    # Try up to 2 times if we hit an exception from the model.
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "reference_answer": {"type": "STRING"},
                        },
                        "required": ["reference_answer"],
                    },
                ),
            )

            # With structured output, response.parsed should already contain a dict-like object.
            parsed = None
            if hasattr(response, "parsed") and response.parsed is not None:
                parsed = response.parsed  # type: ignore[assignment]
            else:
                text = getattr(response, "text", None)
                if not text:
                    logger.warning("Empty response from model for reference answer.")
                    return None
                parsed = json.loads(text)

            # Extract the reference_answer field from the parsed object.
            if isinstance(parsed, dict):
                ref = parsed.get("reference_answer")
            else:
                # For Pydantic models or other objects, fall back to attribute access.
                ref = getattr(parsed, "reference_answer", None)

            if not isinstance(ref, str):
                logger.warning("Model response missing 'reference_answer' string field.")
                return None
            ref = ref.strip()
            if not ref:
                return None
            return ref
        except Exception as e:
            logger.exception(
                "Error generating reference answer (attempt %s/2): %s", attempt + 1, e
            )
            if attempt == 1:
                return None


def create_client() -> genai.Client:
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(
            f"API key environment variable '{API_KEY_ENV_VAR}' is not set."
        )
    return genai.Client(api_key=api_key)


# ======================== Main pipeline ========================

def build_dataset(
    input_path: str = INPUT_FILE,
    output_path: str = OUTPUT_FILE,
) -> int:
    """
    Main entry point.

    Reads questions line-by-line from input_path (JSONL),
    calls Google GenAI once per question to construct a reference answer,
    and writes rows to a CSV file with columns:
        question_text, true_answer, student_answer, is_correct, correction_ratio

    Returns:
        Number of dataset rows written.
    """
    client = create_client()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Stats
    total_rows = 0
    processed_questions = 0
    total_read_questions = 0
    skipped_no_valid_responses = 0
    skipped_no_correct_for_ref = 0
    skipped_model_failure = 0

    # Count total questions for better progress bar (capped by MAX_QUESTIONS)
    try:
        with open(input_path, "r", encoding="utf-8") as f_in:
            total_in_file = sum(1 for _ in f_in)
    except FileNotFoundError:
        logger.error("Input file not found: %s", input_path)
        return 0

    total_for_progress = min(total_in_file, MAX_QUESTIONS)

    with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "question_text",
                "true_answer",
                "student_answer",
                "is_correct",
                "correction_ratio",
            ]
        )

        with tqdm(
            total=total_for_progress,
            desc="📦 Building QA dataset",
            unit="question",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            for question in iter_questions(input_path):
                if processed_questions >= MAX_QUESTIONS:
                    break

                total_read_questions += 1
                q_id = question.get("_id")

                question_text, info, responses = select_responses_for_question(question)
                if not question_text or not responses:
                    skipped_no_valid_responses += 1
                    logger.info(
                        "Skipping question _id=%s: no valid responses after ratio filter",
                        q_id,
                    )
                    pbar.update(1)
                    continue

                # Only use responses that pass the correctness threshold to build the reference answer
                correct_answers = [r["value"] for r in responses if r["is_correct"]]
                if not correct_answers:
                    skipped_no_correct_for_ref += 1
                    logger.info(
                        "Skipping question _id=%s: no responses above correctness threshold",
                        q_id,
                    )
                    pbar.update(1)
                    continue

                true_answer = build_reference_answer(
                    client=client,
                    question_text=question_text,
                    info=info,
                    correct_answers=correct_answers,
                )
                if not true_answer:
                    skipped_model_failure += 1
                    logger.warning(
                        "Skipping question _id=%s: failed to build reference answer",
                        q_id,
                    )
                    pbar.update(1)
                    continue

                processed_questions += 1

                for r in responses:
                    writer.writerow(
                        [
                            question_text,
                            true_answer,
                            r["value"],
                            bool(r["is_correct"]),
                            float(r["ratio"]),
                        ]
                    )
                    total_rows += 1

                pbar.update(1)

    summary = f"""
╔════════════════════════════════════════════════════════╗
║              📊 Dataset Maker Summary                  ║
╠════════════════════════════════════════════════════════╣
║ 📥 Total questions read:         {total_read_questions:>7} items    ║
║ ✅ Questions used (with ref):    {processed_questions:>7} items    ║
║ ❌ Skipped (no valid responses): {skipped_no_valid_responses:>7} items    ║
║ ❌ Skipped (no correct for ref): {skipped_no_correct_for_ref:>7} items    ║
║ ⚠️  Skipped (model failures):     {skipped_model_failure:>7} items    ║
║ 💾 Rows written to CSV:          {total_rows:>7} items    ║
║ 💾 Output CSV file:             {output_path:<14} ║
╚════════════════════════════════════════════════════════╝
"""
    logger.info(summary)

    # Append summary to log file with timestamp
    with open(LOG_FILE, "a", encoding="utf-8") as lf:
        lf.write(f"\nLast updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lf.write(summary)

    return total_rows


if __name__ == "__main__":
    # Simple CLI entry-point (no argument parsing; adjust variables at top).
    build_dataset()

