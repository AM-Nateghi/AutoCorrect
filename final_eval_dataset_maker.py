import json
import os
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from dataset_maker import (
    CORRECTNESS_THRESHOLD,
    build_reference_answer,
    create_client,
    iter_questions,
    logger,
    select_responses_for_question,
)


# ======================== Settings ========================

INPUT_FILE = "result.v2.jsonl"
OUTPUT_FILE = "qa_eval_dataset.jsonl"

# Threshold for turning a score ratio into a binary label.
LABEL_THRESHOLD = 0.6


# ======================== Helpers ========================
def pick_three_responses_for_question(
    responses: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    From all responses of a question (already containing 'ratio' and 'value'),
    pick exactly three distinct responses with the following intent:

    - One with (near) full score (max ratio)
    - One around ratio ~= 0.6
    - One with ratio < 0.6

    If we cannot satisfy the constraints (e.g. no response with ratio < 0.6),
    return an empty list so that the caller can skip this question.
    """
    if not responses:
        return []

    # Ensure we only look at responses that actually have a numeric ratio.
    valid = [r for r in responses if isinstance(r.get("ratio"), (int, float))]
    if not valid:
        return []

    # Index-based work to ensure distinct selections.
    ratios = [float(r["ratio"]) for r in valid]

    # 1) Full score (or closest to full): maximum ratio.
    full_idx = max(range(len(ratios)), key=lambda i: ratios[i])

    # 2) One with ratio < 0.6 (اینجا عمداً سعی می‌کنیم تا حد ممکن
    #    از ۰.۶ دور شویم و به سمت ۰ برویم، پس کمترین نسبت را می‌گیریم).
    below_candidates: List[Tuple[int, float]] = [
        (i, ratios[i]) for i in range(len(ratios)) if ratios[i] < LABEL_THRESHOLD
    ]
    if not below_candidates:
        # بدون پاسخ زیر ۰.۶ نمی‌توانیم سه نمونه مطابق خواسته بسازیم.
        return []
    # Pick the one with the lowest ratio among those below threshold (دورترین از ۰.۶ به سمت ۰).
    low_idx = min(below_candidates, key=lambda t: t[1])[0]

    # 3) One around 0.6 (closest to 0.6 in absolute distance).
    # Make sure it's distinct from full_idx and low_idx.
    candidates_for_mid = [i for i in range(len(ratios)) if i not in {full_idx, low_idx}]
    if not candidates_for_mid:
        return []

    mid_idx = min(
        candidates_for_mid,
        key=lambda i: abs(ratios[i] - LABEL_THRESHOLD),
    )

    selected_indices = {full_idx, mid_idx, low_idx}
    if len(selected_indices) < 3:
        # در صورتی که به هر دلیل نتوانستیم سه پاسخ متمایز انتخاب کنیم، این سؤال را رها می‌کنیم.
        return []

    return [valid[i] for i in (full_idx, mid_idx, low_idx)]


def build_eval_prompt(
    question_text: str,
    true_answer: str,
    student_answer: str,
) -> str:
    """
    Build the evaluation prompt text, following the same structure as
    training prompts in prepare_dataset_for_ft.py but **بدون** توکن‌های YES/NO.
    """
    return (
        f"Question: {question_text}\n"
        f"True response: {true_answer}\n"
        f"Student answer: {student_answer}\n"
        f"Student's answer is correct with True response? YES/NO\n"
        f"Model: "
    )


def ratio_to_label(ratio: float) -> int:
    """
    Map a score ratio to binary label:
    - 1 if ratio > 0.6
    - 0 otherwise (<= 0.6)
    """
    return 1 if ratio > LABEL_THRESHOLD else 0


# ======================== Main pipeline ========================


def build_final_eval_dataset(
    input_path: str = INPUT_FILE,
    output_path: str = OUTPUT_FILE,
) -> int:
    """
    Build the final distributed evaluation dataset directly from result.v2.jsonl.

    For each question:
      - Use Gemma 3-12B (از طریق build_reference_answer) to construct a canonical reference answer.
      - Select 3 student responses with different score levels:
          * one with (near) full score
          * one around 0.6
          * one below 0.6 (تا جای ممکن دور از ۰.۶ و نزدیک به ۰)
      - For each selected response, write one JSON line with:
          * text: the evaluation prompt (بدون YES/NO و برچسب داخل متن)
          * label: 1 if ratio > 0.6, else 0

    Returns:
        Number of rows written.
    """
    client = create_client()
    logger.info(
        "Starting final eval dataset build | input=%s | output=%s",
        input_path,
        output_path,
    )

    # Ensure output directory exists (در صورت استفاده از مسیرهای تو در تو)
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    total_questions_read = 0
    total_questions_used = 0
    total_rows = 0
    skipped_no_valid_responses = 0
    skipped_no_correct_for_ref = 0
    skipped_model_failure = 0
    skipped_not_enough_levels = 0

    # برای tqdm، تعداد کل سؤال‌ها را از روی فایل ورودی حساب می‌کنیم
    try:
        with open(input_path, "r", encoding="utf-8") as f_in:
            total_in_file = sum(1 for _ in f_in)
    except FileNotFoundError:
        logger.error("Input file not found for final eval dataset: %s", input_path)
        return 0

    with open(output_path, "w", encoding="utf-8") as jsonl_file, tqdm(
        total=total_in_file,
        desc="📦 Building final eval dataset",
        unit="question",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ) as pbar:
        for question in iter_questions(input_path):
            total_questions_read += 1
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

            # Use "correct" responses (طبق threshold دیتاست اصلی) to build reference answer.
            correct_answers = [r["value"] for r in responses if r.get("is_correct")]
            if not correct_answers:
                skipped_no_correct_for_ref += 1
                logger.info(
                    "Skipping question _id=%s: no responses above correctness threshold (%.2f)",
                    q_id,
                    CORRECTNESS_THRESHOLD,
                )
                pbar.update(1)
                continue

            true_answer = build_reference_answer(
                client=client,
                question_text=question_text,
                info=info,
                correct_answers=correct_answers[:20],
            )
            if not true_answer:
                skipped_model_failure += 1
                logger.warning(
                    "Skipping question _id=%s: failed to build reference answer",
                    q_id,
                )
                pbar.update(1)
                continue

            picked = pick_three_responses_for_question(responses)
            if len(picked) != 3:
                skipped_not_enough_levels += 1
                logger.info(
                    "Skipping question _id=%s: could not pick 3 distinct responses with required score levels",
                    q_id,
                )
                pbar.update(1)
                continue

            total_questions_used += 1

            # لاگ نسبت‌های انتخاب‌شده برای ردیابی دقیق
            ratios_selected = [float(r["ratio"]) for r in picked]
            logger.debug(
                "Question _id=%s | picked ratios=%s",
                q_id,
                ratios_selected,
            )

            for r in picked:
                ratio = float(r["ratio"])
                student_answer = str(r["value"])
                text = build_eval_prompt(
                    question_text=question_text,
                    true_answer=true_answer,
                    student_answer=student_answer,
                )
                label = ratio_to_label(ratio)

                record = {
                    "text": text,
                    "label": label,
                    "ratio": ratio,
                    "question_id": q_id,
                }
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_rows += 1

            pbar.update(1)

    summary = (
        "Final eval dataset created\n"
        f"  output: {output_path}\n"
        f"  rows: {total_rows}\n"
        f"  questions_read: {total_questions_read}\n"
        f"  questions_used: {total_questions_used}\n"
        f"  skipped_no_valid_responses: {skipped_no_valid_responses}\n"
        f"  skipped_no_correct_for_ref: {skipped_no_correct_for_ref}\n"
        f"  skipped_model_failure: {skipped_model_failure}\n"
        f"  skipped_not_enough_levels: {skipped_not_enough_levels}"
    )
    logger.info(summary)

    return total_rows


if __name__ == "__main__":
    build_final_eval_dataset()
