from pymongo import MongoClient
from tqdm import tqdm
import time
import json
import os
from pathlib import Path
from datetime import datetime

# ======================== Settings ========================
batch_size = 100
OUTPUT_FILE = "results.jsonl"
LOG_FILE = "log.txt"

# ======================== Database Connection ========================
print("🔗 Connecting to database...")
client = MongoClient("mongodb://User:8~ceWCKeXiuLHOak^O@77.238.110.250:57896/NateghiAI")
db = client["NateghiAI"]
questions_coll = db["Question"]
step_resp_coll = db["StepResponse"]
print("✅ Connection successful!\n")

# ======================== Fetching IDs ========================
print("📊 Fetching all question IDs...")
all_question_ids = [doc["_id"] for doc in questions_coll.find({}, {"_id": 1})]
batches = [
    all_question_ids[i : i + batch_size]
    for i in range(0, len(all_question_ids), batch_size)
]
print(f"📈 Total questions: {len(all_question_ids)}")
print(f"📦 Number of batches: {len(batches)}")
print(f"📝 Batch size: {batch_size}\n")

# ======================== Resume/Checkpoint System ========================
start_batch_idx = 0
processed_count = 0
resumed = False

if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
    print(f"📂 Found existing results file: {OUTPUT_FILE}")
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if lines:
            # Get last processed question
            last_line = json.loads(lines[-1])
            last_q_id = last_line.get("_id")
            processed_count = len(lines)

            # Find which batch this question belonged to
            for idx, batch in enumerate(batches):
                batch_q_ids = [str(q) for q in batch]
                if last_q_id in batch_q_ids:
                    position_in_batch = batch_q_ids.index(last_q_id)

                    # If batch is complete, start from next batch
                    if position_in_batch == len(batch) - 1:
                        start_batch_idx = idx + 1
                        print(
                            f"✅ Batch {idx + 1} was complete. Resuming from Batch {start_batch_idx + 1}"
                        )
                    else:
                        # Batch is incomplete, need to complete it
                        start_batch_idx = idx
                        # Remove incomplete batch results
                        incomplete_lines = lines[
                            : processed_count - (position_in_batch + 1)
                        ]
                        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                            f.writelines(incomplete_lines)
                        processed_count = len(incomplete_lines)
                        print(
                            f"⚠️  Batch {idx + 1} was incomplete. Removed {position_in_batch + 1} incomplete items."
                        )
                        print(
                            f"✅ Resuming from incomplete position in Batch {start_batch_idx + 1}"
                        )

                    resumed = True
                    break

            if resumed:
                print(f"📊 Already processed: {processed_count} items\n")
    except Exception as e:
        print(f"⚠️  Error reading checkpoint: {e}")
        print("Starting fresh...\n")
else:
    print("📝 Starting fresh - no checkpoint file found\n")

# Create file for appending (if it doesn't exist)
if not os.path.exists(OUTPUT_FILE):
    Path(OUTPUT_FILE).touch()

# ======================== Statistics Variables ========================
results = []
questions_with_responses = 0
questions_without_responses = 0
total_responses = 0
no_response_questions = []

# ======================== Processing Batches ========================
start_total = time.time()

# Calculate remaining items
remaining_batches = len(batches) - start_batch_idx
remaining_items = sum(len(batches[i]) for i in range(start_batch_idx, len(batches)))

print(
    f"🚀 Processing {remaining_batches} remaining batches ({remaining_items} questions)...\n"
)

with tqdm(
    total=remaining_items,
    desc="📥 Processing remaining questions",
    unit="question",
    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
) as overall_pbar:

    for batch_idx in range(start_batch_idx, len(batches)):
        batch = batches[batch_idx]
        batch_start = time.time()
        batch_responses_count = 0
        batch_with_resp = 0
        batch_without_resp = 0
        batch_results = []

        # Progress bar for batch
        with tqdm(
            total=len(batch),
            desc=f"  ├─ Batch {batch_idx + 1}/{len(batches)}",
            unit="item",
            leave=False,
            bar_format="{desc}: {percentage:3.0f}%|{bar}|",
        ) as batch_pbar:

            for q_id in batch:
                start_time = time.time()

                # Find responses
                responses = []
                cursor = step_resp_coll.find(
                    {"Exam.ExamResponse.QuestionID": str(q_id)},
                    {"Exam.ExamResponse": 1},
                )

                for doc in cursor:
                    ex_resps = [
                        resp
                        for resp in doc.get("Exam", {}).get("ExamResponse", [])
                        if resp["QuestionID"] == str(q_id)
                    ]
                    for ex_resp in ex_resps:
                        score = ex_resp.get("Score")
                        base_score = ex_resp.get("BaseScore")
                        value = (
                            ex_resp["Response"][0]["Value"]
                            if "Response" in ex_resp
                            and isinstance(ex_resp["Response"], list)
                            and len(ex_resp["Response"]) > 0
                            and "Value" in ex_resp["Response"][0]
                            and ex_resp["Response"][0]["Value"]
                            else None
                        )
                        try:
                            responses.append(
                                {
                                    "score": score,
                                    "value": value,
                                    "base_score": base_score,
                                }
                            )
                        except ValueError:
                            print(f"⚠️  Non-integer score for question {q_id}: {score}")
                        batch_responses_count += 1

                # Save condition: only if response exists
                if responses:
                    question_doc = questions_coll.find_one(
                        {"_id": q_id}, {"Text": 1, "Info": 1, "Score": 1}
                    )
                    result = {
                        "_id": str(q_id),
                        "text": question_doc.get("Text"),
                        "info": question_doc.get("Info"),
                        "score": question_doc.get("Score"),
                        "responses_count": len(responses),
                        "responses": responses,
                    }
                    batch_results.append(result)
                    questions_with_responses += 1
                    batch_with_resp += 1
                    total_responses += len(responses)
                else:
                    questions_without_responses += 1
                    batch_without_resp += 1
                    no_response_questions.append(str(q_id))

                elapsed = time.time() - start_time
                batch_pbar.set_postfix(
                    {"time": f"{elapsed:.2f}s", "responses": batch_responses_count}
                )
                batch_pbar.update(1)

        # ======================== Append batch results to file ========================
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for result in batch_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        batch_elapsed = time.time() - batch_start
        status_msg = f"✅ Batch {batch_idx + 1} completed: {batch_with_resp} questions with responses, {batch_without_resp} without responses ({batch_elapsed:.1f}s)"
        print(f"  └─ {status_msg}")

        overall_pbar.update(len(batch))

# ======================== Saving Statistics Log ========================
total_elapsed = time.time() - start_total

# Read final file statistics
final_file_count = 0
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        final_file_count = sum(1 for line in f)

summary = f"""
╔════════════════════════════════════════════════════════╗
║                 📊 Final Results Summary                ║
╠════════════════════════════════════════════════════════╣
║ ✅ Questions with responses:   {questions_with_responses:>7} items    ║
║ ❌ Questions without responses: {questions_without_responses:>7} items    ║
║ 📈 Total responses saved:      {total_responses:>7} items    ║
║ ⏱️  Total time (this run):     {total_elapsed:>7.1f}s        ║
║ 📊 Total saved in file:        {final_file_count:>7} items    ║
║ 💾 Results saved to:          {OUTPUT_FILE:<14} ║
╚════════════════════════════════════════════════════════╝

📋 Checkpoint Info:
   ├─ Batches processed: {len(batches) - start_batch_idx}/{len(batches)}
   ├─ Progress: {final_file_count}/{len(all_question_ids)} questions
   └─ Status: {"✅ Complete!" if final_file_count == len(all_question_ids) else "⏸️  Can resume if interrupted"}
"""
print(summary)

# Save checkpoint log
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(summary)
    if no_response_questions:
        f.write(
            f"\n📋 Questions without responses ({len(no_response_questions)} items):\n"
        )
        for q in no_response_questions[:100]:  # Show first 100
            f.write(f"  - {q}\n")
        if len(no_response_questions) > 100:
            f.write(f"  ... and {len(no_response_questions) - 100} more\n")
