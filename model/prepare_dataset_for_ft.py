from typing import cast

import pandas as pd
from datasets import Dataset

# 1. CSV رو لود کن
df = pd.read_csv("../qa_dataset_old.csv")  # مسیر فایل CSV خودت


# 2. ستون text (prompt) بساز
def create_prompt(row):
    return (
        f"Question: {row['question_text']}\n"
        f"True response: {row['true_answer']}\n"
        f"Student answer: {row['student_answer']}\n"
        f"Student's answer is correct with True response? YES/NO\n"
        f"Model: "
    )


df["text"] = df.apply(create_prompt, axis=1)

# 3. label بساز (binary: 1 برای True, 0 برای False)
df["label"] = df["is_correct"].astype(int)

# 4. فقط ستون‌های لازم رو نگه دار
df = df[["text", "label"]]

# 5. تبدیل به Dataset
dataset = Dataset.from_pandas(df=cast(pd.DataFrame, df))

# 6. (اختیاری) split به train/test
dataset = dataset.train_test_split(test_size=0.1)  # 90% train, 10% test

print(dataset)

dataset.save_to_disk(".\\dataset")
