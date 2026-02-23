from sentence_transformers import SentenceTransformer, util
import hazm

model = SentenceTransformer("PartAI/Tooka-SBERT-V2-Large")

normalizer = hazm.Normalizer()


def normalize_text(text):
    return normalizer.normalize(text)


# مثال استفاده
answer_correct = "اون فرد خودش رو با بازیگر مورد علاقه اش متحد فرض کرده برای همین از موفقیت اونو موفقیت خودش میبینه."
answer_student = "به نام یکتا دادار دادگر سلام و درود سخن دوست گرامی متین است اما در دسته‌بندی لذت‌ها ما لذتی را داریم که در آن فرد مورد نظر خود را با دیگری یا یک تیم، یکی می‌داند و پس از این وهم یا هرچه که آنرا بنامیم؛ پیروزی فرد مقابل را پیروزی خود حساب می‌کند … فکر کنم دیگه باید به پاسخ رسیده باشید ولی بیشتر توضیح می‌دهم و یک مثال مانند مثال شما می‌زنم: سخن شما هم دقیقا یکی از انواع به دنبال لذت بودن هست که فرد با درنظر گرفتن یکی بودن خود و بازیگر مورد علاقه‌اش، پیروزی او را در واقع موفقیت خودش حساب می‌کنه و اینگونه به دنبال لذت می‌رود … نمونه دیگر: زمان بازی تیم پرسپولیس فرا رسیده و میلیون‌ها تن انسان پشت تلویزیون‌ها نشسته‌اند و با دقت و هیجان بازی را می‌بینند هر گلی که پرسپولیس می‌زند آنها خوشحال می‌شوند و شادی می‌کنند و هر گلی که می‌خورد ناراحت می‌شوند؛ این هم نمونه‌ی دیگری از درنظر گرفتن خود با دیگران و به دنبال لذت رفتن … امیدوارم پاسخم شما را قانع کرده باشه؛ اگر پرسش دیگری نیز داشتید در خدمتم. پیروز باشید🌹دوستدار شما امیرحسین صفری."

# نرمال‌سازی
correct_norm = normalize_text(answer_correct)
student_norm = normalize_text(answer_student)

# embedding
emb_correct = model.encode(correct_norm)
emb_student = model.encode(student_norm)

# cosine similarity
similarity = util.cos_sim(emb_correct, emb_student).item()

print(f"similirity: {similarity:.4f}")

# طبقه‌بندی ساده (threshold مثلاً 0.82)
threshold = 0.75
is_correct = similarity >= threshold
print("is true" if is_correct else "is false")
