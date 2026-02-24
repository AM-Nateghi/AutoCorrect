from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_id = "CohereLabs/tiny-aya-earth"
adapter_path = "./results/final-model"

device = "cuda" if torch.cuda.is_available() else "cpu"

# tokenizer را از همون دایرکتوری نهایی که ذخیره کردی بگیر
tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# مدل پایه
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float16,
    device_map="auto" if device == "cuda" else None,
)

# لود کردن LoRA
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()


def build_prompt(question_text, true_answer, student_answer):
    return (
        f"Question: {question_text}\n"
        f"True response: {true_answer}\n"
        f"Student answer: {student_answer}\n"
        "Student's answer is correct with True response? YES/NO\n"
        "Model:"
    )


def predict_is_correct(question_text, true_answer, student_answer, max_new_tokens=3):
    prompt = build_prompt(question_text, true_answer, student_answer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    generated = generated.strip().upper()
    if generated.startswith("YES"):
        return 1, generated
    if generated.startswith("NO"):
        return 0, generated
    return None, generated  # اگر جواب واضح YES/NO نشد


state, text = predict_is_correct(
    "علیت فیزیکی و عقلی را توضیح داده و بیان کنید تفاوتشان در چیست؟",
    "علیت فیزیکی مربوط به حوادث مادی و زمینه‌ساز (علت مُعِدّه) است که در آن معلول می‌تواند در وجود خود مستقل از علت باقی بماند (مانند بنّا و ساختمان). اما علیت عقلی، علیت حقیقی و هستی‌بخش است که در آن تمام وجودِ معلول وابسته به علت است و هیچ استقلالی از خود ندارد؛ به‌طوری که معلول در احاطه وجودی علت است و با قطع رابطه با آن، نابود می‌شود (مانند نسبتِ نفس به تصورات ذهنی).",
    "علتهایی که در عالم ماده معمولا می‌شناسیم علیت فیزیکی هستند که در واقع علت حقیقی نیستند. مثلا می‌گوییم دانه گندم علت پیدایش خوشه‌ی گندم می‌شود. این علیت فیزیکی است که در فلسفه به آنها علت معده گویند. (یا به تعبیر زیباتر حضرت استاد وکیلی موضوع حکمت افعال علت فاعلی هستند) ولی علیت عقلی همانا علت هستی‌بخش و علت حقیقی است است که معلول در هستی خود محتاج به علت است. مثل تصوراتی که داریم نسبت به نفس ما. پر واضح است که خوشه‌ی گندم در تحت احاطه و سیطره‌ی وجودی دانه‌ی گندم نیست. (دانه‌ی گندم صرفا موضوع حکمت علت فاعلی است) ولی تصورات ما در تحت احاطه‌ی وجودی نفس ما هستند. ببین تفاوت ره از کجاست تا به کجا.",
)

print(state, text)
