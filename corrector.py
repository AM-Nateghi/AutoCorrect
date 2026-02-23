from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, List
from sentence_transformers import SentenceTransformer, util
import re
import hazm
import pandas as pd

qa_df = pd.read_csv("./qa_dataset.csv")
model = SentenceTransformer("PartAI/Tooka-SBERT-V2-Large")

normalizer = hazm.Normalizer()
tokenizer = hazm.WordTokenizer()
stemmer = hazm.Stemmer()
lemmatizer = hazm.Lemmatizer()
stopwords = hazm.stopwords_list()

additional_word = [
    "سلام",
    "درود",
    "پیروز",
    "دوستدار",
    "امیدوارم",
    "پرسش",
    "بنام",
    "یاعلی",
    "یا علی",
    "خداحافظ",
    "موفق",
]

# Expanded irrelevant stems/lemmas
irrelevant_stems = set(
    [
        stemmer.stem(w)
        for w in [
            "سلام",
            "درود",
            "پیروز",
            "دوستدار",
            "امیدوارم",
            "پرسش",
            "بنام",
            "یاعلی",
            "یا علی",
            "خداحافظ",
            "موفق",
        ]
    ]
    + [
        lemmatizer.lemmatize(w)
        for w in [
            "سلام",
            "درود",
            "پیروز",
            "دوستدار",
            "امیدوارم",
            "پرسش",
            "خدمت",
            "خدا",
            "بنام",
            "یاعلی",
            "یا علی",
            "خداحافظ",
            "موفق",
            "باشید",
            "🌹",
            "✨",
        ]
    ]
)


def filter_irrelevant(sentences: List[str]) -> List[str]:
    filtered = []
    for s in sentences:
        norm_s = normalizer.normalize(s)
        tokens = tokenizer.tokenize(norm_s)
        stems = [stemmer.stem(t) for t in tokens]
        lems = [lemmatizer.lemmatize(t) for t in tokens]
        stem_str = " ".join(stems)
        if any(st in irrelevant_stems for st in stems + lems):
            # Check position with regex on stemmed string
            if re.search(
                r"^(\W*(" + "|".join(irrelevant_stems) + r")\W*)",
                stem_str,
                re.IGNORECASE,
            ) or re.search(
                r"(\W*(" + "|".join(irrelevant_stems) + r")\W*)$",
                stem_str,
                re.IGNORECASE,
            ):
                continue
        filtered.append(s)
    return filtered


def extract_keywords(text: str, top_n=10) -> set:
    norm_text = normalizer.normalize(text)
    tokens = [
        stemmer.stem(t) for t in tokenizer.tokenize(norm_text) if t not in stopwords
    ]
    if not tokens:
        return set()
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform([" ".join(tokens)])
    feature_names = vectorizer.get_feature_names_out()
    sorted_items = sorted(zip(vectorizer.idf_, feature_names), reverse=True)[:top_n]
    return set(word for _, word in sorted_items)


def check(true_sentence: str, student_sentence: str) -> Tuple[bool, float, float]:
    true_norm = normalizer.normalize(true_sentence)
    stud_norm = normalizer.normalize(student_sentence)

    true_sents = filter_irrelevant(hazm.sent_tokenize(true_norm))
    stud_sents = filter_irrelevant(hazm.sent_tokenize(stud_norm))
    true_filtered = " ".join(true_sents)
    stud_filtered = " ".join(stud_sents)

    emb_true = model.encode(true_filtered) if true_filtered else model.encode(true_norm)
    emb_stud = model.encode(stud_filtered) if stud_filtered else model.encode(stud_norm)
    cos_sim = util.cos_sim(emb_true, emb_stud).item()

    true_keywords = extract_keywords(true_filtered or true_norm)
    stud_keywords = extract_keywords(stud_filtered or stud_norm)
    if not true_keywords:
        keyword_overlap = 0.0
    else:
        keyword_overlap = len(true_keywords.intersection(stud_keywords)) / len(
            true_keywords
        )

    combined = 0.6 * cos_sim + 0.4 * keyword_overlap
    return (combined >= 0.65, combined, keyword_overlap)


# Check the success model ratio
success_count = 0
_cototal = 0

for qt, ta, sa, ic, cr in qa_df.values:
    is_correct, _, _ = check(ta, sa)
    if is_correct == ic:
        success_count += 1
    _cototal += 1
    break

print(success_count)
