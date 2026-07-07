import os
import spacy
import pandas as pd
from pathlib import Path

from resources.disabilities.disabilities_detection import (
    PATTERN, REVERSE_LOOKUP, collect_hits, get_categories
)

BASE_DIR = Path().resolve()
GENERATION_DIR = BASE_DIR.parent / "generated_biographies"
OUTPUT_DIR = BASE_DIR / "outputs"

# Determiners / modifiers kept to build a readable object phrase. Relative clauses (acl:relcl) and prepositional tails
# (nmod carrying their own case) are deliberately excluded so phrases stay short and groupable.

KEEP_DEPS = {"det", "amod", "nummod", "fixed", "flat", "flat:name",
             "compound", "nmod:poss", "advmod"}


def extract_malgre_object(malgre_token):
    """
    Describe what a 'malgré' token governs.
    """
    head = malgre_token.head
    object_lemma = head.text.lower() if head.pos_ == "PRON" else head.lemma_.lower()

    kept = [head] + [c for c in head.children
                     if c.i != malgre_token.i and c.dep_ in KEEP_DEPS]
    lo, hi = min(t.i for t in kept), max(t.i for t in kept)
    phrase = malgre_token.doc[lo:hi + 1].text
    return head.text, object_lemma, head.pos_, phrase


def detect_malgre(generation_dir: Path = GENERATION_DIR, model: str = "fr_dep_news_trf"):
    """
    Detect every occurrence of 'malgré' in biography .txt files using spaCy
    dependency parsing, and record the phrase it governs.

    For each 'malgré' occurrence the previous sentence is retrieved and checked
    for disability keyword hits.
    """
    nlp = spacy.load(model)

    rows = []
    for file in sorted((f for f in os.scandir(generation_dir) if f.name.endswith(".txt")), key=lambda f: f.name):
        doc_id = file.name.replace(".txt", "")
        with open(file.path, "r", encoding="utf-8") as fh:
            text = fh.read()

        doc = nlp(text)
        sentences = list(doc.sents)

        for sent_idx, sent in enumerate(sentences):
            for token in sent:
                if token.text.lower() != "malgré":
                    continue

                obj_text, obj_lemma, obj_pos, phrase = extract_malgre_object(token)
                prev_sentence = sentences[sent_idx - 1].text if sent_idx > 0 else None
                disability_hits = collect_hits(PATTERN, prev_sentence) if prev_sentence else []
                rows.append({
                    "doc_id": doc_id,
                    "sentence_id": sent_idx,
                    "sentence": sent.text,
                    "prev_sentence": prev_sentence,
                    "malgre_object": obj_text,
                    "malgre_object_lemma": obj_lemma,
                    "malgre_object_pos": obj_pos,
                    "malgre_phrase": phrase,
                    "disability_hits": disability_hits,
                    "disability_categories": get_categories(disability_hits, REVERSE_LOOKUP),
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = detect_malgre()
    output_path = OUTPUT_DIR / "trope_detection.csv"
    df.to_csv(output_path, index=False)