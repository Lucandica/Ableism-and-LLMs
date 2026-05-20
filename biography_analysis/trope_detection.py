# /!\ for now, this script only detect occurrences of "malgré cela", but it will evolve later for better inclusivity.

import os
import spacy
import pandas as pd
from pathlib import Path

from resources.disabilities.disabilities_detection import (
    PATTERN_FR, REVERSE_LOOKUP_FR, collect_hits, get_categories
)

BASE_DIR = Path().resolve()
GENERATION_DIR = BASE_DIR.parent / "generated_biographies"
OUTPUT_DIR = BASE_DIR / "outputs"


def detect_malgre_cela(generation_dir: Path = GENERATION_DIR, model: str = "fr_dep_news_trf") -> pd.DataFrame:
    """
    Detect occurrences of 'malgré cela' in biography .txt files using spaCy dependency parsing.

    For each sentence containing 'malgré' whose syntactic head is 'cela', retrieves the
    previous sentence and checks it for disability keyword hits.

    Args:
        generation_dir: folder containing the .txt biography files.
        model: spaCy model name to use for parsing (must support French dependency parsing).

    Returns:
        DataFrame with one row per 'malgré cela' occurrence, with columns:
        doc_id, sentence_id, sentence, prev_sentence, disability_hits, disability_categories.
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
                if token.text.lower() == "malgré" and token.head.text.lower() == "cela":
                    prev_sentence = sentences[sent_idx - 1].text if sent_idx > 0 else None
                    disability_hits = collect_hits(PATTERN_FR, prev_sentence) if prev_sentence else []
                    rows.append({
                        "doc_id": doc_id,
                        "sentence_id": sent_idx,
                        "sentence": sent.text,
                        "prev_sentence": prev_sentence,
                        "disability_hits": disability_hits,
                        "disability_categories": get_categories(disability_hits, REVERSE_LOOKUP_FR),
                    })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = detect_malgre_cela()
    output_path = OUTPUT_DIR / "trope_detection.csv"
    df.to_csv(output_path, index=False)
    print(f"Done. Found {len(df)} occurrences. Output written to {output_path}")
