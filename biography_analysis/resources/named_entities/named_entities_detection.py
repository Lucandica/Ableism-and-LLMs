import os
import re
import pandas as pd
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")

nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Sentence segmentation only: the transformer parser is unnecessary here and
# much slower. Splitting on '.' would break on initials ('J. Dupont'),
# abbreviations ('1er nov.') and truncate the entities the NER model sees.
sentencizer = spacy.blank("fr")
sentencizer.add_pipe("sentencizer")


def normalize_entity(text: str) -> str:
    text = re.sub(r"\s*-\s*", "-", text.strip())
    text = re.sub(r"\s*'\s*", "'", text)
    return text


def split_sentences(text: str) -> list:
    """Split a biography into sentences, keeping punctuation so the NER model
    sees well-formed input."""
    return [sent.text.strip() for sent in sentencizer(text).sents if sent.text.strip()]


def clean_locations(locations: list, name_strings: set) -> list:
    """Filter NER noise out of a location list.

    Rule 1: keep only entries containing at least one capitalized token
            (place names are proper nouns). Drops lowercase artifacts
            such as 'pari', 'capitale', 'région lyonnaise', '##poque',
            while keeping French-style venue names like
            'cimetière du Père-Lachaise' or 'hôpital Cochin'.
    Rule 2: drop entries whose full string was also extracted as a person
            name anywhere in the corpus (entity-type confusion, e.g.
            'Élise' tagged as LOC).
    """
    out = []
    for loc in locations:
        if not any(tok[:1].isupper() for tok in re.split(r"[\s\-']", loc) if tok):
            continue
        if loc.lower() in name_strings:
            continue
        if loc not in out:
            out.append(loc)
    return out


def apply_ner_detection(texts_folder):
    rows = []
    for file in (f for f in os.scandir(texts_folder) if f.name.endswith('.txt')):
        with open(file.path, 'r', encoding='utf-8') as f:
            text = f.read()

        ner_results = []
        for sentence in split_sentences(text):
            ner_results.extend(nlp(sentence))

        names = sorted({normalize_entity(e["word"]) for e in ner_results if e["entity_group"] == "PER"})
        orgs  = sorted({normalize_entity(e["word"]) for e in ner_results if e["entity_group"] == "ORG"})
        locs  = sorted({normalize_entity(e["word"]) for e in ner_results if e["entity_group"] == "LOC"})

        rows.append({
            "doc_id":        file.name,
            "names":         names,
            "organisations": orgs,
            "locations":     locs,
        })

    df_ner = pd.DataFrame(rows).sort_values("doc_id").reset_index(drop=True)

    name_strings = {n.lower() for lst in df_ner["names"] for n in lst}
    df_ner["locations_raw"] = df_ner["locations"]
    df_ner["locations"] = df_ner["locations"].apply(
        lambda l: clean_locations(l, name_strings))

    return df_ner