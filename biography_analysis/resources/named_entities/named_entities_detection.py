"""
Functions to apply named entity recognition model and get names, locations and organisms mentioned.
"""

import os
import re
import pandas as pd
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Initiate model and tokenizer for NER
tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")

nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Initiate a sentencizer for avoid point to always indicate end of sentence (Example: J. Deschamps)
sentencizer = spacy.blank("fr")
sentencizer.add_pipe("sentencizer")


def normalize_entity(text: str) -> str:
    "Remove spaces around hyphens and apostrophes"
    text = re.sub(r"\s*-\s*", "-", text.strip())
    text = re.sub(r"\s*'\s*", "'", text)
    return text


def split_sentences(text: str) -> list:
    "Split a biography into sentences, keeping punctuation so the NER model sees well-formed input."
    return [sent.text.strip() for sent in sentencizer(text).sents if sent.text.strip()]


def clean_locations(locations: list, name_strings: set) -> list:
    """Filter NER noise out of a location list:
    1: keep only entries containing at least one capitalized token (place names are proper nouns). Drops lowercase artifacts, 
       while keeping French-style venue names like 'cimetière du Père-Lachaise' or 'hôpital Cochin'.
    2: drop entries whose full string was also extracted as a person name anywhere in the corpus.
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
    """
    Iterate through a folder of texts, and get all the locations, names, organisms per texts.

    Args: 
        texts_folder: A path to a folder with only txt files on which we want to detect disability keywords.

    Returns:
        df_ner: A dataframe where each row correspond to a text file,
                and the name of the file, locations, names and organisms detected are columns.
    """
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