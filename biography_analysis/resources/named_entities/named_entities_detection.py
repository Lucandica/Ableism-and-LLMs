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

nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first")

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

# Function for Location detection

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

# Constants and Functions for Name detection

# List the titles to later remove them as they are sometimes considered as part of the name
TITLES = {"madame", "mademoiselle", "monsieur", "mme", "mlle", "m.", "mr",
          "dr", "docteur", "professeur", "pr"}

# List particles to later have them with the last name
PARTICLES = {"de", "du", "des", "d", "le", "la", "les", "van", "von", "den",
             "der", "di", "da", "dos", "del", "della", "ten", "ter"}

ELIDED_PARTICLES = {"d", "l"}
ELIDED_RE = re.compile(rf"^({'|'.join(sorted(ELIDED_PARTICLES))})[’']", re.IGNORECASE)


# Have a detector of roman numbers as there could be occurrences of names detected with them (e.g., Louis XVI)
ROMAN_RE = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


def is_bare_particle(token: str) -> bool:
    """
    True if the token is nothing but a separated particle ('de', 'van') not with a name.
    """
    return token.lower().strip(".'’") in PARTICLES

def opens_on_particle(token: str) -> bool:
    """
    True if a token carries a particle that is either elided or not.
    """
    return is_bare_particle(token) or bool(ELIDED_RE.match(token))


def is_name_noise(token: str) -> bool:
    """ Flags names that are incorrectly parsed or after an elided particle"""
    if "##" in token:
        return True
    if is_bare_particle(token):
        return False
    stripped = token.strip(".'-")
    if len(stripped) < 3:
        return True
    if not ELIDED_RE.sub("", stripped)[:1].isupper():
        return True
    return not any(c.isalpha() for c in token)

def is_roman(token: str) -> bool:
    "Flag roman number next to names"
    return bool(token) and token.isupper() and bool(ROMAN_RE.match(token))

def clean_names(names: list) -> list:
    "Drop the person entities made only of noise, keeping the original order."
    out = []
    for name in names:
        if any(not is_name_noise(tok) for tok in name.split()) and name not in out:
            out.append(name)
    return out

def split_full_name(name: str) -> tuple:
    """Split one person entity into (first_name, surname).
    A part can be no if no surname of first name is mentioned.

    """
    tokens = [t for t in str(name).replace("\u00a0", " ").split() if t]
    tokens = [t for t in tokens if t.lower().strip(".") not in TITLES]

    # roman numbers are dropped
    tokens = [t for t in tokens if not is_name_noise(t) and not is_roman(t)]
    if not tokens:
        return None, None

    # an entity made of nothing but particles carries no name
    if all(is_bare_particle(t) for t in tokens):
        return None, None
 
    head, tail = tokens[0], tokens[1:]

    # get names that are starting by a particles
    if opens_on_particle(head):
        return None, " ".join(tokens)
    if not tail:
        return head, None
    return head, " ".join(tail)
 
 
def build_name_lexicons(name_lists) -> tuple:
    """
    Looks up which names or surnames already exist in order to determine if
    a lonely name is rather a first name or a last name, based on what it has seen already.  
    """
    firsts, lasts = {}, {}
    for names in name_lists:
        for name in names:
            first, last = split_full_name(name)
            if first and last:
                firsts[first] = firsts.get(first, 0) + 1
                lasts[last] = lasts.get(last, 0) + 1
                bare = last.split()[-1]
                lasts[bare] = lasts.get(bare, 0) + 1
    return firsts, lasts
 
 
def classify_single(token: str, firsts: dict, lasts: dict,
                    extra_firsts=(), extra_lasts=()) -> str:
    """
    Classify the sing names based on the builded lexicon of first names and last names
    """
    in_first, in_last = token in firsts, token in lasts
    if in_first and in_last:
        return "first" if firsts[token] >= lasts[token] else "last"
    if in_first:
        return "first"
    if in_last:
        return "last"
    if token in extra_firsts:
        return "first"
    if token in extra_lasts:
        return "last"
    return None
 
 
def add_name_columns(df_ner, extra_firsts=(), extra_lasts=()):
    """
    Add duplicate-free 'first_names', 'last_names' and 'unclassified_names' columns
    """
    firsts_lex, lasts_lex = build_name_lexicons(df_ner["names"])
    extra_firsts, extra_lasts = set(extra_firsts), set(extra_lasts)
 
    def split_row(names):
        firsts, lasts, unknown = [], [], []
        for name in names:
            first, last = split_full_name(name)
            if first and last:
                firsts.append(first)
                lasts.append(last)
            elif first:
                role = classify_single(first, firsts_lex, lasts_lex,
                                       extra_firsts, extra_lasts)
                (firsts if role == "first" else
                 lasts if role == "last" else unknown).append(first)
            elif last:
                lasts.append(last)

        return (list(dict.fromkeys(firsts)),
                list(dict.fromkeys(lasts)),
                list(dict.fromkeys(unknown)))
 
    split = df_ner["names"].apply(split_row)
    df_ner["first_names"] = split.str[0]
    df_ner["last_names"] = split.str[1]
    df_ner["unclassified_names"] = split.str[2]
    return df_ner


# Apply named entities detection
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

    df_ner["names_raw"] = df_ner["names"]
    df_ner["names"] = df_ner["names"].apply(clean_names)


    name_strings = {n.lower() for lst in df_ner["names_raw"] for n in lst}
    df_ner["locations_raw"] = df_ner["locations"]
    df_ner["locations"] = df_ner["locations"].apply(
        lambda l: clean_locations(l, name_strings))

    return add_name_columns(df_ner)