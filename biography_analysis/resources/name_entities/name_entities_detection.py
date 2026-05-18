import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")

nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def apply_ner_detection(texts_folder):
    rows = []
    for file in (f for f in os.scandir(texts_folder) if f.name.endswith('.txt')):
        with open(file.path, 'r', encoding='utf-8') as f:
            text = f.read()

        ner_results = []
        for sentence in text.split('.'):
            sentence = sentence.strip()
            if sentence:
                ner_results.extend(nlp(sentence))

        names = sorted({e["word"] for e in ner_results if e["entity_group"] == "PER"})
        orgs  = sorted({e["word"] for e in ner_results if e["entity_group"] == "ORG"})
        locs  = sorted({e["word"] for e in ner_results if e["entity_group"] == "LOC"})

        rows.append({
            "doc_id":          file.name,
            "names":         names,
            "organisations": orgs,
            "locations":     locs,
        })

    df_ner = pd.DataFrame(rows).sort_values("doc_id").reset_index(drop=True)
    return df_ner