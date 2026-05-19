# Imports
from pathlib import Path
import pandas as pd
import warnings

from resources.helper_functions import parse_doc_id
from resources.gender.gender_detection import apply_gender_detection
from resources.disabilities.disabilities_detection import apply_disability_detection
from resources.name_entities.name_entities_detection import apply_ner_detection

warnings.filterwarnings("ignore")


# Dir paths
BASE_DIR = Path().resolve()

GENERATION_DIR = BASE_DIR.parent / "generated_biographies"
RESOURCES_DIR  = BASE_DIR / "resources"
OUTPUT_DIR     = BASE_DIR / "outputs"

GENDER_DIR        = RESOURCES_DIR / "gender"
DISABILITIES_DIR  = RESOURCES_DIR / "disabilities"
NAME_ENTITIES_DIR = RESOURCES_DIR / "name_entities"


def get_gender_dis_ner_representation(generation_dir:str=GENERATION_DIR):
    df_gender = apply_gender_detection(generation_dir)
    df_gender = parse_doc_id(df_gender)

    df_dis = apply_disability_detection(generation_dir)
    df_dis = parse_doc_id(df_dis)

    df_ner = apply_ner_detection(generation_dir)
    df_ner = parse_doc_id(df_ner)

    unique_df2 = df_dis.columns.difference(df_gender.columns)
    unique_df3 = df_ner.columns.difference(df_gender.columns)

    result = pd.concat([df_gender, df_dis[unique_df2], df_ner[unique_df3]], axis=1)
    result.to_csv(OUTPUT_DIR / "gender_dis_ner_detection.csv")
    

if __name__ == "__main__":
    get_gender_dis_ner_representation(generation_dir=GENERATION_DIR)
    print(f"Done. Output written to {OUTPUT_DIR / 'gender_dis_ner_detection.csv'}")