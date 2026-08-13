"""
Applies disability-related keywords search, automatic gender attribution and ner search.
Concatenate all output dfs from those methods into one representation df.
"""


from pathlib import Path
import pandas as pd
import warnings

from resources.helper_functions import parse_doc_id
from resources.gender.gender_detection import apply_gender_detection
from resources.disabilities.disabilities_detection import apply_disability_detection
from resources.named_entities.named_entities_detection import apply_ner_detection

warnings.filterwarnings("ignore")


# Dir paths
BASE_DIR = Path().resolve()

GENERATION_DIR = BASE_DIR.parent / "generated_biographies"
RESOURCES_DIR  = BASE_DIR / "resources"
OUTPUT_DIR     = BASE_DIR / "outputs"

GENDER_DIR        = RESOURCES_DIR / "gender"
DISABILITIES_DIR  = RESOURCES_DIR / "disabilities"
NAME_ENTITIES_DIR = RESOURCES_DIR / "name_entities"

KEY = "doc_id"

def merge_on_doc_id(left, right):
    "Join all the representation analysis dataframe into one, based on doc_id"
    extra = right.columns.difference(left.columns)
    return left.merge(right[[KEY, *extra]], on=KEY, how="outer", validate="one_to_one")



def get_gender_dis_ner_representation(generation_dir:str=GENERATION_DIR):
    """
    Apply all representation-related functions.

    Args:
        generation_dir: folder containing all texts we want to do the representation of.

    Returns:
        gender_dis_ner_detection: a df where each row correspond to one text, and columns are all the one from the three representation methods.
    """
    df_gender = parse_doc_id(apply_gender_detection(generation_dir))
    df_dis    = parse_doc_id(apply_disability_detection(generation_dir))
    df_ner    = parse_doc_id(apply_ner_detection(generation_dir))


    result = merge_on_doc_id(merge_on_doc_id(df_gender, df_dis), df_ner)
    result = result.sort_values(KEY).reset_index(drop=True)

    result.to_csv(OUTPUT_DIR / "gender_dis_ner_detection.csv")
    

if __name__ == "__main__":
    get_gender_dis_ner_representation(generation_dir=GENERATION_DIR)
    print(f"Done. Output written to {OUTPUT_DIR / 'gender_dis_ner_detection.csv'}")