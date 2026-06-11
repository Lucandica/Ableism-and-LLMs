import ast
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def get_cohen_kappa(csv_path:str, column1:str, column2:str):
    """
    Get inter-annotator agreement between two annotators, based on Cohen's kappa.

    Args:
        csv_path (str): the path to the dataframe containing annotations.
        column1 (str): one of the two annotator.
        column2 (str): the second annotator.

    Returns:
        cohen_k_score (float): the cohen's kappa score of between the two annotators
    """

    df = pd.read_csv(csv_path)

    if df[column1].isnull().any() or df[column2].isnull().any():
        raise ValueError(f"Null value detected in annotator columns: '{column1}', '{column2}'")

    cohen_k_score = cohen_kappa_score(df[column1], df[column2])

    return cohen_k_score

def get_versions(df):
    for i, row in df.iterrows():
        if "nodis" in row["file_name"]:
            df.at[i, "version"] = "nodis"
        else: df.at[i, "version"] = "withdis"
    
    return df

def parse_list(val):
    if isinstance(val, str):
        result = ast.literal_eval(val)
    elif isinstance(val, list):
        result = val
    else:
        return []
    return list(dict.fromkeys(result))


def parse_doc_id(df):
    parts = df["doc_id"].str.replace(".txt", "", regex=False).str.split("_")
    
    new_cols = pd.concat([
        parts.str[0].rename("model"),
        (parts.str[1] + "_" + parts.str[2]).rename("prompt_version"),
        parts.str[3].map({"withdis": True, "nodis": False}).rename("disability_in_prompt"),
        parts.str[4].astype(int).rename("run"),
        parts.str[5].rename("quant"),
    ], axis=1)
    
    return pd.concat([df, new_cols], axis=1)