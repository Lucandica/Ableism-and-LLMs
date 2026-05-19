#Imports
import subprocess
import warnings

import pandas as pd
import numpy as np
import pingouin as pg

from pathlib import Path
from itertools import combinations
from functools import reduce

from resources.helper_functions import parse_doc_id

warnings.filterwarnings("ignore")

#Constant directories
BASE_DIR = Path().resolve()

GENERATION_DIR = BASE_DIR.parent / "generated_biographies" # Directory containing the texts to analyse
OUTPUT_DIR     = BASE_DIR / "outputs"  # Directory where outputs will be saved

RESOURCES_DIR  = BASE_DIR / "resources" # Directory where the resources are contained
COMPLEXITY_DIR = RESOURCES_DIR / "complexity_tool" # Directory were the resources for the complexity computation files are contained (DO NOT MODIFY)
ALSI_DIR = COMPLEXITY_DIR / "ALSI-main" # DIrectory of the ALSI tool (DO NOT MODIFY)

# Sheet merging helpers for complexity features

## Change target columns depending on needed ones
TARGET_COLUMNS = [
    "doc_id","word_count","unique_word_count","content_word_count","unique_content_word_count",
    "sentence_count","paragraph_count","avg_word_length","char_count","char_count_content",
    "avg_sentence_length","avg_content_word_length","count_NA","count_ADJ","count_ADP",
    "count_ADV","count_AUX","count_CCONJ","count_DET","count_INTJ","count_NOUN","count_NUM",
    "count_PRON","count_PROPN","count_PUNCT","count_SCONJ","count_SYM","count_VERB","count_X",
    "prop_NA","prop_ADJ","prop_ADP","prop_ADV","prop_AUX","prop_CCONJ","prop_DET","prop_INTJ",
    "prop_NOUN","prop_NUM","prop_PRON","prop_PROPN","prop_PUNCT","prop_SCONJ","prop_SYM",
    "prop_VERB","prop_X","present_count","past_count","future_count","conditional_count",
    "subjunctive_count","indicative_count","imperative_count","infinitive_count",
    "past_participle_count","present_participle_count","past_simple_count","present_prop",
    "past_prop","future_prop","conditional_prop","subjunctive_prop","indicative_prop",
    "imperative_prop","infinitive_prop","past_participle_prop","present_participle_prop",
    "past_simple_prop","TTR","maas","MATTR","simpsons_D","TTR_content","maas_content",
    "MATTR_content","simpsons_D_content","maas_verb","simpsons_D_verb","avg_sent_height",
    "avg_sent_height_adj","sd_sent_height","avg_dependency_depth_adj","avg_sd_depth",
    "avg_branching_factor","avg_max_incomplete_deps","avg_max_incomplete_deps_adj",
    "avg_incomplete_deps","avg_incomplete_deps_adj","n","s","total_paths",
    "avg_dependency_depth","prop_hf","prop_hi","avg_head_distance","avg_head_distance_adj",
    "max_head_distance","max_head_distance_adj","dependency_direction_index",
    "avg_integration_cost","avg_clause_length","complex_nom_per_sent","complex_verb_per_sent",
    "avg_dep_dist","avg_dep_count","clausal_density","mean_pos_surprisal","sd_pos_surprisal",
    "mean_pos_entropy","sd_pos_entropy","mean_pos_entropy_reduction","sd_pos_entropy_reduction",
    "token_sent_overlap_prev1","token_sent_overlap_prev5","lemma_sent_overlap_prev1",
    "lemma_sent_overlap_prev5","token_doc_overlap","content_sent_overlap_prev1",
    "content_sent_overlap_prev5","content_lemma_sent_overlap_prev1",
    "content_lemma_sent_overlap_prev5","content_doc_overlap","cosine_sent","cosine_content"
]

# Columns added by parse_doc_id — excluded from metric computations, those are the columns to compare. Modify if different
PARSED_COLUMNS = ["model", "prompt_version", "technique", "disability_in_prompt", "run"]
NON_METRICS_COLUMNS = ["doc_id"] + PARSED_COLUMNS

def xlsx_to_csv(xlsx_path: str, output_path: str) -> pd.DataFrame:
    all_sheets = pd.read_excel(xlsx_path, sheet_name=None)

    relevant = []
    for name, df in all_sheets.items():
        if "doc_id" not in df.columns:
            continue
        # Skip sheets that are not doc-level (doc_id not unique = sentence/token level)
        if df["doc_id"].duplicated().any():
            continue
        useful_cols = ["doc_id"] + [c for c in df.columns if c in TARGET_COLUMNS and c != "doc_id"]
        if len(useful_cols) > 1:
            relevant.append(df[useful_cols])

    merged = reduce(lambda l, r: pd.merge(l, r, on="doc_id", how="outer"), relevant)
    merged = merged.reindex(columns=TARGET_COLUMNS)
    merged.to_csv(output_path, index=False)
    return merged

# Statistics helper functions

def compute_cohens_d_and_pvalue(df: pd.DataFrame, column_to_compare: str) -> pd.DataFrame:
    metrics = [c for c in df.columns if c not in NON_METRICS_COLUMNS]
    means = df.groupby(column_to_compare)[metrics].mean()
    groups = df[column_to_compare].dropna().unique()

    rows = []
    for g1, g2 in combinations(groups, 2):
        d_dict = {}
        pvals_dict = {}
        direction_dict = {}

        for metric in metrics:
            x = df.loc[df[column_to_compare] == g1, metric].astype(float)
            y = df.loc[df[column_to_compare] == g2, metric].astype(float)

            wilcoxon = pg.wilcoxon(x, y)
            d = pg.compute_effsize(x, y)

            d_dict[metric] = round(d, 3)
            pvals_dict[metric] = wilcoxon['p_val'].iloc[0]
            direction_dict[metric] = f"{g2} > {g1}" if y.mean() > x.mean() else f"{g1} > {g2}"

        features = list(pvals_dict.keys())
        pvals = list(pvals_dict.values())
        dvals = list(d_dict.values())

        reject, pvals_corrected = pg.multicomp(pvals=pvals, alpha=0.05, method='fdr_bh')

        g1_str, g2_str = str(g1), str(g2)
        df_pair = pd.DataFrame({
            "comparison":     f"{g1_str} vs {g2_str}",
            "grouping_var":   column_to_compare,
            "group_1":        g1_str,
            "group_2":        g2_str,
            "feature":        features,
            "cohen_d":        dvals,
            "d_significant":  np.abs(np.array(dvals, dtype=float)) > 0.4,
            "pval_raw":       pvals,
            "pval_corrected": pvals_corrected,
            "reject_H0":      reject,
            "mean_group_1":   [means.loc[g1, m] for m in features],
            "mean_group_2":   [means.loc[g2, m] for m in features],
            "direction":      [direction_dict[m] for m in features],
        })
        rows.append(df_pair)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

# First step: Running ALSI for complexity computation

def run_alsi(generation_dir: Path = GENERATION_DIR):

    # Run alsi tool
    subprocess.run(
        ["Rscript", "R/main.R"],
        cwd=ALSI_DIR,
        capture_output=True, text=True, encoding="utf-8"
    )

    #Conversion of the generated .Rds object to a .xlsx
    subprocess.run(
    ["Rscript", str(COMPLEXITY_DIR / "conversion.R")],
    cwd=ALSI_DIR / "out",
    capture_output=True, text=True, encoding="utf-8"
    )

    df_features = xlsx_to_csv(
    str(ALSI_DIR / "out" / "biographies_features_only.xlsx"),
    str(ALSI_DIR / "out" / "biographies_features_only_combined.csv")
    )
    return df_features


def get_complexity_statistics(df_features: pd.DataFrame = None, add_alsi_run: bool = True, generation_dir: Path = GENERATION_DIR):
    if add_alsi_run:
        df_features = run_alsi(generation_dir)

    df_features = parse_doc_id(df_features) # /!\ The parser gets the columns to compare based on the name of the files using this function.
                                            # Modify the function if the files have different names.

    output_path = OUTPUT_DIR / "biographies_stats_complexity.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for col in PARSED_COLUMNS:
            df_stats = compute_cohens_d_and_pvalue(df_features, col)
            if not df_stats.empty:
                df_stats.to_excel(writer, sheet_name=col, index=False)

    return df_features


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute complexity statistics for generated biographies.")
    parser.add_argument("--csv", type=str, default=None, help="Path to an existing features CSV. If omitted, ALSI is run first.")
    parser.add_argument("--run-alsi", action="store_true", help="Force ALSI to run even if --csv is provided.")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        get_complexity_statistics(df_features=df, add_alsi_run=args.run_alsi)
    else:
        get_complexity_statistics(add_alsi_run=True)

    print(f"Done. Output written to {OUTPUT_DIR / 'biographies_stats_complexity.xlsx'}")
