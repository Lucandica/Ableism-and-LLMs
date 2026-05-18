import ast
import pandas as pd
from collections import Counter
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from functools import reduce

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
        return ast.literal_eval(val)
    return val if isinstance(val, list) else []

def plot_version_summary(df, columns: list[str], title: str, fig_width: int = 14):
    """
    Plot horizontal bar charts of value counts per version for given columns.

    Args:
        df:        DataFrame with a "version" column and the specified columns.
        columns:   List of column names to count and plot (one subplot each).
        title:     Main figure title.
        fig_width: Width of the figure (default 14, increase for more columns).
    """
    versions = df["version"].unique()
    n_versions = len(versions)
    n_cols = len(columns)

    # Build counters for each version
    height_ratios = []
    counters_per_version = []
    for version in versions:
        group = df[df["version"] == version]
        counters = [
            Counter(
                item.strip()
                for sublist in group[col].apply(parse_list)
                for item in sublist
            )
            for col in columns
        ]
        counters_per_version.append(counters)
        height_ratios.append(max(len(c) for c in counters))

    # Build figure
    fig_height = 2 + sum(height_ratios) * 0.20
    fig, axes = plt.subplots(
        n_versions, n_cols,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": height_ratios}
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    if n_versions == 1:
        axes = [axes]

    # Plot
    for row, (version, counters) in enumerate(zip(versions, counters_per_version)):
        for col, (counter, subtitle) in enumerate(zip(counters, columns)):
            ax = axes[row][col]
            labels, values = zip(*counter.most_common()) if counter else ([], [])
            bars = ax.barh(labels[::-1], values[::-1],
                           color="#4C72B0", edgecolor="white", linewidth=0.6, height=0.6)
            for bar, val in zip(bars, values[::-1]):
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                        str(val), va="center", ha="left", fontsize=9)
            ax.set_title(f"{version.upper()} — {subtitle.capitalize()}",
                         fontsize=11, fontweight="bold", loc="left")
            ax.set_xlabel("Count", fontsize=9)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="both", labelsize=9)
            ax.grid(axis="x", linestyle="--", alpha=0.4, color="grey")

    plt.tight_layout()
    plt.savefig(f"./outputs/{title}.pdf")
    plt.show()


# Sheet merging for complexity features
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

def parse_doc_id(df):
    parts = df["doc_id"].str.replace(".txt", "", regex=False).str.split("_")
    
    new_cols = pd.concat([
        parts.str[0].rename("model"),
        parts.str[1].rename("prompt_version"),
        parts.str[2].map({"short": "MLX", "long": "Transformers"}).rename("technique"),
        parts.str[3].map({"withdis": True, "nodis": False}).rename("disability_in_prompt"),
        parts.str[4].astype(int).rename("run"),
    ], axis=1)
    
    return pd.concat([df, new_cols], axis=1)