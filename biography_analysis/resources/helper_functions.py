import ast
import pandas as pd
from collections import Counter
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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