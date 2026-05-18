import pandas as pd
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations



def compute_cohens_d_and_pvalue(df:pd.DataFrame, column_to_compare:str):
    metrics = [c for c in df.columns if c not in ["doc_id", "prompt_version", "disability_in_prompt","run", "model", "version", "technique", "paragraph"]]
    means = df.groupby(column_to_compare)[metrics].mean()
    groups = df[column_to_compare].unique()

    rows = []
    for g1, g2 in combinations(groups, 2):
        d_dict = {}
        pvals_dict = {}
        direction_dict = {}

        for metric in metrics:
            x = df.loc[df[column_to_compare]==g1,metric].astype(float)
            y = df.loc[df[column_to_compare]==g2,metric].astype(float)

            wilcoxon = pg.wilcoxon(x, y)
            d = pg.compute_effsize(x, y)

            d_dict[metric] = round(d, 3)
            pvals_dict[metric] = wilcoxon['p_val'].iloc[0]
            direction_dict[metric] = f"{g2} > {g1}" if y.mean() > x.mean() else f"{g1} > {g2}"

        features = list(pvals_dict.keys())
        pvals = list(pvals_dict.values())
        dvals = list(d_dict.values())
        
        reject, pvals_corrected = pg.multicomp(pvals=pvals, alpha=0.05, method='fdr_bh')

        df_pair = pd.DataFrame({
            "comparison": f"{g1} vs {g2}",
            "feature":        features,
            "cohen_d":        dvals,
            "d_significant":  np.array(dvals) > 0.4,
            "pval_raw":       pvals,
            "pval_corrected": pvals_corrected,
            "reject_H0":      reject,
            f"mean_{g1}":     [means.loc[g1, m] for m in features],
            f"mean_{g2}":   [means.loc[g2, m] for m in features],
            "direction":      [direction_dict[m] for m in features],
            })
        rows.append(df_pair)

    return pd.concat(rows, ignore_index=True)