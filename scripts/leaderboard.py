"""Method leaderboard shared by the figures and the review dashboard.

Per-dataset means over runs; methods ranked per dataset (1 = best) on each
metric and the ranks averaged over datasets. Metrics with an ideal of 0.5
(attack/detection AUCs) are ranked by distance from 0.5.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "results", "tables")

UTILITY = {"tstr_catboost_auroc": +1, "tstr_catboost_f1_macro": +1, "alpha_precision": +1}
PRIVACY = {"P": +1, "mia_dist_auc": 0, "mia_domias_auc": 0, "aia_risk": -1}
# 2 = ideal value 1.0
FAIRNESS = {"repr_parity": +1, "minority_class_ratio": 2, "subgroup_util_ratio_min": +1,
            "rare_category_coverage": +1, "entropy_preservation": +1, "tail_preservation": 2}
EXTRA = {"c2st_auc": 0, "beta_recall": +1, "link_risk": -1, "share_closer_train": 0, "js_mean": -1,
         "stat_parity_diff_max": -1, "subgroup_util_ratio_mean": +1, "uniqueness": +1, "category_coverage": +1}

GEN_LABEL = {"rtf": "REaLTabFormer", "ctabgan_plus": "CTAB-GAN+", "bayesian_network": "BayesNet",
             "arf": "ARF", "tvae": "TVAE", "ctgan": "CTGAN", "tabddpm": "TabDDPM", "pategan": "PATE-GAN",
             "ctabgan_plus_dp": "DP-CTAB-GAN+ (clip)", "ctabgan_plus_dpgp": "DP-CTAB-GAN+ (GP)"}
VARIANT_LABEL = {"base": "", "psygentab": " + PSyGenTAB", "psygentab_min": " + PSyGenTAB (min.)",
                 "filter": " + DCR filter"}


def split_method(m):
    gen, _, var = m.partition("+")
    return gen, (var or "base")


def family(m):
    gen, var = split_method(m)
    if "_eps" in gen:
        return "dp"
    return var


def label(m):
    gen, var = split_method(m)
    if "_eps" in gen:
        base, eps = gen.rsplit("_eps", 1)
        return f"{GEN_LABEL.get(base, base)} ε={eps}"
    return GEN_LABEL.get(gen, gen) + VARIANT_LABEL.get(var, f" + {var}")


def per_dataset(df, metrics):
    return df.groupby(["method", "dataset"])[list(metrics)].mean()


def compute(df=None, include_filter=False):
    if df is None:
        df = pd.read_csv(os.path.join(TAB, "all_runs_long.csv"))
    if not include_filter:
        df = df[~df.method.str.endswith("+filter")]
    mets = {**UTILITY, **PRIVACY, **FAIRNESS, **EXTRA}
    mets = {k: v for k, v in mets.items() if k in df}
    dm = per_dataset(df, mets)
    out = dm.groupby("method").mean()
    ranks = {}
    for m, d in mets.items():
        piv = dm[m].unstack("method")
        score = -(piv - 0.5).abs() if d == 0 else -(piv - 1.0).abs() if d == 2 else d * piv
        ranks[m] = score.rank(axis=1, ascending=False).mean()
    R = pd.DataFrame(ranks)
    out["rank_utility"] = R[[m for m in UTILITY if m in R]].mean(1)
    out["rank_privacy"] = R[[m for m in PRIVACY if m in R]].mean(1)
    fair = [m for m in FAIRNESS if m in R]
    out["rank_fairness"] = R[fair].mean(1) if fair else np.nan
    out["rank_overall"] = (out.rank_utility + out.rank_privacy) / 2
    out["n_datasets"] = dm.reset_index().groupby("method").dataset.nunique()
    out["label"] = [label(m) for m in out.index]
    out["family"] = [family(m) for m in out.index]
    return out.sort_values("rank_overall"), dm


if __name__ == "__main__":
    lb, _ = compute()
    lb.to_csv(os.path.join(TAB, "leaderboard.csv"))
    cols = ["label", "rank_overall", "rank_utility", "rank_privacy", "tstr_catboost_auroc", "mia_domias_auc", "P"]
    print(lb[cols].round(3).to_string())
