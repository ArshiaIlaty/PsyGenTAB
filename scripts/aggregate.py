"""Aggregate run results, run the statistical tests, and write LaTeX tables.

Statistics (datasets are the blocks / pairing unit, using per-dataset means):
  * Friedman test across methods + Nemenyi post-hoc and average ranks;
  * Wilcoxon signed-rank of PSyGenTAB vs. every other method, Holm-corrected,
    with matched-pairs rank-biserial effect size;
  * Wilcoxon of G+psygentab vs. G for every generator G (model-agnosticism).
Metrics whose ideal is 0.5 (detection/attack AUCs, share closer to train) are
ranked by their distance from 0.5.
"""
import glob
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
TAB = os.path.join(RES, "tables")

# metric -> (direction, label); +1 higher better, -1 lower better, 0 ideal 0.5, 2 ideal 1.0
METRICS = {
    "tstr_catboost_auroc": (+1, "TSTR AUROC (CatBoost)"),
    "tstr_catboost_f1_macro": (+1, "TSTR macro-F1 (CatBoost)"),
    "tstr_catboost_bal_acc": (+1, "TSTR bal. acc. (CatBoost)"),
    "tstr_hgb_auroc": (+1, "TSTR AUROC (HistGB)"),
    "tstr_hgb_f1_macro": (+1, "TSTR macro-F1 (HistGB)"),
    "marginal_err": (-1, "Marginal error"),
    "wasserstein_norm": (-1, "Norm. Wasserstein"),
    "assoc_err": (-1, "Association error"),
    "c2st_auc": (0, "C2ST detection AUC"),
    "alpha_precision": (+1, "alpha-precision"),
    "beta_recall": (+1, "beta-recall"),
    "authenticity": (+1, "Authenticity"),
    "prdc_density": (+1, "Density"),
    "prdc_coverage": (+1, "Coverage"),
    "P": (+1, "Privacy score P"),
    "share_closer_train": (0, "Share closer to train"),
    "dcr_ratio": (+1, "DCR ratio (syn/holdout)"),
    "nndr_ratio": (+1, "NNDR ratio (syn/holdout)"),
    "exact_match_syn": (-1, "Exact-match rate"),
    "mia_dist_auc": (0, "Distance-MIA AUC"),
    "mia_domias_auc": (0, "DOMIAS AUC"),
    "aia_risk": (-1, "Attribute-inference risk"),
    "link_risk": (-1, "Linkability risk"),
    "js_mean": (-1, "Mean JS divergence"),
    "repr_parity": (+1, "Representation parity"),
    "stat_parity_diff_max": (-1, "Max. statistical-parity gap"),
    "minority_class_ratio": (2, "Minority-class share ratio"),
    "subgroup_util_ratio_min": (+1, "Worst-subgroup utility ratio"),
    "subgroup_util_ratio_mean": (+1, "Mean subgroup utility ratio"),
    "uniqueness": (+1, "Uniqueness"),
    "category_coverage": (+1, "Category coverage"),
    "rare_category_coverage": (+1, "Rare-category coverage"),
    "entropy_preservation": (+1, "Entropy preservation"),
    "tail_preservation": (2, "Tail preservation"),
}
MAIN = "rtf+psygentab"


def load():
    rows = []
    for p in glob.glob(os.path.join(RES, "runs", "*", "*", "split*_seed*", "*.json")):
        if os.path.basename(p) in ("fit.json", "alm_trace.json"):
            continue
        if not os.path.exists(os.path.join(os.path.dirname(p), "DONE")):
            continue  # run still in progress
        d = json.load(open(p))
        rows.append({k: v for k, v in d.items() if not isinstance(v, (dict, list))})
    df = pd.DataFrame(rows)
    tr = [json.load(open(p)) for p in glob.glob(os.path.join(RES, "trtr", "*.json"))]
    return df, pd.DataFrame(tr)


def score(df, m):
    d, _ = METRICS[m]
    x = df[m].astype(float)
    if d == 0:
        return -np.abs(x - 0.5)
    if d == 2:
        return -np.abs(x - 1.0)
    return d * x  # higher is better


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def summarise(df):
    g = df.groupby(["dataset", "method"])
    out = []
    for (ds, m), sub in g:
        rec = {"dataset": ds, "method": m, "n_runs": len(sub)}
        for k in METRICS:
            if k in sub:
                x = sub[k].astype(float)
                rec[f"{k}_mean"], rec[f"{k}_sd"], rec[f"{k}_ci95"] = x.mean(), x.std(ddof=1), ci95(x)
        out.append(rec)
    return pd.DataFrame(out)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(np.nan_to_num(p, nan=1.0))
    adj = np.empty_like(p)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - r) * (1.0 if np.isnan(p[i]) else p[i])))
        adj[i] = running
    return adj


def rank_biserial(diff):
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    r = stats.rankdata(np.abs(diff))
    return float((r[diff > 0].sum() - r[diff < 0].sum()) / r.sum())


def wilcoxon_table(piv, ref, others):
    rows = []
    for m in others:
        pair = piv[[ref, m]].dropna()
        if len(pair) < 5:
            continue
        diff = pair[ref].values - pair[m].values
        try:
            p = stats.wilcoxon(pair[ref].values, pair[m].values).pvalue
        except ValueError:
            p = 1.0
        rows.append({"reference": ref, "method": m, "n_datasets": len(pair), "mean_diff": diff.mean(),
                     "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()), "p": p,
                     "rank_biserial": rank_biserial(diff)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"].values)
    return out


def friedman(piv):
    piv = piv.dropna()
    if piv.shape[0] < 3 or piv.shape[1] < 3:
        return None
    stat, p = stats.friedmanchisquare(*[piv[c].values for c in piv.columns])
    ranks = piv.rank(axis=1, ascending=False).mean()
    res = {"stat": stat, "p": p, "n_datasets": piv.shape[0], "avg_rank": ranks.sort_values().to_dict()}
    try:
        import scikit_posthocs as sp
        nem = sp.posthoc_nemenyi_friedman(piv.values)
        nem.index = nem.columns = piv.columns
        res["nemenyi"] = nem
    except Exception as e:
        res["nemenyi_error"] = str(e)
    return res


def fmt(mean, sd, best=False):
    if np.isnan(mean):
        return "--"
    s = f"{mean:.3f}$\\pm${sd:.3f}" if not np.isnan(sd) else f"{mean:.3f}"
    return f"\\textbf{{{s}}}" if best else s


def latex_table(summ, methods, metric, path, caption, label):
    d, _ = METRICS[metric]
    sub = summ[summ.method.isin(methods)]
    datasets = sorted(sub.dataset.unique())
    lines = ["\\begin{table*}[t]", "\\centering", "\\scriptsize", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", "\\resizebox{\\textwidth}{!}{%",
             "\\begin{tabular}{l" + "c" * len(methods) + "}", "\\toprule",
             "Dataset & " + " & ".join(m.replace("_", "\\_") for m in methods) + " \\\\", "\\midrule"]
    for ds in datasets:
        r = sub[sub.dataset == ds].set_index("method")
        vals = {m: (r.loc[m, f"{metric}_mean"], r.loc[m, f"{metric}_sd"]) if m in r.index else (np.nan, np.nan)
                for m in methods}
        key = {m: (-abs(v[0] - 0.5) if d == 0 else -abs(v[0] - 1) if d == 2 else d * v[0])
               for m, v in vals.items() if not np.isnan(v[0])}
        best = max(key, key=key.get) if key else None
        lines.append(ds.replace("_", " ").title() + " & " +
                     " & ".join(fmt(*vals[m], best=(m == best)) for m in methods) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    os.makedirs(TAB, exist_ok=True)
    df, tr = load()
    if df.empty:
        print("no results yet")
        return
    df.to_csv(os.path.join(TAB, "all_runs_long.csv"), index=False)
    summ = summarise(df)
    summ.to_csv(os.path.join(TAB, "summary_mean_sd_ci.csv"), index=False)
    if len(tr):
        tr.groupby("dataset").mean(numeric_only=True).to_csv(os.path.join(TAB, "trtr_reference.csv"))
    methods = sorted(df.method.unique())
    print("runs per method:\n", df.groupby("method").size().to_string())

    # per-dataset mean of the "higher is better" score of each metric
    tests = []
    for m in METRICS:
        if m not in df:
            continue
        df["_s"] = score(df, m)
        piv = df.pivot_table(index="dataset", columns="method", values="_s", aggfunc="mean")
        fr = friedman(piv)
        if fr:
            tests.append({"metric": m, "friedman_stat": fr["stat"], "friedman_p": fr["p"],
                          "n_datasets": fr["n_datasets"],
                          "avg_rank_" + MAIN: fr["avg_rank"].get(MAIN, np.nan),
                          "best_avg_rank_method": min(fr["avg_rank"], key=fr["avg_rank"].get)})
            pd.Series(fr["avg_rank"]).to_csv(os.path.join(TAB, f"avg_rank__{m}.csv"))
            if "nemenyi" in fr:
                fr["nemenyi"].to_csv(os.path.join(TAB, f"nemenyi__{m}.csv"))
        if MAIN in piv:
            w = wilcoxon_table(piv, MAIN, [c for c in piv.columns if c != MAIN])
            if len(w):
                w.insert(0, "metric", m)
                w.to_csv(os.path.join(TAB, f"wilcoxon_{MAIN}__{m}.csv"), index=False)
        # model-agnostic: G+<variant> vs G, and minimal intervention vs ALM
        rows = []
        for g in sorted({c.split("+")[0] for c in piv.columns if c.endswith("+psygentab")}):
            for ref, other in [(f"{g}+psygentab", g), (f"{g}+psygentab_min", g),
                               (f"{g}+psygentab_min", f"{g}+psygentab")]:
                if ref in piv and other in piv:
                    rows.append(wilcoxon_table(piv, ref, [other]))
        rows = [r for r in rows if len(r)]
        if rows:
            w = pd.concat(rows)
            w["p_holm"] = holm(w["p"].values)
            w.insert(0, "metric", m)
            w.to_csv(os.path.join(TAB, f"wilcoxon_agnostic__{m}.csv"), index=False)
    df.drop(columns="_s", inplace=True, errors="ignore")
    pd.DataFrame(tests).to_csv(os.path.join(TAB, "friedman_summary.csv"), index=False)

    core = [m for m in ["rtf", "rtf+filter", "rtf+psygentab", "rtf+psygentab_min", "ctabgan_plus",
                        "ctabgan_plus+filter", "ctabgan_plus+psygentab", "ctabgan_plus+psygentab_min"]
            + [f"ctabgan_plus_{v}_eps{e}" for v in ("dp", "dpgp") for e in (1, 3, 10)]
            + [f"pategan_eps{e}" for e in (1, 3, 10)]
            if m in methods]
    for metric, cap in [("P", "Holdout-calibrated privacy score $P$ (test split; higher is better)."),
                        ("tstr_catboost_auroc", "TSTR AUROC with CatBoost on the real test split."),
                        ("mia_domias_auc", "DOMIAS membership-inference AUC (0.5 = no leakage)."),
                        ("mia_dist_auc", "Distance-based membership-inference AUC (0.5 = no leakage)."),
                        ("c2st_auc", "C2ST detection AUC (0.5 = indistinguishable)."),
                        ("aia_risk", "Attribute-inference risk (Anonymeter, control-corrected).")]:
        if f"{metric}_mean" in summ:
            latex_table(summ, core, metric, os.path.join(TAB, f"table_core__{metric}.tex"), cap,
                        f"tab:core_{metric}")
    print("wrote", TAB)


if __name__ == "__main__":
    main()
