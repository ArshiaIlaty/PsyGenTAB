"""Generate every results table and quoted number of the manuscript.

Writes into the paper folder:
  numbers.tex        \\newcommand macros for every number quoted in the text
  tab_datasets.tex   tab_main.tex  tab_stats.tex  tab_rtf_datasets.tex
  tab_tapas.tex      tab_fairness.tex  tab_leaderboard.tex  tab_ablation.tex
  tab_cost.tex
and copies the paper figures into figures/. Run after scripts.aggregate,
scripts.ablations and scripts.figures:
    python -m scripts.paper_tables
"""
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd

from psygentab.datasets import describe_all
from scripts import ci as CI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
TAB = os.path.join(RES, "tables")
ABL = os.path.join(RES, "ablations")
PAPER = os.path.join(ROOT, "2026_JBHI___PSyGenTAB__Privacy_Preserving_Synthetic_Health_Data")

GENS = ["rtf", "ctabgan_plus", "bayesian_network", "arf", "tvae", "ctgan", "tabddpm"]
GL = {"rtf": "REaLTabFormer", "ctabgan_plus": "CTAB-GAN+", "bayesian_network": "BayesNet", "arf": "ARF",
      "tvae": "TVAE", "ctgan": "CTGAN", "tabddpm": "TabDDPM"}
DP = [("ctabgan_plus_dp", "DP-CTAB-GAN+ (clip)"), ("ctabgan_plus_dpgp", "DP-CTAB-GAN+ (GP)"), ("pategan", "PATE-GAN")]
DSL = {"breast_cancer": "Breast Cancer", "diabetes": "Diabetes", "heart_failure": "Heart Failure",
       "hypothyroid": "Hypothyroid", "liver_disorders": "Liver Disorders", "obesity": "Obesity",
       "parkinsons": "Parkinson's", "adult": "Adult", "pir_vision": "PIR Vision", "vn_banking": "VN Banking"}
ORDER = list(DSL)
# metric -> direction: +1 higher better, -1 lower better, 0 ideal 0.5, 2 ideal 1
DIR = {"domias_adv": -1, "tstr_catboost_auroc": 1, "tstr_catboost_f1_macro": 1, "alpha_precision": 1, "beta_recall": 1,
       "c2st_auc": 0, "P": 1, "mia_domias_auc": 0, "mia_dist_auc": 0, "aia_risk": -1, "link_risk": -1,
       "repr_parity": 1, "subgroup_util_ratio_min": 1, "minority_class_ratio": 2, "rare_category_coverage": 1,
       "entropy_preservation": 1, "tail_preservation": 2, "js_mean": -1}

macros = {}


def mac(name, value, fmt="{:.3f}"):
    macros[name] = fmt.format(value) if not isinstance(value, str) else value


def better(m, a, b):
    d = DIR[m]
    if d == 0:
        return abs(a - 0.5) < abs(b - 0.5)
    if d == 2:
        return abs(a - 1) < abs(b - 1)
    return d * a > d * b


def load():
    df = pd.read_csv(os.path.join(TAB, "all_runs_long.csv"))
    df["gen"] = df.method.str.split("+").str[0]
    df["variant"] = df.method.str.split("+").str[1].fillna("base")
    return df


def dmean(df, method, m):
    """Mean over datasets of per-dataset means (and SD across datasets)."""
    x = df[df.method == method].groupby("dataset")[m].mean()
    return x.mean(), x.std(ddof=1), len(x)


def bold_head(s, sep="{"):
    """Bold the point estimate of a 'value{\\scriptsize ...}' cell, not its interval."""
    head = s.split(sep)[0]
    return f"\\textbf{{{head}}}" + s[len(head):]


def write(name, body):
    with open(os.path.join(PAPER, name), "w") as f:
        f.write(body)


def f3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


# ------------------------------------------------------------------ tables

def tab_datasets():
    d = describe_all().set_index("dataset").loc[ORDER]
    rows = []
    for k, r in d.iterrows():
        rows.append(f"{DSL[k]} & {r['domain']} & {int(r['N']):,} & {int(r['cont'])} & {int(r['cat'])} & "
                    f"{int(r['classes'])} & {100 * r['minority']:.1f}\\% \\\\")
        if k == "parkinsons":
            rows.append("\\midrule")
    write("tab_datasets.tex", r"""\begin{table}[t]
\centering\footnotesize
\caption{Benchmarks. $N$: rows; Cont./Cat.: continuous/categorical features (target excluded); Min.: minority-class share. ID columns removed; Liver Disorders uses the recommended \texttt{drinks}$>3$ target~\cite{mcdermott2016diagnosing}.}
\label{tab:datasets}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llrrrrr@{}}
\toprule
Dataset & Domain & $N$ & Cont. & Cat. & Cls. & Min. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


MAIN_COLS = [("tstr_catboost_auroc", "AUROC$\\uparrow$"), ("tstr_catboost_f1_macro", "F1$\\uparrow$"),
             ("alpha_precision", "$\\alpha$-P$\\uparrow$"), ("c2st_auc", "C2ST$\\rightarrow$.5"),
             ("P", "$P\\uparrow$"), ("mia_domias_auc", "DOMIAS$\\rightarrow$.5"),
             ("mia_dist_auc", "D-MIA$\\rightarrow$.5"), ("aia_risk", "AIA$\\downarrow$"),
             ("link_risk", "Link$\\downarrow$")]


def tab_main(df, trtr):
    """Mean over datasets with hierarchical-bootstrap 95% CI (shown as ± half-width)."""
    lines = []
    for g in GENS:
        for var, lab in [("base", GL[g]), ("psygentab", "\\quad + PSyGenTAB")]:
            m = g if var == "base" else f"{g}+psygentab"
            other = f"{g}+psygentab" if var == "base" else g
            cells = []
            for c, _ in MAIN_COLS:
                p, lo, hi = CI.mean_ci(df, m, c)
                b = dmean(df, other, c)[0]
                s = CI.pm(p, lo, hi)
                cells.append(bold_head(s) if better(c, p, b) else s)
            lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        lines.append("\\addlinespace[2pt]")
    lines.append("\\midrule")
    for fam, lab in DP:
        for e in (1, 3, 10):
            m = f"{fam}_eps{e}"
            cells = [CI.pm(*CI.mean_ci(df, m, c)) for c, _ in MAIN_COLS]
            lines.append(f"{lab}, $\\varepsilon$={e} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    tr = pd.read_csv(os.path.join(TAB, "trtr_reference.csv"))
    t_auc = CI.t_ci(tr.trtr_catboost_auroc)
    t_f1 = CI.t_ci(tr.trtr_catboost_f1_macro)
    lines.append(f"Real data (TRTR) & {CI.pm(*t_auc)} & {CI.pm(*t_f1)} & "
                 + " & ".join(["--"] * (len(MAIN_COLS) - 2)) + " \\\\")
    write("tab_main.tex", r"""\begin{table*}[t]
\centering\scriptsize
\caption{Model-agnostic results: every generator with and without PSyGenTAB ($P_{\min}=0.90$), and differentially private baselines. Mean over the 10 datasets of per-dataset means, $\pm$ half-width of the 95\% hierarchical-bootstrap confidence interval (5{,}000 resamples of datasets and, within datasets, of runs). REaLTabFormer, CTAB-GAN+ and DP-CTAB-GAN+: 5 splits $\times$ 3 seeds per dataset; all others: 5 splits. Utility: TSTR with CatBoost on the untouched real test split; C2ST: detection AUC; $P$: holdout-calibrated privacy score; DOMIAS and D-MIA: membership-inference AUCs; AIA and Link: Anonymeter attribute-inference and linkability risks. Arrows give the ideal direction; $\rightarrow$.5 means closer to 0.5 is better. Bold: better of each raw/PSyGenTAB pair. Intervals for means over datasets include between-dataset heterogeneity; the paired effect of PSyGenTAB is tested in Table~\ref{tab:stats}.}
\label{tab:main}
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}l""" + "c" * len(MAIN_COLS) + r"""@{}}
\toprule
& \multicolumn{4}{c}{Utility \& fidelity} & \multicolumn{5}{c}{Privacy} \\
\cmidrule(lr){2-5}\cmidrule(l){6-10}
Method & """ + " & ".join(l for _, l in MAIN_COLS) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


STAT_METS = [("mia_domias_auc", "DOMIAS"), ("mia_dist_auc", "Distance MIA"), ("P", "$P$"),
             ("tstr_catboost_auroc", "TSTR AUROC"), ("repr_parity", "Repr. parity")]


def tab_stats(df):
    rows = []
    W = {m: pd.read_csv(os.path.join(TAB, f"wilcoxon_agnostic__{m}.csv")) for m, _ in STAT_METS}
    for g in GENS:
        cells = []
        for m, _ in STAT_METS:
            p, lo, hi = CI.paired_diff_ci(df, f"{g}+psygentab", g, m, DIR[m])
            w = W[m]
            r = w[(w.reference == f"{g}+psygentab") & (w.method == g)]
            star = "$^{*}$" if (not r.empty and r.iloc[0].p_holm < 0.05) else ""
            cells.append(CI.bracket(p, lo, hi, sign=True) + star)
        rows.append(f"{GL[g]} & " + " & ".join(cells) + " \\\\")
    write("tab_stats.tex", r"""\begin{table*}[t]
\centering\footnotesize
\caption{Paired effect of PSyGenTAB per generator: mean improvement over the raw generator with its 95\% hierarchical-bootstrap confidence interval (runs paired on dataset, split and seed; positive = PSyGenTAB better in the metric's ideal direction, attack AUCs measured as distance from 0.5). $^{*}$: Wilcoxon signed-rank test across the 10 datasets significant at $p<0.05$ after Holm correction across generators and variants.}
\label{tab:stats}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}l""" + "c" * len(STAT_METS) + r"""@{}}
\toprule
Generator & """ + " & ".join(l for _, l in STAT_METS) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def tab_rtf_datasets(df):
    """Per dataset: mean over the 15 runs with Student-t 95% CI. DOMIAS is reported
    as leakage |AUC-0.5|, the quantity the Wilcoxon tests compare."""
    rows = []
    df = df.assign(domias_adv=(df.mia_domias_auc - 0.5).abs())
    for d in ORDER:
        cells = []
        for m in ["tstr_catboost_auroc", "domias_adv", "aia_risk"]:
            v = {meth: CI.t_ci(df[(df.method == meth) & (df.dataset == d)][m]) for meth in ["rtf", "rtf+psygentab"]}
            a, b = v["rtf"], v["rtf+psygentab"]
            sb = CI.pm(*b)
            if better(m, b[0], a[0]):
                head = sb.split("{")[0]
                sb = f"\\textbf{{{head}}}" + sb[len(head):]
            cells += [CI.pm(*a), sb]
        rows.append(f"{DSL[d]} & " + " & ".join(cells) + " \\\\")
    write("tab_rtf_datasets.tex", r"""\begin{table*}[t]
\centering\footnotesize
\caption{REaLTabFormer without (raw) and with PSyGenTAB on every dataset: mean of 15 runs $\pm$ half-width of the Student-$t$ 95\% confidence interval. DOMIAS leak.: membership advantage $|\mathrm{AUC}-0.5|$. Bold: PSyGenTAB better.}
\label{tab:rtf_datasets}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lcccccc@{}}
\toprule
& \multicolumn{2}{c}{AUROC$\uparrow$} & \multicolumn{2}{c}{DOMIAS leak.$\downarrow$} & \multicolumn{2}{c}{AIA$\downarrow$} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}
Dataset & raw & PSyGenTAB & raw & PSyGenTAB & raw & PSyGenTAB \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def _tapas_boot(feat_path, n_targets=5, B=2000, seed=CI.SEED):
    """Refit the per-target attackers exactly as scripts/tapas_attack.py and
    bootstrap the 20 evaluation worlds (jointly across targets) for the mean AUC."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    R = np.load(feat_path)["rows"]
    kinds, labels = R[:, 1], R[:, 3:3 + n_targets]
    nf = (R.shape[1] - 3 - n_targets) // (2 * n_targets)
    F = R[:, 3 + n_targets:]
    out = {}
    for name, Fx in [("raw", F[:, : n_targets * nf].reshape(len(R), n_targets, nf)),
                     ("psygentab", F[:, n_targets * nf:].reshape(len(R), n_targets, nf))]:
        sh, ev = kinds == 0, kinds == 1
        scores, ys = [], []
        for t in range(n_targets):
            if len(np.unique(labels[sh, t])) < 2 or len(np.unique(labels[ev, t])) < 2:
                continue
            clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=t).fit(Fx[sh, t], labels[sh, t])
            scores.append(clf.predict_proba(Fx[ev, t])[:, 1])
            ys.append(labels[ev, t])
        point = np.mean([roc_auc_score(y, s) for y, s in zip(ys, scores)])
        rng = np.random.default_rng(seed)
        n_ev = len(ys[0])
        boots = []
        for _ in range(B):
            idx = rng.integers(0, n_ev, n_ev)
            aucs = [roc_auc_score(y[idx], s[idx]) for y, s in zip(ys, scores) if len(np.unique(y[idx])) == 2]
            if aucs:
                boots.append(np.mean(aucs))
        out[name] = (point, *np.percentile(boots, [2.5, 97.5]))
    return out


def tab_tapas():
    """Generator x dataset matrix of mean shadow-model AUC (raw -> PSyGenTAB);
    per-target values and bootstrap CIs are in the Supplementary Material."""
    gens = ["rtf", "ctabgan_plus", "bayesian_network", "arf", "tvae", "ctgan", "tabddpm"]
    TOL = 0.01  # change in |AUC-0.5| below this is treated as no change
    improved = same = worse = n = 0
    res = {}
    for d in ORDER:
        for g in gens:
            fp = os.path.join(RES, "tapas", f"{d}__{g}__features.npz")
            if os.path.exists(fp) and os.path.exists(fp.replace("__features.npz", ".json")):
                res[(d, g)] = _tapas_boot(fp)
    rows = []
    for g in gens:
        cells = []
        for d in ORDER:
            if (d, g) not in res:
                cells.append("{\\tiny pending}")
                continue
            (a, alo, ahi), (b, blo, bhi) = res[(d, g)]["raw"], res[(d, g)]["psygentab"]
            n += 1
            gain = abs(a - 0.5) - abs(b - 0.5)
            improved += gain > TOL
            same += abs(gain) <= TOL
            worse += gain < -TOL
            sb = f"{b:.2f}"
            cells.append(f"{a:.2f}$\\to$" + (f"\\textbf{{{sb}}}" if gain > TOL else sb))
        rows.append(f"{GL[g]} & " + " & ".join(cells) + " \\\\")
    for key, (dd, gg) in {"TapasHfRtf": ("heart_failure", "rtf"), "TapasObTabddpm": ("obesity", "tabddpm"),
                          "TapasBcBn": ("breast_cancer", "bayesian_network"),
                          "TapasHfBn": ("heart_failure", "bayesian_network")}.items():
        if (dd, gg) in res:
            (a, alo, ahi), (b, blo, bhi) = res[(dd, gg)]["raw"], res[(dd, gg)]["psygentab"]
            mac(key + "Raw", a, "{:.2f}")
            mac(key + "Psy", b, "{:.2f}")
            mac(key + "RawCI", f"[{alo:.2f}, {ahi:.2f}]", "{}")
            mac(key + "PsyCI", f"[{blo:.2f}, {bhi:.2f}]", "{}")
    mac("TapasImproved", str(improved), "{}")
    mac("TapasSame", str(same), "{}")
    mac("TapasWorse", str(worse), "{}")
    mac("TapasTotal", str(n), "{}")
    short = {"breast_cancer": "BrCa", "diabetes": "Diab", "heart_failure": "HeFa", "hypothyroid": "Hypo",
             "liver_disorders": "Liver", "obesity": "Obes", "parkinsons": "Park", "adult": "Adult",
             "pir_vision": "PIR", "vn_banking": "VNB"}
    write("tab_tapas.tex", r"""\begin{table*}[t]
\centering\scriptsize
\caption{Shadow-model membership inference (TAPAS-style; 40 shadow and 20 evaluation worlds; the 5 most isolated training records of each dataset). Mean target AUC, raw generator $\rightarrow$ PSyGenTAB on the same fitted worlds; 0.5 = no leakage. Bold: membership advantage $|\mathrm{AUC}-0.5|$ reduced by more than 0.01. Bootstrap CIs over evaluation worlds and per-target AUCs are given in the Supplementary Material.}
\label{tab:tapas}
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}l""" + "c" * len(ORDER) + r"""@{}}
\toprule
Generator & """ + " & ".join(short[d] for d in ORDER) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


FAIR_COLS = [("repr_parity", "Parity$\\uparrow$"), ("subgroup_util_ratio_min", "Worst-sub.$\\uparrow$"),
             ("minority_class_ratio", "Minority$\\rightarrow$1"), ("rare_category_coverage", "Rare cov.$\\uparrow$"),
             ("entropy_preservation", "Entropy$\\uparrow$"), ("tail_preservation", "Tails$\\rightarrow$1"),
             ("js_mean", "JS$\\downarrow$")]


def tab_fairness(df):
    lines = []
    for g in GENS:
        for var, lab in [("base", GL[g]), ("psygentab", "\\quad + PSyGenTAB")]:
            m = g if var == "base" else f"{g}+psygentab"
            other = f"{g}+psygentab" if var == "base" else g
            cells = []
            for c, _ in FAIR_COLS:
                p, lo, hi = CI.mean_ci(df, m, c)
                b = dmean(df, other, c)[0]
                s = CI.pm(p, lo, hi)
                if better(c, p, b):
                    head = s.split("{")[0]
                    s = f"\\textbf{{{head}}}" + s[len(head):]
                cells.append(s)
            lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        lines.append("\\addlinespace[2pt]")
    write("tab_fairness.tex", r"""\begin{table*}[t]
\centering\footnotesize
\caption{Fairness and diversity: mean over datasets $\pm$ half-width of the 95\% hierarchical-bootstrap CI. Subgroups are the levels (with $\geq$30 real rows) of every categorical covariate. Parity: mean $\min(p_{\text{syn}}/p_{\text{real}}, p_{\text{real}}/p_{\text{syn}})$; Worst-sub.: worst subgroup TSTR balanced accuracy relative to a real-trained model; Minority: minority-class share ratio; Rare cov.: share of categories with $<5\%$ prevalence reproduced; Entropy: per-column entropy ratio; Tails: synthetic mass beyond the real 5th/95th percentiles relative to real (columns whose real tails hold $\geq$2\% of rows); JS: mean Jensen--Shannon divergence. Bold: better of each pair.}
\label{tab:fairness}
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}l""" + "c" * len(FAIR_COLS) + r"""@{}}
\toprule
Method & """ + " & ".join(l for _, l in FAIR_COLS) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def label(m):
    from scripts.leaderboard import label as L
    return L(m).replace("ε", "$\\varepsilon$").replace("α", "$\\alpha$")


def _rank_boot(B=CI.B, seed=CI.SEED):
    """Average ranks with a dataset-bootstrap 95% CI (ranks are computed per
    dataset; datasets are resampled)."""
    from scripts.leaderboard import FAIRNESS, PRIVACY, UTILITY, compute
    lb, dm = compute()
    per = {}
    for grp, mets in [("utility", UTILITY), ("privacy", PRIVACY), ("fairness", FAIRNESS)]:
        rk = []
        for m, d in mets.items():
            piv = dm[m].unstack("method")
            score = -(piv - 0.5).abs() if d == 0 else -(piv - 1.0).abs() if d == 2 else d * piv
            rk.append(score.rank(axis=1, ascending=False))
        per[grp] = sum(rk) / len(rk) if grp != "fairness" else pd.concat(rk).groupby(level=0).mean()
    per["overall"] = (per["utility"] + per["privacy"]) / 2
    rng = np.random.default_rng(seed)
    out = {}
    for grp, R in per.items():
        R = R.loc[ORDER] if set(ORDER) <= set(R.index) else R
        vals = R.to_numpy()
        idx = rng.integers(0, len(R), size=(B, len(R)))
        boots = np.nanmean(vals[idx], axis=1)              # (B, methods)
        lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
        out[grp] = pd.DataFrame({"mean": np.nanmean(vals, 0), "lo": lo, "hi": hi}, index=R.columns)
    return lb, out


def tab_leaderboard():
    lb, rb = _rank_boot()
    order = rb["overall"]["mean"].sort_values().index
    def cell(grp, m):
        r = rb[grp].loc[m]
        return f"{r['mean']:.1f}\\,{{\\scriptsize[{r.lo:.1f}, {r.hi:.1f}]}}"
    rows = [f"{i + 1} & {label(m)} & {cell('overall', m)} & {cell('utility', m)} & {cell('privacy', m)} & "
            f"{cell('fairness', m)} \\\\" for i, m in enumerate(order[:12])]
    rows.append("\\midrule")
    for m in ["pategan_eps10", "ctabgan_plus_dp_eps10", "ctabgan_plus_dpgp_eps10"]:
        rows.append(f"{list(order).index(m) + 1} & {label(m)} & {cell('overall', m)} & {cell('utility', m)} & "
                    f"{cell('privacy', m)} & {cell('fairness', m)} \\\\")
    mac("NMethods", str(len(order)), "{}")
    r = rb["overall"].loc["rtf+psygentab"]
    mac("RtfPsyRankCI", f"{r['mean']:.1f} [{r.lo:.1f}, {r.hi:.1f}]", "{}")
    write("tab_leaderboard.tex", r"""\begin{table*}[t]
\centering\footnotesize
\caption{Average rank over the 10 datasets among all """ + str(len(order)) + r""" methods (1 = best), with 95\% bootstrap CIs over datasets. Utility: TSTR AUROC, macro-F1, $\alpha$-precision; privacy: $P$, distance MIA, DOMIAS, attribute inference; fairness: the Table~\ref{tab:fairness} metrics. Overall: mean of utility and privacy ranks.}
\label{tab:leaderboard}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}rlcccc@{}}
\toprule
\# & Method & Overall & Utility & Privacy & Fairness \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def tab_ablation():
    a1 = pd.read_csv(os.path.join(ABL, "A1_pmin_sweep.csv"))
    a2 = pd.read_csv(os.path.join(ABL, "A2_outer_iters.csv"))
    a3 = pd.read_csv(os.path.join(ABL, "A3_per_run.csv"))
    sels = [("alm", "ALM (PSyGenTAB)"), ("min_intervention", "Minimal intervention"),
            ("penalty_lam1", "Fixed penalty, $\\lambda$=1"), ("penalty_lam10", "Fixed penalty, $\\lambda$=10"),
            ("oracle", "Oracle constrained"), ("unconstrained", "Unconstrained (max $U$)")]
    pms = [0.80, 0.90, 0.95]
    rows = []
    for s, lab in sels:
        cells = []
        for pm in pms:
            sub = a1[(a1.selector == s) & np.isclose(a1.p_min, pm)]
            cells += [f"{100 * sub.feasible_val.mean():.0f}", f"{100 * sub.feasible_test.mean():.0f}",
                      f"{sub.U_test.mean():.3f}"]
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    alm90 = a1[(a1.selector == "alm") & np.isclose(a1.p_min, 0.9)]
    orc90 = a1[(a1.selector == "oracle") & np.isclose(a1.p_min, 0.9)]
    mi90 = a1[(a1.selector == "min_intervention") & np.isclose(a1.p_min, 0.9)]
    p1 = a1[(a1.selector == "penalty_lam1") & np.isclose(a1.p_min, 0.9)]
    un = a1[(a1.selector == "unconstrained") & np.isclose(a1.p_min, 0.9)]
    def wil(x):
        k, n = int(x.sum()), int(x.count())
        p_, lo, hi = CI.wilson(k, n)
        return f"[{100 * lo:.0f}, {100 * hi:.0f}]"
    mac("AblAlmFeasValCI", wil(alm90.feasible_val), "{}")
    mac("AblAlmFeasTestCI", wil(alm90.feasible_test), "{}")
    mac("AblMinFeasTestCI", wil(mi90.feasible_test), "{}")
    mac("AblPenFeasValCI", wil(p1.feasible_val), "{}")
    worst = 0.0
    for _, sub in a1.groupby(["selector", "p_min"]):
        for g in (sub.feasible_val, sub.feasible_test):
            p_, lo, hi = CI.wilson(int(g.sum()), int(g.count()))
            worst = max(worst, hi - p_, p_ - lo)
    mac("AblWilsonMax", 100 * worst, "{:.0f}")
    mac("AblAlmFeasVal", 100 * alm90.feasible_val.mean(), "{:.0f}")
    mac("AblAlmFeasTest", 100 * alm90.feasible_test.mean(), "{:.0f}")
    mac("AblOracleFeasVal", 100 * orc90.feasible_val.mean(), "{:.0f}")
    mac("AblMinFeasTest", 100 * mi90.feasible_test.mean(), "{:.0f}")
    mac("AblPenFeasVal", 100 * p1.feasible_val.mean(), "{:.0f}")
    mac("AblUncFeasTest", 100 * un.feasible_test.mean(), "{:.0f}")
    mac("AblNRuns", str(alm90.groupby(["dataset", "generator", "run"]).ngroups), "{}")
    same = (alm90.reset_index(drop=True)[["U_test", "P_test"]].values ==
            orc90.reset_index(drop=True)[["U_test", "P_test"]].values).all(1).mean()
    mac("AblAlmEqOracle", 100 * same, "{:.0f}")
    k = a2.groupby("outer_iters")[["feasible_val", "feasible_test", "U_test"]].mean()
    krows = " & ".join(f"{int(K)}" for K in k.index)
    kval = " & ".join(f"{100 * v:.0f}" for v in k.feasible_val)
    ktest = " & ".join(f"{100 * v:.0f}" for v in k.feasible_test)
    mac("AblKOneFeasVal", 100 * k.feasible_val.iloc[0], "{:.0f}")
    mac("AblKFiveFeasVal", 100 * k.loc[5, "feasible_val"], "{:.0f}")
    mac("AblWeightSame", 100 * a3.same_selection.mean(), "{:.0f}")
    mac("AblWeightPsd", a3.P_test_sd.mean(), "{:.3f}")
    mac("AblWeightUsd", a3.U_test_sd.mean(), "{:.3f}")
    write("tab_ablation.tex", r"""\begin{table*}[t]
\centering\footnotesize
\caption{Ablation of the selection rule over all """ + macros["AblNRuns"] + r""" non-DP runs, recomputed offline from the cached candidate tables. Val/Test: share of runs (\%) whose selected release meets $P\geq P_{\min}$ on the calibration split (used to select) and on the untouched test split (95\% Wilson intervals extend at most """ + macros["AblWilsonMax"] + r""" points above or below each value); $U$: test utility. Bottom: ALM outer iterations $K$ at $P_{\min}=0.90$.}
\label{tab:ablation}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lccccccccc@{}}
\toprule
& \multicolumn{3}{c}{$P_{\min}=0.80$} & \multicolumn{3}{c}{$P_{\min}=0.90$} & \multicolumn{3}{c}{$P_{\min}=0.95$} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}
Selector & Val & Test & $U$ & Val & Test & $U$ & Val & Test & $U$ \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
\multicolumn{10}{@{}l}{Outer iterations $K$: \quad """ + krows.replace(" & ", ", ") + r"""} \\
\multicolumn{10}{@{}l}{\quad feasible on val (\%): """ + kval.replace(" & ", ", ") + r"""; \quad feasible on test (\%): """ + ktest.replace(" & ", ", ") + r"""} \\
\bottomrule
\end{tabular}
\end{table*}
""")


def tab_cost():
    rows = []
    for g in GENS + ["ctabgan_plus_dp_eps3", "ctabgan_plus_dpgp_eps3", "pategan_eps3"]:
        fs = [json.load(open(p)) for p in glob.glob(os.path.join(RES, "runs", "*", g, "*", "fit.json"))]
        if not fs:
            continue
        fit = np.median([f["fit_seconds"] for f in fs])
        samp = np.median([f["sample_seconds"] for f in fs])
        hw = "GPU (T4)" if g.startswith(("rtf", "ctabgan")) else "CPU"
        rows.append(f"{label(g) if '_eps' in g else GL[g]} & {hw} & {fit / 60:.1f} & {samp / 60:.1f} \\\\")
    write("tab_cost.tex", r"""\begin{table}[t]
\centering\footnotesize
\caption{Median wall-clock cost per run (minutes): generator fitting and drawing the PSyGenTAB candidate pools. PSyGenTAB adds only pool sampling and the candidate scoring (seconds to a few minutes per run).}
\label{tab:cost}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llrr@{}}
\toprule
Generator & Hardware & Fit & Sample \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def numbers(df, trtr):
    for g, k in [("rtf", "Rtf"), ("ctabgan_plus", "Ctab"), ("bayesian_network", "Bn"), ("arf", "Arf"),
                 ("tabddpm", "Ddpm"), ("tvae", "Tvae"), ("ctgan", "Ctgan")]:
        for var, s in [("", "Raw"), ("+psygentab", "Psy"), ("+psygentab_min", "Min")]:
            for m, mk in [("tstr_catboost_auroc", "Auroc"), ("mia_domias_auc", "Domias"), ("mia_dist_auc", "Dist"),
                          ("P", "P"), ("aia_risk", "Aia"), ("repr_parity", "Parity"),
                          ("subgroup_util_ratio_min", "Worst"), ("share_closer_train", "Share")]:
                if f"{g}{var}" in set(df.method):
                    mac(f"{k}{s}{mk}", dmean(df, f"{g}{var}", m)[0])
    mac("TrtrAuroc", trtr["trtr_catboost_auroc"])
    for g, k in [("rtf", "Rtf"), ("ctabgan_plus", "Ctab"), ("bayesian_network", "Bn"), ("arf", "Arf"),
                 ("tabddpm", "Ddpm"), ("tvae", "Tvae"), ("ctgan", "Ctgan")]:
        for m, mk in [("tstr_catboost_auroc", "Auroc"), ("mia_domias_auc", "Domias"), ("mia_dist_auc", "Dist"),
                      ("aia_risk", "Aia"), ("repr_parity", "Parity"), ("P", "P")]:
            pnt, lo, hi = CI.paired_diff_ci(df, f"{g}+psygentab", g, m, DIR[m])
            mac(f"{k}Diff{mk}", pnt, "{:+.3f}")
            mac(f"{k}Diff{mk}CI", f"[{lo:+.3f}, {hi:+.3f}]", "{}")
    for g, k in [("rtf", "Rtf"), ("ctabgan_plus", "Ctab")]:
        for var, sfx in [("", "Raw"), ("+psygentab", "Psy")]:
            for m, mk in [("tstr_catboost_auroc", "Auroc"), ("mia_domias_auc", "Domias"), ("aia_risk", "Aia")]:
                pnt, lo, hi = CI.mean_ci(df, f"{g}{var}", m)
                mac(f"{k}{sfx}{mk}CI", f"[{lo:.3f}, {hi:.3f}]", "{}")
    for fam, k in [("pategan", "Pate"), ("ctabgan_plus_dp", "DpClip"), ("ctabgan_plus_dpgp", "DpGp")]:
        for e, ek in [(1, "One"), (3, "Three"), (10, "Ten")]:
            mac(f"{k}{ek}Auroc", dmean(df, f"{fam}_eps{e}", "tstr_catboost_auroc")[0])
            mac(f"{k}{ek}Domias", dmean(df, f"{fam}_eps{e}", "mia_domias_auc")[0])
    # pooled leakage reduction across all non-DP generators (excess AUC over 0.5)
    for m, k in [("mia_dist_auc", "Dist"), ("mia_domias_auc", "Domias")]:
        raw = sum(abs(dmean(df, g, m)[0] - 0.5) for g in GENS)
        psy = sum(abs(dmean(df, f"{g}+psygentab", m)[0] - 0.5) for g in GENS)
        mac(f"Pooled{k}Reduction", 100 * (raw - psy) / raw, "{:.0f}")
    # DP valid-row rates
    v = []
    for p in glob.glob(os.path.join(RES, "runs", "*", "ctabgan_plus_dp*", "*", "fit.json")):
        si = json.load(open(p)).get("sample_info") or {}
        if "valid_rate" in si:
            v.append(si["valid_rate"])
    mac("DpValidRate", 100 * np.mean(v), "{:.0f}")
    fr = pd.read_csv(os.path.join(TAB, "friedman_summary.csv")).set_index("metric")
    mac("FriedmanP", "$p<10^{-%d}$" % int(-np.log10(fr.loc[["tstr_catboost_auroc", "mia_domias_auc", "P"], "friedman_p"].max())), "{}")
    n_rel = len(df)
    mac("NReleases", f"{n_rel:,}", "{}")
    mac("NRuns", f"{df.groupby(['dataset', 'generator', 'split', 'seed']).ngroups:,}", "{}")
    lines = ["% Auto-generated by scripts/paper_tables.py -- do not edit by hand."]
    for k, v in sorted(macros.items()):
        lines.append(f"\\newcommand{{\\num{k}}}{{{v}}}")
    write("numbers.tex", "\n".join(lines) + "\n")


def figures():
    os.makedirs(os.path.join(PAPER, "figures"), exist_ok=True)
    for f in ["fig_privacy_utility_map", "fig_model_agnostic", "fig_pareto_pmin", "fig_constraint_iterations",
              "fig_dp_epsilon", "fig_delta_heatmap", "fig_leaderboard", "fig_weight_sensitivity"]:
        shutil.copy2(os.path.join(RES, "figures", f + ".pdf"), os.path.join(PAPER, "figures", f + ".pdf"))


def main():
    df = load()
    trtr = pd.read_csv(os.path.join(TAB, "trtr_reference.csv")).mean(numeric_only=True)
    tab_datasets()
    tab_main(df, trtr)
    tab_stats(df)
    tab_rtf_datasets(df)
    tab_tapas()
    tab_fairness(df)
    tab_leaderboard()
    tab_ablation()
    tab_cost()
    numbers(df, trtr)
    figures()
    print(f"wrote tables + numbers.tex ({len(macros)} macros) into {PAPER}")


if __name__ == "__main__":
    main()
