"""Supplementary material: long per-dataset tables, per-generator ablations,
full shadow-model results, DP diagnostics and extra figures.

Writes supp_*.tex and supplementary.tex into the paper folder:
    python -m scripts.supplement
then compile supplementary.tex. Every value comes from results/; intervals are
95% (Student-t over runs unless stated otherwise).
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from scripts import ci as CI
from scripts.paper_tables import DIR, DSL, GENS, GL, ORDER, PAPER, RES, TAB, _tapas_boot, load

ABL = os.path.join(RES, "ablations")
VARIANTS = [("", "raw"), ("+filter", "+ DCR filter"), ("+psygentab", "+ PSyGenTAB"),
            ("+psygentab_min", "+ PSyGenTAB (min.)")]
DPM = [(f"ctabgan_plus_dp_eps{e}", f"DP-CTAB-GAN+ (clip) $\\varepsilon$={e}") for e in (1, 3, 10)] + \
      [(f"ctabgan_plus_dpgp_eps{e}", f"DP-CTAB-GAN+ (GP) $\\varepsilon$={e}") for e in (1, 3, 10)] + \
      [(f"pategan_eps{e}", f"PATE-GAN $\\varepsilon$={e}") for e in (1, 3, 10)]


def methods():
    out = []
    for g in GENS:
        for suf, lab in VARIANTS:
            out.append((g + suf, GL[g] if suf == "" else "\\quad " + lab))
    return out + DPM


def cell(x, d=3):
    if x.notna().sum() == 0:
        return "n/a"
    p, lo, hi = CI.t_ci(x)
    return CI.pm(p, lo, hi, d=d)


def write(name, body):
    with open(os.path.join(PAPER, name), "w") as f:
        f.write(body)


def long_table(df, cols, caption, label, fname, landscape=True):
    """One table per dataset: every method x the given metrics, mean +- 95% CI."""
    parts = []
    head = " & ".join(h for _, h in cols)
    for d in ORDER:
        sub = df[df.dataset == d]
        rows = []
        for m, lab in methods():
            s = sub[sub.method == m]
            if s.empty:
                continue
            rows.append(f"{lab} & {len(s)} & " + " & ".join(cell(s[c]) if c in s else "--" for c, _ in cols) + " \\\\")
            if m.endswith("+psygentab_min") or m == "ctabgan_plus_dpgp_eps10":
                rows.append("\\addlinespace[1pt]")
        body = (r"""\begin{table}[p]
\centering\scriptsize
\caption{""" + caption + f" \\textbf{{{DSL[d]}}}." + r"""}
\label{""" + f"{label}:{d}" + r"""}
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{@{}lr""" + "c" * len(cols) + r"""@{}}
\toprule
Method & $n$ & """ + head + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
        parts.append(("\\begin{landscape}\n" + body + "\\end{landscape}\n") if landscape else body)
    write(fname, "\n\\clearpage\n".join(parts) + "\n")


S1 = [("tstr_catboost_auroc", "AUROC$\\uparrow$"), ("tstr_catboost_f1_macro", "F1$\\uparrow$"),
      ("tstr_catboost_bal_acc", "Bal.acc.$\\uparrow$"), ("alpha_precision", "$\\alpha$-P$\\uparrow$"),
      ("beta_recall", "$\\beta$-R$\\uparrow$"), ("c2st_auc", "C2ST$\\rightarrow$.5"), ("P", "$P\\uparrow$"),
      ("mia_domias_auc", "DOMIAS$\\rightarrow$.5"), ("mia_dist_auc", "D-MIA$\\rightarrow$.5"),
      ("aia_risk", "AIA$\\downarrow$"), ("link_risk", "Link$\\downarrow$")]
S2 = [("marginal_err", "Marg.err$\\downarrow$"), ("wasserstein_norm", "W$_1\\downarrow$"),
      ("assoc_err", "Assoc.err$\\downarrow$"), ("js_mean", "JS$\\downarrow$"), ("prdc_precision", "Prec.$\\uparrow$"),
      ("prdc_recall", "Rec.$\\uparrow$"), ("prdc_density", "Dens.$\\uparrow$"), ("prdc_coverage", "Cov.$\\uparrow$"),
      ("authenticity", "Auth.$\\uparrow$"), ("tstr_hgb_auroc", "AUROC (HGB)$\\uparrow$")]
S3 = [("share_closer_train", "Closer-to-train$\\rightarrow$.5"), ("dcr_ratio", "DCR ratio$\\uparrow$"),
      ("nndr_ratio", "NNDR ratio$\\uparrow$"), ("exact_match_syn", "Exact match$\\downarrow$"),
      ("P_dcr", "$P_{\\text{DCR}}$"), ("P_nndr", "$P_{\\text{NNDR}}$"), ("P_em", "$P_{\\text{EM}}$"),
      ("mia_dist_tpr@fpr0.1", "D-MIA TPR@.1$\\downarrow$"), ("mia_domias_tpr@fpr0.1", "DOMIAS TPR@.1$\\downarrow$")]
S4 = [("repr_parity", "Parity$\\uparrow$"), ("stat_parity_diff_max", "Max gap$\\downarrow$"),
      ("minority_class_ratio", "Minority$\\rightarrow$1"), ("subgroup_util_ratio_min", "Worst-sub.$\\uparrow$"),
      ("subgroup_util_ratio_mean", "Mean-sub.$\\uparrow$"), ("category_coverage", "Cat.cov.$\\uparrow$"),
      ("rare_category_coverage", "Rare cov.$\\uparrow$"), ("entropy_preservation", "Entropy$\\uparrow$"),
      ("tail_preservation", "Tails$\\rightarrow$1"), ("uniqueness", "Unique$\\uparrow$")]


def paired_per_dataset(df):
    mets = [("tstr_catboost_auroc", "AUROC"), ("mia_domias_auc", "DOMIAS"), ("mia_dist_auc", "D-MIA"),
            ("P", "$P$"), ("aia_risk", "AIA"), ("repr_parity", "Parity")]
    key = ["dataset", "split", "seed"]
    parts = []
    for g in GENS:
        rows = []
        for d in ORDER:
            cells = []
            for m, _ in mets:
                a = df[(df.method == f"{g}+psygentab") & (df.dataset == d)].set_index(key)[m]
                b = df[(df.method == g) & (df.dataset == d)].set_index(key)[m]
                j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
                dr = DIR[m]
                dd = ((j.b - 0.5).abs() - (j.a - 0.5).abs()) if dr == 0 else dr * (j.a - j.b)
                p, lo, hi = CI.t_ci(dd)
                cells.append("n/a" if np.isnan(p) else
                             f"{p:+.3f}\\,{{\\tiny[{lo:+.3f}, {hi:+.3f}]}}" if not np.isnan(lo) else f"{p:+.3f}")
            rows.append(f"{DSL[d]} & " + " & ".join(cells) + " \\\\")
        parts.append(r"""\begin{table}[p]
\centering\scriptsize
\caption{Paired effect of PSyGenTAB on \textbf{""" + GL[g] + r"""} per dataset: mean improvement over the raw generator (runs paired on split and seed; positive = better in the metric's ideal direction; attack AUCs as distance from 0.5) with Student-$t$ 95\% CI.}
\label{tab:s_paired:""" + g + r"""}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}l""" + "c" * len(mets) + r"""@{}}
\toprule
Dataset & """ + " & ".join(h for _, h in mets) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write("supp_paired.tex", "\n".join(parts))


def ablation_per_generator():
    a1 = pd.read_csv(os.path.join(ABL, "A1_pmin_sweep.csv"))
    a2 = pd.read_csv(os.path.join(ABL, "A2_outer_iters.csv"))
    a3 = pd.read_csv(os.path.join(ABL, "A3_per_run.csv"))
    sels = [("alm", "ALM"), ("min_intervention", "Min. interv."), ("penalty_lam1", "Penalty $\\lambda$=1"),
            ("penalty_lam10", "Penalty $\\lambda$=10"), ("oracle", "Oracle"), ("unconstrained", "Unconstr.")]
    parts = []
    for g in GENS:
        sub = a1[a1.generator == g]
        if sub.empty:
            continue
        pms = sorted(sub.p_min.unique())
        rows = []
        for s, lab in sels:
            cells = []
            for pm in pms:
                x = sub[(sub.selector == s) & np.isclose(sub.p_min, pm)]
                k, n = int(x.feasible_test.sum()), int(x.feasible_test.count())
                p, lo, hi = CI.wilson(k, n)
                cells.append(f"{100 * p:.0f}\\,{{\\tiny[{100 * lo:.0f},{100 * hi:.0f}]}} / {x.U_test.mean():.3f}")
            rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        k = a2[a2.generator == g].groupby("outer_iters")[["feasible_val", "feasible_test"]].mean()
        w = a3[a3.generator == g][["same_selection", "P_test_sd", "U_test_sd"]].agg(["mean"]).iloc[0]
        parts.append(r"""\begin{landscape}
\begin{table}[p]
\centering\scriptsize
\caption{Selection-rule ablation for \textbf{""" + GL[g] + r"""} (""" + str(sub.groupby(["dataset", "run"]).ngroups) + r""" runs). Each cell: share of runs (\%, 95\% Wilson CI) meeting $P\geq P_{\min}$ on the untouched test split / mean test utility $U$. Below: ALM outer iterations $K$ and weight sensitivity.}
\label{tab:s_abl:""" + g + r"""}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}l""" + "c" * len(pms) + r"""@{}}
\toprule
Selector & """ + " & ".join(f"$P_{{\\min}}$={pm:.2f}" for pm in pms) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
\multicolumn{""" + str(len(pms) + 1) + r"""}{@{}l}{$K$ = """ + ", ".join(str(int(i)) for i in k.index) + r""": feasible on val (\%) """ +
            ", ".join(f"{100 * v:.0f}" for v in k.feasible_val) + "; on test (\\%) " +
            ", ".join(f"{100 * v:.0f}" for v in k.feasible_test) + r"""} \\
\multicolumn{""" + str(len(pms) + 1) + r"""}{@{}l}{Weights (200 Dirichlet(4) draws per run): same release """ +
            f"{100 * w.same_selection:.0f}\\%, SD of test $P$ {w.P_test_sd:.3f}, SD of test $U$ {w.U_test_sd:.3f}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}
\end{landscape}
""")
    write("supp_ablation.tex", "\n".join(parts))


def tapas_full():
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, "tapas", "*__features.npz"))):
        d, g = os.path.basename(p).replace("__features.npz", "").split("__")
        js = p.replace("__features.npz", ".json")
        if not os.path.exists(js):
            continue
        meta = json.load(open(js))
        r = _tapas_boot(p)
        per_raw = ", ".join(f"{x:.2f}" for x in meta.get("raw_auc_per_target") or [])
        per_psy = ", ".join(f"{x:.2f}" for x in meta.get("psygentab_auc_per_target") or [])
        rows.append((ORDER.index(d), GENS.index(g) if g in GENS else 99,
                     f"{DSL[d]} & {GL.get(g, g)} & {meta['m']} & {CI.bracket(*r['raw'], d=2)} & "
                     f"{CI.bracket(*r['psygentab'], d=2)} & {per_raw} & {per_psy} \\\\"))
    rows = [r for *_, r in sorted(rows)]
    write("supp_tapas.tex", r"""\begin{landscape}
\begin{longtable}{@{}llrccll@{}}
\caption{Shadow-model membership inference on every dataset and generator: world size $m$, mean target AUC with 95\% bootstrap CI over the 20 evaluation worlds, and per-target AUCs for the 5 most isolated training records.}\label{tab:s_tapas}\\
\toprule
Dataset & Generator & $m$ & Raw & PSyGenTAB & Raw per target & PSyGenTAB per target \\
\midrule\endfirsthead
\toprule
Dataset & Generator & $m$ & Raw & PSyGenTAB & Raw per target & PSyGenTAB per target \\
\midrule\endhead
""" + "\n".join(rows) + r"""
\bottomrule
\end{longtable}
\end{landscape}
""")


def dp_diagnostics():
    rows = []
    for m, lab in DPM[:6]:
        fs = [json.load(open(p)) for p in glob.glob(os.path.join(RES, "runs", "*", m, "*", "fit.json"))]
        if not fs:
            continue
        info = [f.get("fit_info", {}) for f in fs]
        eps = [i.get("dp_epsilon") for i in info if i.get("dp_epsilon") is not None]
        sig = [i.get("noise_multiplier") for i in info if i.get("noise_multiplier") is not None]
        q = [i.get("sample_rate") for i in info if i.get("sample_rate") is not None]
        steps = [i.get("critic_steps") for i in info if i.get("critic_steps") is not None]
        valid = [f["sample_info"]["valid_rate"] for f in fs if (f.get("sample_info") or {}).get("valid_rate") is not None]
        rows.append(f"{lab} & {len(fs)} & {np.max(eps):.3f} & {np.min(sig):.2f}--{np.max(sig):.2f} & "
                    f"{np.min(q):.3f}--{np.max(q):.3f} & {int(np.median(steps)):,} & {100 * np.mean(valid):.1f}\\% \\\\")
    write("supp_dp.tex", r"""\begin{table}[t]
\centering\footnotesize
\caption{DP-CTAB-GAN+ diagnostics over all runs: achieved $\varepsilon$ (maximum over runs, from the RDP accountant at $\delta=n^{-1.1}$), range of calibrated noise multipliers $\sigma$ and Poisson sampling rates $q$, median number of critic steps, and share of generated rows inside the training ranges (CTAB-GAN+'s validity criterion).}
\label{tab:s_dp}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrccccr@{}}
\toprule
Model & Runs & max $\varepsilon$ & $\sigma$ & $q$ & Steps & Valid rows \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def figures_extra(df):
    """Per-dataset privacy-utility maps and the REaLTabFormer P_min frontier per dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.figures import S1 as C1, S2 as C2, S3 as C3, INK2, INK3, style
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.6), sharey=False)
    for ax, d in zip(axes.ravel(), ORDER):
        style(ax, "both")
        sub = df[df.dataset == d].groupby("method")[["mia_domias_auc", "tstr_catboost_auroc"]].mean()
        for g in GENS:
            if g in sub.index and f"{g}+psygentab" in sub.index:
                a, b = sub.loc[g], sub.loc[f"{g}+psygentab"]
                ax.annotate("", (b.mia_domias_auc, b.tstr_catboost_auroc), (a.mia_domias_auc, a.tstr_catboost_auroc),
                            arrowprops=dict(arrowstyle="-|>", color=INK3, lw=0.6, shrinkA=2, shrinkB=2))
                ax.plot(a.mia_domias_auc, a.tstr_catboost_auroc, "o", color=C3, ms=3.5, mec="white", mew=0.3)
                ax.plot(b.mia_domias_auc, b.tstr_catboost_auroc, "D", color=C1, ms=3.5, mec="white", mew=0.3)
        for m in sub.index:
            if "_eps" in m:
                ax.plot(sub.loc[m, "mia_domias_auc"], sub.loc[m, "tstr_catboost_auroc"], "s", color=C2, ms=3, mec="white", mew=0.3)
        ax.axvline(0.5, color=INK3, ls="--", lw=0.6)
        ax.set_title(DSL[d], loc="left", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    for ax in axes[1]:
        ax.set_xlabel("DOMIAS AUC", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("TSTR AUROC", fontsize=7)
    h = [plt.Line2D([], [], marker=mk, color=c, ls="", ms=5) for mk, c in [("o", C3), ("D", C1), ("s", C2)]]
    fig.legend(h, ["Raw generator", "+ PSyGenTAB (ALM)", "DP baseline"], loc="lower center", ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs(os.path.join(PAPER, "figures"), exist_ok=True)
    fig.savefig(os.path.join(PAPER, "figures", "supp_maps_per_dataset.pdf"))
    plt.close(fig)

    a1 = pd.read_csv(os.path.join(ABL, "A1_pmin_sweep.csv"))
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.6))
    for ax, d in zip(axes.ravel(), ORDER):
        style(ax, "both")
        for g, col, mk in [("rtf", C1, "D"), ("ctabgan_plus", C2, "s"), ("bayesian_network", C3, "o")]:
            s = a1[(a1.generator == g) & (a1.dataset == d) & (a1.selector == "alm")].groupby("p_min")[["P_test", "U_test"]].mean()
            if len(s):
                ax.plot(s.P_test, s.U_test, marker=mk, color=col, ms=3, lw=1, label=GL[g])
        ax.set_title(DSL[d], loc="left", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    for ax in axes[1]:
        ax.set_xlabel("Test privacy $P$", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Test utility $U$", fontsize=7)
    hh, ll = axes[0, 0].get_legend_handles_labels()
    fig.legend(hh, ll, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(PAPER, "figures", "supp_pareto_per_dataset.pdf"))
    plt.close(fig)


def hyper_table():
    write("supp_hyper.tex", r"""\begin{table}[t]
\centering\footnotesize
\caption{Generator hyper-parameters ($n$: training rows, capped at 2{,}000).}
\label{tab:s_hyper}
\begin{tabular}{@{}lp{0.68\linewidth}@{}}
\toprule
Generator & Setting \\
\midrule
REaLTabFormer & tabular GPT-2; batch 64; epochs 100/60/40 for $n\leq500/\leq1000/>1000$; no internal critic; sampling temperatures 0.8, 1.0, 1.2 (top-$p$ = 1); numeric width 10 (11 for PIR Vision, VN Banking) \\
CTAB-GAN+ & official implementation; epochs 300/200/150 for $n\leq500/\leq2000/>2000$; batch $\min(500, n/2)$; mixed columns: numerical columns whose mode holds $\geq30\%$ of rows \\
DP-CTAB-GAN+ & as CTAB-GAN+ with DP-SGD on the critic: Poisson sampling $q=B/n$, clipping $C=1$, $\delta=n^{-1.1}$, RDP accounting; clip variant: weights in $[-0.01,0.01]$; GP variant: per-record slerp penalty, weight 10; auxiliary classifier and information loss disabled \\
BayesNet, ARF & synthcity defaults \\
TVAE, CTGAN & synthcity; epochs as CTAB-GAN+ \\
TabDDPM & synthcity; 20{,}000 gradient steps; batch 256 \\
PATE-GAN & synthcity; logistic-regression teachers; $\delta=1/(n\sqrt n)$ \\
PSyGenTAB & $P_{\min}=0.90$; pool $2n$ per configuration; $\tau$ quantiles $0,0.05,\dots,0.50$; $\lambda_0=0$, $\mu_0=10$, growth 10, $K\leq20$; weights $U$: $\tfrac14$ each, $P$: $\tfrac12,\tfrac14,\tfrac14$ \\
DCR filter & drop records closer to $T$ than the 5th percentile of holdout-to-train DCR \\
\bottomrule
\end{tabular}
\end{table}
""")


def driver():
    write("supplementary.tex", r"""\documentclass[10pt]{article}
\usepackage[margin=1.8cm]{geometry}
\usepackage{times}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,longtable,pdflscape,graphicx,caption}
\usepackage[hidelinks]{hyperref}
\captionsetup{font=small}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\thesection}{S\arabic{section}}
\title{Supplementary Material for\\ \textbf{PSyGenTAB: Privacy-Constrained Synthetic Health Data Generation at Sampling Time}}
\author{Arshia Ilaty, Hossein Shirazi, Manasi Chitale, Kedar Hegde, Dhanalakshmi Ramesh,\\ Rashmi S. Manjunath, Amir M. Rahmani, and Hajar Homayouni}
\date{}
\begin{document}
\maketitle
\noindent All values are generated from the released run records. Unless stated otherwise, entries are means over runs $\pm$ half-width of the Student-$t$ 95\% confidence interval; $n$ is the number of runs (REaLTabFormer, CTAB-GAN+ and DP-CTAB-GAN+: 5 splits $\times$ 3 seeds; all other generators: 5 splits). Metric definitions follow Section~V of the main paper; $\rightarrow$.5 and $\rightarrow$1 denote metrics whose ideal value is 0.5 or 1.

\tableofcontents
\clearpage

\section{Hyper-parameters and DP diagnostics}
\input{supp_hyper}
\input{supp_dp}
\clearpage

\section{Full results per dataset}
Utility, fidelity and privacy of every method on every dataset (Tables~\ref{tab:s_full:breast_cancer}ff.).
\input{supp_full}
\clearpage

\section{Fidelity details per dataset}
\input{supp_fidelity}
\clearpage

\section{Privacy details per dataset}
\input{supp_privacy}
\clearpage

\section{Fairness and diversity per dataset}
Entries marked n/a are undefined by construction, not missing. Breast Cancer, Liver Disorders and Parkinson's have no categorical features, so there are no subgroups for representation parity, the maximum statistical-parity gap or subgroup utility. Rare-category coverage needs at least one category with under 5\% of training rows; Breast Cancer, Heart Failure, Liver Disorders, Parkinson's and PIR Vision have none.
\input{supp_fairness}
\clearpage

\section{Paired effect of PSyGenTAB per dataset and generator}
\input{supp_paired}
\clearpage

\section{Selection-rule ablation per generator}
\input{supp_ablation}
\clearpage

\section{Shadow-model attacks: complete results}
\input{supp_tapas}
\clearpage

\section{Additional figures}
\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{figures/supp_maps_per_dataset.pdf}
\caption{Privacy--utility map per dataset: DOMIAS AUC versus TSTR AUROC for every generator, raw and with PSyGenTAB, and the DP baselines.}
\end{figure}
\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{figures/supp_pareto_per_dataset.pdf}
\caption{Privacy-floor sweep per dataset ($P_{\min}\in[0.70,0.98]$, ALM) for REaLTabFormer, CTAB-GAN+ and BayesNet.}
\end{figure}
\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{figures/fig_leaderboard.pdf}
\caption{Average utility and privacy rank of every method.}
\end{figure}
\end{document}
""")


def main():
    df = load()
    long_table(df, S1, "Utility, fidelity and privacy of every method", "tab:s_full", "supp_full.tex")
    long_table(df, S2, "Fidelity details", "tab:s_fid", "supp_fidelity.tex")
    long_table(df, S3, "Privacy details (proximity components and attack TPR at 10\\% FPR)", "tab:s_priv", "supp_privacy.tex")
    long_table(df, S4, "Fairness and diversity", "tab:s_fair", "supp_fairness.tex")
    paired_per_dataset(df)
    ablation_per_generator()
    tapas_full()
    dp_diagnostics()
    hyper_table()
    figures_extra(df)
    driver()
    print("wrote supplementary into", PAPER)


if __name__ == "__main__":
    main()
