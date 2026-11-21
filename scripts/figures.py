"""Paper figures for the full PSyGenTAB study.

Every generator in the study gets its slot in every figure. Generators with no
finished runs yet (e.g. the GPU lane before the AWS runs) are drawn as labelled
"pending" slots, so layouts are final and fill in automatically on re-run.

Inputs : results/tables/all_runs_long.csv, results/tables/trtr_reference.csv,
         results/ablations/A1_pmin_sweep.csv, A2_outer_iters.csv, A3_per_run.csv
Outputs: results/figures/*.pdf and *.png
"""
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "results", "tables")
ABL = os.path.join(ROOT, "results", "ablations")
OUT = os.path.join(ROOT, "results", "figures")

# ---------------------------------------------------------------- study design
GENS = ["rtf", "ctabgan_plus", "bayesian_network", "arf", "tvae", "ctgan", "tabddpm"]
GEN_LABEL = {"rtf": "REaLTabFormer", "ctabgan_plus": "CTAB-GAN+", "bayesian_network": "BayesNet",
             "arf": "ARF", "tvae": "TVAE", "ctgan": "CTGAN", "tabddpm": "TabDDPM"}
DP_FAMILIES = {"pategan": "PATE-GAN", "ctabgan_plus_dp": "DP-CTAB-GAN+ (weight clip)",
               "ctabgan_plus_dpgp": "DP-CTAB-GAN+ (grad. penalty)"}
EPSILONS = [1, 3, 10]
DATASETS = ["breast_cancer", "diabetes", "heart_failure", "hypothyroid", "liver_disorders",
            "obesity", "parkinsons", "adult", "pir_vision", "vn_banking"]
DS_LABEL = {"breast_cancer": "Breast C.", "diabetes": "Diabetes", "heart_failure": "Heart F.",
            "hypothyroid": "Hypothyroid", "liver_disorders": "Liver", "obesity": "Obesity",
            "parkinsons": "Parkinson's", "adult": "Adult", "pir_vision": "PIR Vision",
            "vn_banking": "VN Bank"}

# ---------------------------------------------------------------- style
# Validated categorical slots (reference palette, light mode), fixed order.
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8c8b86"
GRID = "#e6e5e1"
# (variant, label, colour, marker, filled). The minimal-intervention variant
# shares PSyGenTAB's hue with a hollow marker instead of a 4th categorical hue.
VARIANTS = [("psygentab", "+ PSyGenTAB (ALM)", S1, "D", True),
            ("psygentab_min", "+ PSyGenTAB (min. intervention)", S1, "D", False),
            ("filter", "+ DCR filter", S2, "s", True), ("base", "Raw generator", S3, "o", True)]
OFFSETS = [0.30, 0.10, -0.10, -0.30]
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#1c5cab", "#f0efec", "#e34948"])

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK3, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.5,
    "lines.markersize": 4.5, "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.dpi": 300, "pdf.fonttype": 42, "figure.facecolor": "white",
})
PENDING = "pending (GPU run)"
GPU_GENS = {"rtf", "ctabgan_plus"}


def pending(g=None):
    """Placeholder label for a model with no finished runs yet, by compute lane."""
    if g is None or g in GPU_GENS or str(g).startswith("ctabgan_plus"):
        return "pending (AWS GPU run)"
    return "pending (CPU run)"


def style(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT, f"{name}.png"))
    plt.close(fig)
    print("wrote", name)


def mean_ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return x[0], 0.0
    return x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def load_runs():
    p = os.path.join(TAB, "all_runs_long.csv")
    df = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=["method", "dataset"])
    df["gen"] = df["method"].str.split("+").str[0]
    df["variant"] = df["method"].str.split("+").str[1].fillna("base")
    return df


def dataset_means(df, metric):
    """Per-dataset mean over runs -> one value per (gen, variant, dataset)."""
    return df.groupby(["gen", "variant", "dataset"])[metric].mean().reset_index()


# ---------------------------------------------------------------- Fig. 1
def fig_model_agnostic(df, trtr):
    metrics = [("P", "Privacy $P$ (↑)", None),
               ("mia_domias_auc", "DOMIAS AUC (→0.5)", 0.5),
               ("mia_dist_auc", "Dist. MIA AUC (→0.5)", 0.5),
               ("tstr_catboost_auroc", "TSTR AUROC (↑)", trtr)]
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.9), sharey=True)
    ys = np.arange(len(GENS))[::-1]
    for ax, (m, title, ref) in zip(axes, metrics):
        style(ax)
        dm = dataset_means(df, m) if m in df else pd.DataFrame(columns=["gen", "variant", m])
        lo, hi = [], []
        for y, g in zip(ys, GENS):
            sub = dm[dm.gen == g]
            if sub.empty:
                ax.text(0.5, y, pending(g), transform=ax.get_yaxis_transform(), ha="center",
                        va="center", color=INK3, fontsize=6.5, style="italic")
                continue
            for k, (v, _, col, mk, filled) in enumerate(VARIANTS):
                vals = sub[sub.variant == v][m]
                if vals.empty:
                    continue
                mu, ci = mean_ci(vals)
                ax.errorbar(mu, y + OFFSETS[k], xerr=ci, fmt=mk, color=col, ms=4.2,
                            elinewidth=0.9, capsize=0, mfc=col if filled else "white",
                            mec="white" if filled else col, mew=0.5 if filled else 1.0)
                lo.append(mu - ci)
                hi.append(mu + ci)
        if ref is not None and not np.isnan(ref):
            ax.axvline(ref, color=INK3, lw=0.8, ls="--", zorder=0)
            ax.text(ref, len(GENS) - 0.45, "TRTR" if ref != 0.5 else "random", color=INK2,
                    fontsize=6, ha="center", va="bottom")
        if lo:
            pad = 0.08 * (max(hi) - min(lo) + 1e-9)
            left, right = min(lo) - pad, max(hi) + pad
            if ref is not None and not np.isnan(ref):
                left, right = min(left, ref - pad), max(right, ref + pad)
            ax.set_xlim(left, right)
        ax.set_title(title, loc="left")
        ax.set_ylim(-0.6, len(GENS) - 0.2)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([GEN_LABEL[g] for g in GENS])
    handles = [plt.Line2D([], [], marker=mk, color=col, ls="", ms=5, mfc=col if f else "white",
                          mec="white" if f else col, mew=0.5 if f else 1.0)
               for _, _, col, mk, f in VARIANTS]
    fig.legend(handles, [v[1] for v in VARIANTS], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(wspace=0.12, bottom=0.18)
    save(fig, "fig_model_agnostic")


# ---------------------------------------------------------------- Fig. 2
def fig_pareto(a1):
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.6))
    axes = axes.ravel()
    sel_style = [("alm", "PSyGenTAB (ALM)", S1, "D", "-"),
                 ("min_intervention", "PSyGenTAB (min. intervention)", S1, "d", ":"),
                 ("penalty_lam1", "Fixed penalty λ=1", S2, "s", "--")]
    for ax, g in zip(axes, GENS):
        style(ax, "both")
        ax.set_title(GEN_LABEL[g], loc="left")
        sub = a1[a1.generator == g] if len(a1) else a1
        if sub is None or len(sub) == 0:
            ax.text(0.5, 0.5, pending(g), transform=ax.transAxes, ha="center", va="center",
                    color=INK3, fontsize=7, style="italic")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        for sel, lab, col, mk, ls in sel_style:
            s = sub[sub.selector == sel].groupby("p_min")[["P_test", "U_test"]].mean()
            hollow = sel == "min_intervention"
            ax.plot(s.P_test, s.U_test, ls=ls, marker=mk, color=col, ms=3.5, lw=1.2,
                    mfc="white" if hollow else col, mec=col if hollow else "white", mew=0.8 if hollow else 0.4)
            if sel == "alm":
                for pm in (s.index.min(), s.index.max()):
                    ax.annotate(f"{pm:.2f}", (s.P_test[pm], s.U_test[pm]), textcoords="offset points",
                                xytext=(3, 3), fontsize=5.5, color=INK2)
        u = sub[sub.selector == "unconstrained"][["P_test", "U_test"]].mean()
        ax.plot(u.P_test, u.U_test, "o", color=S3, ms=5, mec="white", mew=0.5)
    for ax in axes[4:7]:
        ax.set_xlabel("Test privacy $P$")
    for ax in (axes[0], axes[4]):
        ax.set_ylabel("Test utility $U$")
    leg = axes[7]
    leg.axis("off")
    handles = [plt.Line2D([], [], color=c, marker=m, ls=l, ms=4, mfc="white" if sel == "min_intervention" else c)
               for sel, _, c, m, l in sel_style]
    handles.append(plt.Line2D([], [], color=S3, marker="o", ls="", ms=5))
    leg.legend(handles, [s[1] for s in sel_style] + ["Unconstrained (max $U$)"], loc="center left")
    leg.text(0.02, 0.12, "Numbers: $P_{\\min}$ range swept", transform=leg.transAxes, fontsize=6, color=INK2)
    fig.tight_layout(w_pad=0.6, h_pad=0.8)
    save(fig, "fig_pareto_pmin")


# ---------------------------------------------------------------- Fig. 3
def fig_constraint(a1, a2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.3))
    for ax in (ax1, ax2):
        style(ax, "y")
    if len(a1):
        series = [("alm", "PSyGenTAB (ALM)", S1, "D", "-"), ("min_intervention", "PSyGenTAB (min. interv.)", S1, "d", ":"),
                  ("penalty_lam1", "Penalty λ=1", S2, "s", "--"),
                  ("penalty_lam10", "Penalty λ=10", S3, "^", ":"), ("oracle", "Oracle constrained", S4, "", "-")]
        for sel, lab, col, mk, ls in series:
            s = a1[a1.selector == sel].groupby("p_min").feasible_test.mean()
            ax1.plot(s.index, s.values, ls=ls, marker=mk or None, color=col, label=lab, ms=3.5,
                     mfc="white" if sel == "min_intervention" else col,
                     lw=2.4 if sel == "oracle" else 1.3, alpha=0.55 if sel == "oracle" else 1,
                     zorder=1 if sel == "oracle" else 2)
        ax1.legend(loc="lower left")
    ax1.set_xlabel("Privacy target $P_{\\min}$")
    ax1.set_ylabel("Share of runs with test $P \\geq P_{\\min}$")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("(a) Constraint met on the test split", loc="left")
    if len(a2):
        for col_name, lab, col, mk in [("feasible_val", "validation (used to select)", S1, "D"),
                                       ("feasible_test", "test (reported)", S2, "s")]:
            s = a2.groupby("outer_iters")[col_name].mean()
            ax2.plot(s.index, s.values, marker=mk, color=col, label=lab, ms=3.5, lw=1.3)
        ax2.set_xscale("log")
        ax2.minorticks_off()
        ax2.set_xticks([1, 2, 3, 5, 10, 20])
        ax2.set_xticklabels(["1", "2", "3", "5", "10", "20"])
        ax2.legend(loc="lower right")
    ax2.set_xlabel("ALM outer iterations $K$")
    ax2.set_ylabel("Share feasible ($P_{\\min}=0.9$)")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("(b) Sensitivity to outer iterations", loc="left")
    fig.tight_layout(w_pad=2)
    save(fig, "fig_constraint_iterations")


# ---------------------------------------------------------------- Fig. 4
def fig_dp(df, trtr):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8))
    for ax in (ax1, ax2):
        style(ax, "y")
    cols = [S1, S2, S3]
    for ax, m, lab, ref, ref_lab in [(ax1, "tstr_catboost_auroc", "TSTR AUROC, CatBoost (↑)", trtr, "TRTR (real data)"),
                                     (ax2, "mia_domias_auc", "DOMIAS AUC (→0.5)", 0.5, "random guessing")]:
        for (fam, name), col, mk in zip(DP_FAMILIES.items(), cols, ["o", "s", "^"]):
            pts = []
            for e in EPSILONS:
                vals = dataset_means(df[df.method == (f"{fam}_eps{e}")], m)[m] if m in df else []
                if len(vals):
                    pts.append((e, *mean_ci(vals)))
            if pts:
                e, mu, ci = map(np.array, zip(*pts))
                ax.errorbar(e, mu, yerr=ci, marker=mk, color=col, label=name, ms=4, lw=1.3,
                            elinewidth=0.8, capsize=0)
            else:
                ax.plot([], [], marker=mk, color=col, label=f"{name} — {pending(fam)}")
        for g, col, ls in [("rtf", INK2, "-."), ("ctabgan_plus", INK3, ":")]:
            vals = dataset_means(df[df.method == f"{g}+psygentab"], m)[m] if m in df else []
            if len(vals):
                ax.axhline(np.mean(vals), color=col, ls=ls, lw=1, label=f"{GEN_LABEL[g]} + PSyGenTAB")
            else:
                ax.plot([], [], color=col, ls=ls, label=f"{GEN_LABEL[g]} + PSyGenTAB — {pending(g)}")
        if ref is not None and not np.isnan(ref):
            ax.axhline(ref, color=INK3, ls="--", lw=0.8, zorder=0)
            ax.text(12.5, ref, ref_lab, fontsize=6, color=INK2, va="bottom", ha="right")
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(EPSILONS)
        ax.set_xticklabels([str(e) for e in EPSILONS])
        ax.set_xlim(0.8, 13)
        ax.set_xlabel("Privacy budget ε  (δ < 1/n)")
        ax.set_title(lab, loc="left")
    h, l = ax1.get_legend_handles_labels()
    fig.tight_layout(w_pad=2, rect=(0, 0.24, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=6.5, bbox_to_anchor=(0.5, 0))
    save(fig, "fig_dp_epsilon")


# ---------------------------------------------------------------- Fig. 5
def fig_delta_heatmap(df):
    panels = [("P", "Δ privacy score $P$"), ("mia_domias_auc", "DOMIAS leakage reduction"),
              ("tstr_catboost_auroc", "Δ TSTR AUROC")]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), sharey=True)
    for ax, (m, title) in zip(axes, panels):
        mat = np.full((len(GENS), len(DATASETS)), np.nan)
        if m in df:
            dm = dataset_means(df, m)
            for i, g in enumerate(GENS):
                for j, d in enumerate(DATASETS):
                    a = dm[(dm.gen == g) & (dm.variant == "psygentab") & (dm.dataset == d)][m]
                    b = dm[(dm.gen == g) & (dm.variant == "base") & (dm.dataset == d)][m]
                    if len(a) and len(b):
                        x, y = a.iloc[0], b.iloc[0]
                        mat[i, j] = (abs(x - 0.5) - abs(y - 0.5)) if "auc" in m and "mia" in m else (x - y)
        # privacy leakage should read "blue = better" in every panel
        shown = -mat if "mia" in m else mat
        lim = np.nanmax(np.abs(shown)) if np.isfinite(shown).any() else 0.05
        im = ax.imshow(shown, cmap=DIVERGING.reversed(), vmin=-lim, vmax=lim, aspect="auto")
        for i, g in enumerate(GENS):
            if np.isnan(mat[i]).all():
                ax.add_patch(plt.Rectangle((-0.5, i - 0.5), len(DATASETS), 1, facecolor="white",
                                           edgecolor=GRID, hatch="////", lw=0))
                ax.text(len(DATASETS) / 2 - 0.5, i, pending(g), ha="center", va="center", fontsize=6,
                        color=INK2, style="italic")
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels([DS_LABEL[d] for d in DATASETS], rotation=60, ha="right")
        ax.set_yticks(range(len(GENS)))
        ax.set_yticklabels([GEN_LABEL[g] for g in GENS])
        ax.set_title(title, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_visible(False)
    axes[0].figure.text(0.01, -0.02, "PSyGenTAB minus raw generator, per dataset (mean over runs). "
                        "Blue = PSyGenTAB better (more private / more useful).", fontsize=6, color=INK2)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_delta_heatmap")


# ---------------------------------------------------------------- Fig. 6
def fig_weights(a3):
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2), sharey=True)
    ys = np.arange(len(GENS))[::-1]
    for ax, (m, title) in zip(axes, [("same_selection", "Same release as default weights"),
                                     ("P_test_sd", "SD of test $P$ across weight draws"),
                                     ("U_test_sd", "SD of test $U$ across weight draws")]):
        style(ax)
        for y, g in zip(ys, GENS):
            sub = a3[a3.generator == g][m] if len(a3) else []
            if len(sub) == 0:
                ax.text(0.5, y, pending(g), transform=ax.get_yaxis_transform(), ha="center", va="center",
                        color=INK3, fontsize=6.5, style="italic")
                continue
            mu, ci = mean_ci(sub)
            ax.errorbar(mu, y, xerr=ci, fmt="D", color=S1, ms=4, elinewidth=0.9, mec="white", mew=0.5)
        ax.set_title(title, loc="left")
        ax.set_ylim(-0.6, len(GENS) - 0.4)
    axes[0].set_xlim(0, 1)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([GEN_LABEL[g] for g in GENS])
    fig.text(0.01, -0.04, "200 Dirichlet(4) draws of the U and P weights per run, $P_{\\min}=0.9$; "
             "mean ± 95% CI over runs.", fontsize=6, color=INK2)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_weight_sensitivity")


# ---------------------------------------------------------------- Fig. 7
def fig_progress(df):
    """Run-completion matrix: which (generator, dataset) cells have results."""
    rows = GENS + [f"{f}_eps{e}" for f in DP_FAMILIES for e in EPSILONS]
    labels = [GEN_LABEL.get(r, r) for r in GENS] + [f"{DP_FAMILIES[f]} ε={e}" for f in DP_FAMILIES for e in EPSILONS]
    target = {r: (15 if (r in ("rtf", "ctabgan_plus") or r.startswith("ctabgan_plus_dp")) else 5) for r in rows}
    mat = np.zeros((len(rows), len(DATASETS)))
    base = df[df.variant == "base"]
    for i, r in enumerate(rows):
        for j, d in enumerate(DATASETS):
            mat[i, j] = len(base[(base.gen == r) & (base.dataset == d)]) / target[r]
    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    cmap = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#86b6ef", "#1c5cab"])
    im = ax.imshow(np.clip(mat, 0, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS], rotation=45, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("Share of planned runs finished", fontsize=7)
    cb.outline.set_visible(False)
    ax.set_title("Study progress (planned: 5 splits × 3 seeds for GPU models, 5 × 1 for the zoo)", loc="left")
    fig.tight_layout()
    save(fig, "fig_progress")


# ---------------------------------------------------------------- Fig. 8
def fig_privacy_utility_map(df, trtr):
    """Every method in (attack AUC, utility) space; arrows: raw -> + PSyGenTAB."""
    from scripts.leaderboard import compute, label
    if df.empty:
        return
    lb, _ = compute(df)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2), sharey=True)
    for ax, (xm, xl) in zip(axes, [("mia_domias_auc", "DOMIAS AUC (0.5 = no leakage)"),
                                   ("mia_dist_auc", "Distance-MIA AUC (0.5 = no leakage)")]):
        style(ax, "both")
        for g in GENS:
            if g not in lb.index:
                continue
            x0, y0 = lb.loc[g, xm], lb.loc[g, "tstr_catboost_auroc"]
            ax.plot(x0, y0, "o", color=S3, ms=5, mec="white", mew=0.5, zorder=3)
            if f"{g}+psygentab" in lb.index:
                x1, y1 = lb.loc[f"{g}+psygentab", xm], lb.loc[f"{g}+psygentab", "tstr_catboost_auroc"]
                ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK3, lw=0.8,
                                                                     shrinkA=3, shrinkB=3), zorder=2)
                ax.plot(x1, y1, "D", color=S1, ms=5, mec="white", mew=0.5, zorder=4)
                # dense cluster near AUC 0.5: move labels into empty space with leader lines
                lead = {"rtf": (6, 20), "tvae": (-4, -24), "ctgan": (34, -14), "ctabgan_plus": (55, -22)}
                if g in lead:
                    ax.annotate(GEN_LABEL[g], (x1, y1), textcoords="offset points", xytext=lead[g], fontsize=6,
                                color=INK2, ha="left", arrowprops=dict(arrowstyle="-", color=INK3, lw=0.5,
                                                                       shrinkA=0, shrinkB=3))
                else:
                    ax.annotate(GEN_LABEL[g], (x1, y1), textcoords="offset points", xytext=(4, 5),
                                fontsize=6, color=INK2)
        dpc = []
        for m in lb.index:
            if lb.loc[m, "family"] == "dp":
                x, y = lb.loc[m, xm], lb.loc[m, "tstr_catboost_auroc"]
                ax.plot(x, y, "s", color=S2, ms=4.5, mec="white", mew=0.5, zorder=3)
                if m.startswith("pategan"):
                    left = m.endswith("eps10")
                    ax.annotate(label(m).replace("PATE-GAN ", "PATE "), (x, y), textcoords="offset points",
                                xytext=(-6, -3) if left else (5, -3), ha="right" if left else "left",
                                fontsize=5.5, color=INK2)
                else:
                    dpc.append((x, y))
        if dpc:
            # one label for the DP-CTAB-GAN+ cluster (both variants, three budgets each)
            cx, cy = max(p[0] for p in dpc), min(p[1] for p in dpc)
            ax.annotate(f"DP-CTAB-GAN+ ({len(dpc)} variants)", (cx, cy), textcoords="offset points",
                        xytext=(8, -2), fontsize=5.5, color=INK2)
        ax.axvline(0.5, color=INK3, ls="--", lw=0.8, zorder=0)
        if not np.isnan(trtr):
            ax.axhline(trtr, color=INK3, ls=":", lw=0.8, zorder=0)
        ax.set_xlabel(xl)
    axes[0].set_ylabel("TSTR AUROC, CatBoost (↑)")
    axes[0].text(0.5005, trtr, " TRTR", fontsize=6, color=INK2, va="bottom")
    missing = [GEN_LABEL[g] for g in GENS if g not in lb.index]
    if not any(m.startswith("ctabgan_plus_dp") for m in lb.index):
        missing.append("DP-CTAB-GAN+ (clip, GP)")
    handles = [plt.Line2D([], [], marker="o", color=S3, ls="", ms=5), plt.Line2D([], [], marker="D", color=S1, ls="", ms=5),
               plt.Line2D([], [], marker="s", color=S2, ls="", ms=5)]
    fig.legend(handles, ["Raw generator", "+ PSyGenTAB (ALM)", "DP baseline"], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02))
    if missing:
        fig.text(0.5, -0.07, "Pending (AWS GPU run): " + ", ".join(missing), ha="center", fontsize=6,
                 color=INK2, style="italic")
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.5)
    save(fig, "fig_privacy_utility_map")


# ---------------------------------------------------------------- Fig. 9
def fig_leaderboard(df):
    """Average rank across datasets (1 = best) on utility and privacy metrics."""
    from scripts.leaderboard import compute
    if df.empty:
        return
    lb, _ = compute(df)
    pend = [f"{GEN_LABEL[g]} (+ PSyGenTAB)" for g in GENS if g not in lb.index] + \
        ["DP-CTAB-GAN+ clip / GP"] * (not any(m.startswith("ctabgan_plus_dp") for m in lb.index))
    n = len(lb) + len(pend)
    fig, ax = plt.subplots(figsize=(3.5, 0.17 * n + 0.9))
    style(ax)
    ys = np.arange(n)[::-1]
    for y, (m, r) in zip(ys, lb.iterrows()):
        ax.plot([r.rank_utility, r.rank_privacy], [y, y], color=GRID, lw=2, zorder=1, solid_capstyle="round")
        ax.plot(r.rank_utility, y, "o", color=S1, ms=4.5, mec="white", mew=0.5, zorder=3)
        ax.plot(r.rank_privacy, y, "s", color=S2, ms=4.5, mec="white", mew=0.5, zorder=3)
    for y, p in zip(ys[len(lb):], pend):
        ax.text(0.5, y, "pending (AWS GPU run)", transform=ax.get_yaxis_transform(), ha="center",
                va="center", fontsize=6, color=INK3, style="italic")
    labels = list(lb.label) + pend
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=6)
    for t, m in zip(ax.get_yticklabels(), list(lb.index) + [""] * len(pend)):
        if "psygentab" in m:
            t.set_fontweight("bold")
    ax.set_xlabel("Average rank over datasets (1 = best)")
    ax.invert_xaxis()
    ax.legend([plt.Line2D([], [], marker="o", color=S1, ls="", ms=5), plt.Line2D([], [], marker="s", color=S2, ls="", ms=5)],
              ["Utility rank", "Privacy rank"], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    ax.set_ylim(-0.7, n - 0.3)
    fig.tight_layout()
    save(fig, "fig_leaderboard")


def main():
    df = load_runs()
    tr = os.path.join(TAB, "trtr_reference.csv")
    trtr = pd.read_csv(tr).trtr_catboost_auroc.mean() if os.path.exists(tr) else np.nan
    rd = lambda n: pd.read_csv(os.path.join(ABL, n)) if os.path.exists(os.path.join(ABL, n)) else pd.DataFrame()
    a1, a2, a3 = rd("A1_pmin_sweep.csv"), rd("A2_outer_iters.csv"), rd("A3_per_run.csv")
    fig_model_agnostic(df, trtr)
    fig_pareto(a1)
    fig_constraint(a1, a2)
    fig_dp(df, trtr)
    fig_delta_heatmap(df)
    fig_weights(a3)
    fig_progress(df)
    fig_privacy_utility_map(df, trtr)
    fig_leaderboard(df)


if __name__ == "__main__":
    main()
