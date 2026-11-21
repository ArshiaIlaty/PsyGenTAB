"""Build the public research website (static, self-contained) into site/.

    python -m scripts.build_site

site/index.html embeds all data (works from file:// and GitHub Pages);
paper figures are copied to site/assets/. Re-run after new results land;
models with no finished runs appear as labelled "pending" slots.
"""
import glob
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

from scripts.leaderboard import GEN_LABEL
from scripts.review_dashboard import ALL_GENS, DP, DS_LABEL, build_payload

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
SITE = os.path.join(ROOT, "site")
TEMPLATE = os.path.join(ROOT, "scripts", "site_template.html")

PLANNED = {g: 15 for g in ["rtf", "ctabgan_plus"] + [f"ctabgan_plus_{v}_eps{e}" for v in ("dp", "dpgp")
                                                        for e in (1, 3, 10)]}
PLANNED.update({g: 5 for g in ["bayesian_network", "arf", "tvae", "ctgan", "tabddpm"]
                + [f"pategan_eps{e}" for e in (1, 3, 10)]})

FIGURES = [
    ("fig_privacy_utility_map", "Privacy–utility map: every method, arrows from raw generator to + PSyGenTAB."),
    ("fig_model_agnostic", "PSyGenTAB on every generator: privacy, two membership attacks and utility."),
    ("fig_pareto_pmin", "Sweeping the privacy target P_min traces each generator's privacy–utility frontier."),
    ("fig_constraint_iterations", "Constraint satisfaction on the untouched test split, and ALM outer iterations."),
    ("fig_dp_epsilon", "Differentially private baselines across ε, with PSyGenTAB reference lines."),
    ("fig_delta_heatmap", "PSyGenTAB minus raw generator for every generator × dataset."),
    ("fig_leaderboard", "Average utility and privacy rank of every method across datasets."),
    ("fig_weight_sensitivity", "Robustness to the choice of utility and privacy weights."),
    ("fig_progress", "Study progress."),
]


def sweep():
    p = os.path.join(RES, "ablations", "A1_pmin_sweep.csv")
    if not os.path.exists(p):
        return {}
    a = pd.read_csv(p)
    a = a[a.selector.isin(["alm", "min_intervention", "unconstrained"])]
    cols = ["P_test", "U_test", "tstr_f1_test", "mia_dist_auc_test", "share_test", "feasible_test", "tau_q"]
    g = a.groupby(["generator", "selector", "p_min"])[cols].mean().reset_index()
    out = {}
    for gen, sub in g.groupby("generator"):
        out[gen] = {sel: [{k: (round(float(r[k]), 4) if k != "p_min" else float(r[k]))
                           for k in ["p_min"] + cols} for _, r in s.iterrows()]
                    for sel, s in sub.groupby("selector")}
        out[gen]["n_runs"] = int(a[(a.generator == gen) & (a.selector == "alm")].groupby(["dataset", "run"]).ngroups)
    return out


def tapas():
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, "tapas", "*.json"))):
        d = json.load(open(p))
        rows.append({k: d.get(k) for k in ["dataset", "generator", "m", "n_shadow", "n_eval",
                                           "raw_auc_mean", "psygentab_auc_mean"]}
                    | {"n_targets": len(d.get("raw_auc_per_target") or [])})
    done = {(r["dataset"], r["generator"]) for r in rows}
    for ds in ["heart_failure", "breast_cancer", "obesity"]:
        for g in ["rtf", "ctabgan_plus", "bayesian_network", "tabddpm", "tvae"]:
            if (ds, g) not in done:
                rows.append({"dataset": ds, "generator": g, "pending": True})
    return rows


def stats():
    out = []
    for m in ["P", "mia_domias_auc", "mia_dist_auc", "tstr_catboost_auroc", "repr_parity", "subgroup_util_ratio_min"]:
        p = os.path.join(RES, "tables", f"wilcoxon_agnostic__{m}.csv")
        if not os.path.exists(p):
            continue
        w = pd.read_csv(p)
        w = w[(w.method == w.reference.str.split("+").str[0])]
        for r in w.itertuples():
            out.append({"metric": m, "reference": r.reference, "n": int(r.n_datasets), "mean_diff": float(r.mean_diff),
                        "wins": int(r.wins), "losses": int(r.losses), "p": float(r.p), "p_holm": float(r.p_holm),
                        "rank_biserial": float(r.rank_biserial)})
    return out


def datasets():
    from psygentab.datasets import describe_all
    d = describe_all()
    d["label"] = d.dataset.map(DS_LABEL)
    return d.round(4).to_dict(orient="records")


def progress():
    done = {}
    for p in glob.glob(os.path.join(RES, "runs", "*", "*", "split*_seed*", "DONE")):
        g = p.split(os.sep)[-3]
        done[g] = done.get(g, 0) + 1
    return [{"generator": g, "done": done.get(g, 0), "planned": n * 10} for g, n in PLANNED.items()]


def paper_numbers():
    """The macros the paper is typeset from (numbers.tex), so site and paper agree."""
    import re
    path = os.path.join(ROOT, "2026_JBHI___PSyGenTAB__Privacy_Preserving_Synthetic_Health_Data", "numbers.tex")
    if not os.path.exists(path):
        return {}
    out = {}
    for k, v in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", open(path).read(), flags=re.M):
        out[k] = v.replace("$", "").replace("-", "−")
    return out


def main():
    os.makedirs(os.path.join(SITE, "assets"), exist_ok=True)
    payload = build_payload()
    payload.update(sweep=sweep(), tapas=tapas(), stats=stats(), datasets_table=datasets(), progress=progress(),
                   gen_labels=GEN_LABEL, nums=paper_numbers(), built=time.strftime("%B %d, %Y"))
    figs = []
    for name, cap in FIGURES:
        src = os.path.join(RES, "figures", f"{name}.png")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(SITE, "assets", f"{name}.png"))
            pdf = os.path.join(RES, "figures", f"{name}.pdf")
            if os.path.exists(pdf):
                shutil.copy2(pdf, os.path.join(SITE, "assets", f"{name}.pdf"))
            figs.append({"name": name, "caption": cap})
    payload["figures"] = figs
    html = open(TEMPLATE).read().replace("__DATA__", json.dumps(payload, default=float))
    with open(os.path.join(SITE, "index.html"), "w") as f:
        f.write(html)
    open(os.path.join(SITE, ".nojekyll"), "w").close()
    print("wrote", os.path.join(SITE, "index.html"), f"({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
