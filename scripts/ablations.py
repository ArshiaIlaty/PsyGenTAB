"""Ablations from the cached candidate tables (no re-generation needed).

For every non-DP run, `candidates.csv` scores each (config, tau) candidate on
val (used for selection) and test (used for reporting). We re-run the
selectors offline:

  A1  P_min sweep        ALM vs. fixed penalty (lambda=1, 10) vs. oracle
                         constrained argmax vs. unconstrained (-> Pareto front)
  A2  outer iterations   ALM with K in {1,2,3,5,10,20}
  A3  weight sensitivity 200 Dirichlet draws of the U and P weights
Outputs: results/ablations/*.csv
"""
import glob
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from psygentab import core
from psygentab.metrics import DEFAULT_PW

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "ablations")
P_MINS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98]
ITERS = [1, 2, 3, 5, 10, 20]
P_MAIN = 0.90


def test_view(tab, i):
    r = tab.iloc[i]
    return {"U_val": core.utility(r), "P_val": core.privacy(r),
            "U_test": core.utility(r, prefix="test_"), "P_test": core.privacy(r, prefix="test_"),
            "tstr_f1_test": r["test_tstr_f1"], "mia_dist_auc_test": r["test_mia_dist_auc"],
            "share_test": r["test_share_closer_train"], "tau_q": r["tau_q"],
            "config_idx": int(r["config_idx"]), "accept_rate": r["accept_rate"]}


def load_tables():
    for p in sorted(glob.glob(os.path.join(RES, "runs", "*", "*", "split*_seed*", "candidates.csv"))):
        parts = p.split(os.sep)
        ds, gen, run = parts[-4], parts[-3], parts[-2]
        if not os.path.exists(os.path.join(os.path.dirname(p), "DONE")):
            continue
        tab = pd.read_csv(p)
        yield ds, gen, run, tab


def main():
    os.makedirs(OUT, exist_ok=True)
    a1, a2, a3 = [], [], []
    rng = np.random.default_rng(0)
    U_KEYS, P_KEYS = list(core.DEFAULT_UW), list(DEFAULT_PW)
    draws = [(dict(zip(U_KEYS, rng.dirichlet(4 * np.ones(len(U_KEYS))))),
              dict(zip(P_KEYS, rng.dirichlet(4 * np.ones(len(P_KEYS)))))) for _ in range(200)]
    for ds, gen, run, tab in load_tables():
        key = dict(dataset=ds, generator=gen, run=run)
        i0 = core.select_unconstrained(tab)
        for pm in P_MINS:
            for name, (i, feas) in {
                "alm": core.select_alm(tab, pm),
                "penalty_lam1": core.select_penalty(tab, pm, lam=1.0),
                "penalty_lam10": core.select_penalty(tab, pm, lam=10.0),
                "oracle": core.select_oracle(tab, pm),
                "min_intervention": core.select_min_intervention(tab, pm),
                "unconstrained": (i0, None),
            }.items():
                v = test_view(tab, i)
                a1.append(dict(key, p_min=pm, selector=name, feasible_val=v["P_val"] >= pm,
                               feasible_test=v["P_test"] >= pm, **v))
        for K in ITERS:
            i, feas = core.select_alm(tab, P_MAIN, outer_iters=K)
            v = test_view(tab, i)
            a2.append(dict(key, outer_iters=K, feasible_val=v["P_val"] >= P_MAIN,
                           feasible_test=v["P_test"] >= P_MAIN, **v))
        i_ref, _ = core.select_alm(tab, P_MAIN)
        for d, (uw, pw) in enumerate(draws):
            i, feas = core.select_alm(tab, P_MAIN, uw=uw, pw=pw)
            v = test_view(tab, i)
            a3.append(dict(key, draw=d, same_as_default=(i == i_ref),
                           P_test_default_w=core.privacy(tab.iloc[i], prefix="test_"), **v))
    pd.DataFrame(a1).to_csv(os.path.join(OUT, "A1_pmin_sweep.csv"), index=False)
    pd.DataFrame(a2).to_csv(os.path.join(OUT, "A2_outer_iters.csv"), index=False)
    pd.DataFrame(a3).to_csv(os.path.join(OUT, "A3_weight_sensitivity.csv"), index=False)

    A1 = pd.DataFrame(a1)
    if len(A1):
        s = (A1.groupby(["generator", "selector", "p_min"])
             [["feasible_val", "feasible_test", "U_test", "P_test", "tstr_f1_test", "mia_dist_auc_test"]]
             .mean().round(3))
        s.to_csv(os.path.join(OUT, "A1_summary.csv"))
        A2 = pd.DataFrame(a2).groupby(["generator", "outer_iters"])[
            ["feasible_val", "feasible_test", "U_test", "P_test"]].mean().round(3)
        A2.to_csv(os.path.join(OUT, "A2_summary.csv"))
        # spread across weight draws *within* each run, then averaged over runs
        per_run = pd.DataFrame(a3).groupby(["generator", "dataset", "run"]).agg(
            same_selection=("same_as_default", "mean"), U_test_sd=("U_test", "std"),
            P_test_sd=("P_test_default_w", "std"), tstr_f1_sd=("tstr_f1_test", "std"))
        per_run.to_csv(os.path.join(OUT, "A3_per_run.csv"))
        A3 = per_run.groupby("generator").mean()
        A3.to_csv(os.path.join(OUT, "A3_summary.csv"))
        print(s.xs(P_MAIN, level="p_min").to_string())
        print(A2.to_string())
        print(A3.to_string())


if __name__ == "__main__":
    main()
