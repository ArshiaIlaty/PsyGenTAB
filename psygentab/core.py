"""PSyGenTAB: sampling-time, privacy-constrained selection of synthetic data.

Given a trained generator G, the custodian's train split T (what G saw) and a
calibration holdout V (never seen by G), PSyGenTAB solves

    max_theta  U(theta)   s.t.   P(theta) >= P_min

over sampling decisions theta = (c, tau): a sampling configuration c of G
(e.g. temperature for REaLTabFormer) and a record-acceptance threshold tau on
the per-record proximity ratio

    rho(s) = d(s, T_ref) / d(s, V),   |T_ref| = |V|,

which is ~1 for records that are no closer to the training data than to
unseen data. The released dataset is n random accepted records. U and P are
computed against V; every reported number is recomputed against the
untouched test split. The constrained problem is solved with the
Powell-Hestenes-Rockafellar augmented Lagrangian over the candidate set.
"""
import numpy as np
import pandas as pd

from . import metrics as M
from .encoding import exact_keys

DEFAULT_UW = {"marginal": 0.25, "assoc": 0.25, "detect": 0.25, "eff": 0.25}
TAU_QUANTILES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


class RealContext:
    """Real-data quantities reused across all candidates of one run."""

    def __init__(self, train, holdout, meta, enc, enc_x, seed=0):
        self.train, self.hold, self.meta = train, holdout, meta
        self.enc, self.enc_x = enc, enc_x
        self.X_train = enc.transform(train)
        self.X_hold = enc.transform(holdout)
        self.k_train = exact_keys(train, meta)
        self.k_hold = exact_keys(holdout, meta)
        rng = np.random.default_rng(seed)
        self.tref_idx = rng.choice(len(train), min(len(holdout), len(train)), replace=False)
        self.A_hold = M.association_matrix(holdout, meta)
        trtr = M.tstr(train, holdout, meta, enc_x, model="hgb", seed=seed)
        self.trtr_f1 = max(trtr["f1_macro"], 1e-6)
        self.seed = seed

    def utility_components(self, syn, X_syn):
        marg = M.marginal_scores(self.hold, syn, self.meta)["marginal_err"]
        assoc = M.association_error(self.hold, syn, self.meta, A_real=self.A_hold)
        auc = M.c2st_auc(self.X_hold, X_syn, seed=self.seed, folds=3, fast=True)
        eff = M.tstr(syn, self.hold, self.meta, self.enc_x, model="hgb", seed=self.seed)["f1_macro"]
        return {"u_marginal": 1 - marg, "u_assoc": 1 - assoc,
                "u_detect": 1 - 2 * abs(auc - 0.5), "u_eff": float(np.clip(eff / self.trtr_f1, 0, 1)),
                "c2st_lr_auc": auc, "tstr_f1": eff}

    def privacy_components(self, syn, X_syn):
        rep = M.privacy_report(X_syn, self.X_train, self.X_hold, exact_keys(syn, self.meta),
                               self.k_train, self.k_hold, seed=self.seed)
        sc = M.privacy_score(rep)
        rep.update({k: v for k, v in sc.items() if k != "P"})
        return rep

    def light_eval(self, syn, X_syn):
        out = self.utility_components(syn, X_syn)
        out.update(self.privacy_components(syn, X_syn))
        Xe, ye = M.mia_members(self.X_train, self.X_hold, seed=self.seed)
        out.update(M.distance_mia(X_syn, Xe, ye))
        return out


def utility(row, w=None, prefix=""):
    w = DEFAULT_UW if w is None else w
    tot = sum(w.values())
    return sum(w[k] * row[f"{prefix}u_{k}"] for k in w) / tot


def privacy(row, w=None, prefix=""):
    w = M.DEFAULT_PW if w is None else w
    tot = sum(w.values())
    return sum(w[k] * row[f"{prefix}P_{k}"] for k in w) / tot


def candidate_table(pools, n_out, val_ctx, test_ctx, seed=0, tau_quantiles=TAU_QUANTILES):
    """Score every (config, tau) candidate on val (for selection) and test (for reporting).

    pools: list of (config_dict, pool_df). Returns (table, subsets) where
    subsets[i] holds the released rows of candidate i.
    """
    rows, subsets = [], []
    for ci, (cfg, pool) in enumerate(pools):
        Xp = val_ctx.enc.transform(pool)
        d_tref = M._nn(val_ctx.X_train[val_ctx.tref_idx], Xp, 1)[:, 0]
        d_val = M._nn(val_ctx.X_hold, Xp, 1)[:, 0]
        rho = d_tref / (d_val + 1e-12)
        seen = set()
        for q in tau_quantiles:
            tau = float(np.quantile(rho, q)) if q > 0 else 0.0
            acc = np.nonzero(rho >= tau)[0] if q > 0 else np.arange(len(pool))
            if len(acc) < n_out:
                continue
            key = (ci, len(acc))
            if key in seen:
                continue
            seen.add(key)
            rng = np.random.default_rng(seed + 7919 * ci + int(1000 * q))
            pick = rng.choice(acc, n_out, replace=False)
            syn = pool.iloc[pick].reset_index(drop=True)
            Xs = Xp[pick]
            rec = {"cand": len(rows), "config_idx": ci, "config": cfg, "tau_q": q, "tau": tau,
                   "accept_rate": len(acc) / len(pool)}
            rec.update(val_ctx.utility_components(syn, Xs))
            rec.update(val_ctx.privacy_components(syn, Xs))
            Xs_t = test_ctx.enc.transform(syn)
            rec.update({f"test_{k}": v for k, v in test_ctx.light_eval(syn, Xs_t).items()})
            rows.append(rec)
            subsets.append(syn)
    tab = pd.DataFrame(rows)
    return tab, subsets


def filter_baseline(pool, n_out, val_ctx, q=0.05, seed=0):
    """Common post-hoc privacy filter: drop records closer to train than the
    q-quantile of holdout-to-train DCR, then release n random survivors."""
    Xp = val_ctx.enc.transform(pool)
    d_train = M._nn(val_ctx.X_train, Xp, 1)[:, 0]
    d_hold = M._nn(val_ctx.X_train, val_ctx.X_hold, 1)[:, 0]
    thr = float(np.quantile(d_hold, q))
    acc = np.nonzero(d_train > thr)[0]
    rng = np.random.default_rng(seed)
    pick = rng.choice(acc, n_out, replace=len(acc) < n_out)
    return pool.iloc[pick].reset_index(drop=True), {"filter_threshold": thr,
                                                    "accept_rate": len(acc) / len(pool)}


# ---------------------------------------------------------------- selectors

def _scores(tab, uw=None, pw=None):
    U = tab.apply(lambda r: utility(r, uw), axis=1).values
    P = tab.apply(lambda r: privacy(r, pw), axis=1).values
    return U, P


def select_default(tab, default_config_idx=0):
    sub = tab[(tab.config_idx == default_config_idx) & (tab.tau_q == 0.0)]
    return int(sub.index[0])


def select_unconstrained(tab, uw=None, pw=None):
    U, _ = _scores(tab, uw, pw)
    sub = np.nonzero(tab.tau_q.values == 0.0)[0]
    return int(sub[np.argmax(U[sub])])


def select_oracle(tab, p_min, uw=None, pw=None):
    U, P = _scores(tab, uw, pw)
    feas = np.nonzero(P >= p_min)[0]
    if len(feas) == 0:
        return int(np.argmax(P)), False
    return int(feas[np.argmax(U[feas])]), True


def select_penalty(tab, p_min, lam=1.0, uw=None, pw=None):
    U, P = _scores(tab, uw, pw)
    L = (1 - U) + lam * np.maximum(0.0, p_min - P)
    i = int(np.argmin(L))
    return i, bool(P[i] >= p_min)


def select_alm(tab, p_min, outer_iters=20, lam0=0.0, mu0=10.0, growth=10.0, tol=1e-4,
               uw=None, pw=None, return_trace=False):
    """PHR augmented Lagrangian for  min (1-U)  s.t.  g = P_min - P <= 0.

    L(theta; lam, mu) = (1-U) + (max(0, lam + mu g)^2 - lam^2) / (2 mu)
    Inner step: exact minimisation over the candidate set.
    Outer step: lam <- max(0, lam + mu g); mu <- growth * mu if the violation
    did not shrink by at least 4x.
    """
    U, P = _scores(tab, uw, pw)
    g = p_min - P
    lam, mu = lam0, mu0
    prev_v = np.inf
    trace = []
    i = 0
    for k in range(outer_iters):
        L = (1 - U) + (np.maximum(0.0, lam + mu * g) ** 2 - lam ** 2) / (2 * mu)
        i = int(np.argmin(L))
        v = max(0.0, g[i])
        trace.append({"iter": k + 1, "cand": i, "U": float(U[i]), "P": float(P[i]),
                      "lam": float(lam), "mu": float(mu), "violation": float(v)})
        lam_new = max(0.0, lam + mu * g[i])
        if v > tol and v > 0.25 * prev_v:
            mu *= growth
        prev_v = v
        if v <= tol and abs(lam_new - lam) <= tol:
            break
        lam = lam_new
    feasible = bool(P[i] >= p_min)
    if return_trace:
        return i, feasible, trace
    return i, feasible


def select_min_intervention(tab, p_min, uw=None, pw=None, **alm_kw):
    """Minimal-intervention PSyGenTAB: release unfiltered output (tau = 0, best
    sampling configuration by U) whenever it already satisfies P >= P_min on
    the calibration split; filter only when it does not, then solve the
    constrained problem with the ALM over all candidates."""
    U, P = _scores(tab, uw, pw)
    unf = np.nonzero((tab.tau_q.values == 0.0) & (P >= p_min))[0]
    if len(unf):
        return int(unf[np.argmax(U[unf])]), True
    return select_alm(tab, p_min, uw=uw, pw=pw, **alm_kw)
