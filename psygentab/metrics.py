"""Fidelity, utility and privacy metrics.

All distance-based quantities use the shared `Encoder` fit on real train.
Reported numbers always use the untouched test split as the real holdout;
PSyGenTAB's calibration calls the same functions with the val split.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from .encoding import Encoder, exact_keys

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- fidelity

def marginal_scores(real, syn, meta):
    """Per-column KS (numerical) and TVD (categorical); lower is better."""
    ks, tvd = [], []
    for c in meta["numerical"]:
        r, s = real[c].dropna().values, syn[c].dropna().values
        if len(r) and len(s):
            ks.append(stats.ks_2samp(r, s).statistic)
    for c in meta["categorical"] + [meta["target"]]:
        p = real[c].value_counts(normalize=True)
        q = syn[c].value_counts(normalize=True)
        tvd.append(0.5 * p.subtract(q, fill_value=0).abs().sum())
    return {"ks_mean": float(np.mean(ks)) if ks else np.nan,
            "tvd_mean": float(np.mean(tvd)) if tvd else np.nan,
            "marginal_err": float(np.mean(ks + tvd))}


def wasserstein_norm(real, syn, meta, ref):
    """Mean 1-Wasserstein over numericals after min-max scaling by `ref` range."""
    out = []
    for c in meta["numerical"]:
        lo, hi = ref[c].min(), ref[c].max()
        if not np.isfinite(hi - lo) or hi == lo:
            continue
        r = (real[c].dropna().values - lo) / (hi - lo)
        s = (syn[c].dropna().values - lo) / (hi - lo)
        if len(r) and len(s):
            out.append(stats.wasserstein_distance(r, s))
    return float(np.mean(out)) if out else np.nan


def _cramers_v(x, y):
    ct = pd.crosstab(x, y).values
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.sum()
    r, k = ct.shape
    # Bergsma (2013) bias correction
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    d = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2 / d)) if d > 0 else 0.0


def _corr_ratio(cat, num):
    num = np.asarray(num, dtype=float)
    mu = num.mean()
    den = ((num - mu) ** 2).sum()
    if den == 0:
        return 0.0
    num_s = pd.Series(num)
    g = num_s.groupby(np.asarray(cat))
    between = (g.count() * (g.mean() - mu) ** 2).sum()
    return float(np.sqrt(between / den))


def association_matrix(df, meta, max_rows=3000, seed=0):
    """Pearson (num-num), bias-corrected Cramer's V (cat-cat) and correlation
    ratio (num-cat), including the target."""
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
    nums = meta["numerical"]
    cats = meta["categorical"] + [meta["target"]]
    cols = nums + cats
    X = df[cols].copy()
    for c in nums:
        X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0.0)
    k = len(cols)
    A = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            a, b = cols[i], cols[j]
            if a in nums and b in nums:
                v = X[a].corr(X[b]) if X[a].std() > 0 and X[b].std() > 0 else 0.0
            elif a in cats and b in cats:
                v = _cramers_v(X[a], X[b])
            else:
                cat, num = (a, b) if a in cats else (b, a)
                v = _corr_ratio(X[cat].values, X[num].values)
            A[i, j] = A[j, i] = 0.0 if not np.isfinite(v) else v
    return A


def association_error(real, syn, meta, A_real=None):
    A_r = association_matrix(real, meta) if A_real is None else A_real
    A_s = association_matrix(syn, meta)
    iu = np.triu_indices_from(A_r, k=1)
    return float(np.mean(np.abs(A_r[iu] - A_s[iu]))) if len(iu[0]) else 0.0


def c2st_auc(X_real, X_syn, seed=0, folds=5, fast=False):
    """Classifier two-sample test (detection) AUC; 0.5 is indistinguishable."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    n = min(len(X_real), len(X_syn))
    Xr = X_real[rng.choice(len(X_real), n, replace=False)]
    Xs = X_syn[rng.choice(len(X_syn), n, replace=False)]
    X = np.vstack([Xr, Xs])
    y = np.r_[np.ones(n), np.zeros(n)]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = (LogisticRegression(max_iter=500) if fast else
               HistGradientBoostingClassifier(max_iter=200, random_state=seed))
        clf.fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, p))


def prdc(X_real, X_syn, k=5, max_n=3000, seed=0):
    """Precision / recall / density / coverage (Naeem et al., ICML 2020)."""
    rng = np.random.default_rng(seed)
    if len(X_real) > max_n:
        X_real = X_real[rng.choice(len(X_real), max_n, replace=False)]
    if len(X_syn) > max_n:
        X_syn = X_syn[rng.choice(len(X_syn), max_n, replace=False)]
    k = min(k, len(X_real) - 1, len(X_syn) - 1)
    rr = NearestNeighbors(n_neighbors=k + 1).fit(X_real).kneighbors(X_real)[0][:, -1]
    ss = NearestNeighbors(n_neighbors=k + 1).fit(X_syn).kneighbors(X_syn)[0][:, -1]
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X_real, X_syn)  # [n_real, n_syn]
    precision = float((D < rr[:, None]).any(axis=0).mean())
    recall = float((D < ss[None, :]).any(axis=1).mean())
    density = float((1.0 / k) * (D < rr[:, None]).sum(axis=0).mean())
    coverage = float((D.min(axis=1) < rr).mean())
    return {"prdc_precision": precision, "prdc_recall": recall,
            "prdc_density": density, "prdc_coverage": coverage}


def alpha_beta_auth(X_real, X_syn, max_n=3000, seed=0, n_alpha=30):
    """alpha-precision, beta-recall and authenticity (Alaa et al., ICML 2022),
    computed in the shared encoded space with a mean-centred hypersphere
    (the 'naive' variant without the one-class embedding network)."""
    rng = np.random.default_rng(seed)
    if len(X_real) > max_n:
        X_real = X_real[rng.choice(len(X_real), max_n, replace=False)]
    if len(X_syn) > max_n:
        X_syn = X_syn[rng.choice(len(X_syn), max_n, replace=False)]
    c_r = X_real.mean(0)
    c_s = X_syn.mean(0)
    dr = np.linalg.norm(X_real - c_r, axis=1)
    ds_to_r = np.linalg.norm(X_syn - c_r, axis=1)
    ds = np.linalg.norm(X_syn - c_s, axis=1)
    alphas = np.linspace(0, 1, n_alpha)
    # alpha-precision: share of synthetic samples inside the real alpha-support
    pa = np.array([(ds_to_r <= np.quantile(dr, a)).mean() for a in alphas])
    # beta-recall: share of real samples inside the synthetic beta-support,
    # using the k-NN-in-support definition of Alaa et al.
    nn_s = NearestNeighbors(n_neighbors=1).fit(X_syn)
    d_rs, idx = nn_s.kneighbors(X_real)
    nn_r = NearestNeighbors(n_neighbors=2).fit(X_real)
    k_r = nn_r.kneighbors(X_real)[0][:, 1]
    rb = []
    for b in alphas:
        inside = ds[idx[:, 0]] <= np.quantile(ds, b)
        rb.append((inside & (d_rs[:, 0] <= k_r)).mean())
    rb = np.array(rb)
    ip = 1 - 2 * np.mean(np.abs(pa - alphas))
    ir = 1 - 2 * np.mean(np.abs(rb - alphas))
    # authenticity: synthetic sample is unauthentic if it is closer to its
    # nearest real record than that record is to its own nearest real neighbour
    d_sr, j = NearestNeighbors(n_neighbors=1).fit(X_real).kneighbors(X_syn)
    auth = float((d_sr[:, 0] > k_r[j[:, 0]]).mean())
    return {"alpha_precision": float(ip), "beta_recall": float(ir), "authenticity": auth}


# ---------------------------------------------------------------- utility

def _xy(df, enc_x, meta):
    X = enc_x.transform(df)
    y = df[meta["target"]].astype(str).map({c: i for i, c in enumerate(meta["classes"])})
    return X, y.fillna(-1).astype(int).values


def _fit_predict(model_name, Xtr, ytr, Xte, n_classes, seed):
    present = np.unique(ytr[ytr >= 0])
    keep = ytr >= 0
    Xtr, ytr = Xtr[keep], ytr[keep]
    P = np.zeros((len(Xte), n_classes))
    if len(present) == 1:
        P[:, present[0]] = 1.0
        return P
    remap = {c: i for i, c in enumerate(present)}
    y_local = np.array([remap[v] for v in ytr])
    # XGBoost is not used: its Homebrew libomp deadlocks with torch's libomp
    # in the same process on macOS.
    if model_name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=seed)
    elif model_name == "hgb_fast":
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=seed)
    elif model_name == "catboost":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=500, depth=6, random_seed=seed,
                               verbose=0, thread_count=4)
    elif model_name == "logreg":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=1000)
    else:
        raise ValueError(model_name)
    m.fit(Xtr, y_local)
    P[:, present] = m.predict_proba(Xte)
    return P


def classification_scores(y, P):
    n_classes = P.shape[1]
    pred = P.argmax(1)
    out = {"f1_macro": float(f1_score(y, pred, average="macro", labels=range(n_classes), zero_division=0)),
           "bal_acc": float(balanced_accuracy_score(y, pred))}
    try:
        if n_classes == 2:
            out["auroc"] = float(roc_auc_score(y, P[:, 1]))
        else:
            present = np.unique(y)
            Q = P[:, present] + 1e-12
            Q = Q / Q.sum(1, keepdims=True)
            out["auroc"] = float(roc_auc_score(y, Q, multi_class="ovr", average="macro", labels=present)
                                 if len(present) > 2 else roc_auc_score(y == present[1], Q[:, 1]))
    except ValueError:
        out["auroc"] = np.nan
    return out


def tstr(train_df, test_df, meta, enc_x, model="catboost", seed=0):
    """Train on `train_df` (synthetic for TSTR, real for TRTR), test on real test."""
    Xtr, ytr = _xy(train_df, enc_x, meta)
    Xte, yte = _xy(test_df, enc_x, meta)
    P = _fit_predict(model, Xtr, ytr, Xte, len(meta["classes"]), seed)
    return classification_scores(yte, P)


# ---------------------------------------------------------------- privacy

def _nn(X_fit, X_query, k=1):
    d, _ = NearestNeighbors(n_neighbors=k).fit(X_fit).kneighbors(X_query)
    return d


def privacy_report(X_syn, X_train, X_hold, keys_syn, keys_train, keys_hold, seed=0):
    """DCR / NNDR / exact-match against a real-holdout baseline.

    share_closer_train: fraction of synthetic records strictly closer to an
    equal-size random subset of train than to the holdout (ties count 1/2);
    0.5 is the ideal for a generator that does not memorise.
    """
    rng = np.random.default_rng(seed)
    n_h = len(X_hold)
    m = min(n_h, len(X_train))
    tr_ref = X_train[rng.choice(len(X_train), m, replace=False)]
    # both reference sets must have the same size, otherwise the larger one is
    # "closer" to every record and the share is biased away from 0.5. The
    # holdout is subsampled only when it is the larger set (test splits of the
    # big datasets: 3,000 vs a 2,000-row train cap), after the train draw, so
    # every equal-or-smaller-holdout case is bit-identical to before.
    h_ref = X_hold if n_h == m else X_hold[rng.choice(n_h, m, replace=False)]
    d_st = _nn(X_train, X_syn, 2)
    d_ht = _nn(X_train, X_hold, 2)
    d_s_tref = _nn(tr_ref, X_syn, 1)[:, 0]
    d_s_hold = _nn(h_ref, X_syn, 1)[:, 0]
    share = float(((d_s_tref < d_s_hold) + 0.5 * (d_s_tref == d_s_hold)).mean())
    eps = 1e-12
    nndr_s = d_st[:, 0] / (d_st[:, 1] + eps)
    nndr_h = d_ht[:, 0] / (d_ht[:, 1] + eps)
    tr_set = set(keys_train.tolist())
    em_s = float(np.mean([k in tr_set for k in keys_syn]))
    em_h = float(np.mean([k in tr_set for k in keys_hold]))
    out = {
        "dcr_syn_median": float(np.median(d_st[:, 0])),
        "dcr_syn_p5": float(np.quantile(d_st[:, 0], 0.05)),
        "dcr_hold_median": float(np.median(d_ht[:, 0])),
        "dcr_hold_p5": float(np.quantile(d_ht[:, 0], 0.05)),
        "dcr_ratio": float(np.median(d_st[:, 0]) / (np.median(d_ht[:, 0]) + eps)),
        "nndr_syn_median": float(np.median(nndr_s)),
        "nndr_hold_median": float(np.median(nndr_h)),
        "nndr_ratio": float(np.median(nndr_s) / (np.median(nndr_h) + eps)),
        "share_closer_train": share,
        "exact_match_syn": em_s,
        "exact_match_hold": em_h,
    }
    return out


DEFAULT_PW = {"dcr": 0.5, "nndr": 0.25, "em": 0.25}


def privacy_score(rep, weights=None):
    """Composite privacy-preservability P in [0, 1], holdout-calibrated.

    P_dcr  = 1 - 2 |share_closer_train - 0.5|           (two-sided: records pushed
             *away* from training points are also a membership signal)
    P_nndr = clip(nndr_ratio, 0, 1)                      (1 when synthetic NNDR >= holdout NNDR)
    P_em   = 1 - clip((EM_syn - EM_hold) / (1 - EM_hold), 0, 1)
    """
    w = DEFAULT_PW if weights is None else weights
    p_dcr = float(np.clip(1 - 2 * abs(rep["share_closer_train"] - 0.5), 0, 1))
    p_nndr = float(np.clip(rep["nndr_ratio"], 0, 1))
    denom = max(1e-9, 1 - rep["exact_match_hold"])
    p_em = 1 - float(np.clip((rep["exact_match_syn"] - rep["exact_match_hold"]) / denom, 0, 1))
    tot = sum(w.values())
    P = (w["dcr"] * p_dcr + w["nndr"] * p_nndr + w["em"] * p_em) / tot
    return {"P": float(P), "P_dcr": p_dcr, "P_nndr": p_nndr, "P_em": p_em}


# ---------------------------------------------------------------- attacks

def _auc_tpr(scores, labels):
    auc = float(roc_auc_score(labels, scores))
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    tpr_at = float(np.interp(0.1, fpr, tpr))
    return auc, tpr_at


def mia_members(X_train, X_test, seed=0):
    """Balanced member / non-member evaluation sets."""
    rng = np.random.default_rng(seed)
    m = min(len(X_train), len(X_test))
    Xm = X_train[rng.choice(len(X_train), m, replace=False)]
    Xn = X_test[rng.choice(len(X_test), m, replace=False)]
    return np.vstack([Xm, Xn]), np.r_[np.ones(m), np.zeros(m)]


def distance_mia(X_syn, X_eval, y_eval):
    """Black-box distance attack: closer to the synthetic data => member."""
    d = _nn(X_syn, X_eval, 1)[:, 0]
    auc, tpr = _auc_tpr(-d, y_eval)
    return {"mia_dist_auc": auc, "mia_dist_tpr@fpr0.1": tpr}


def domias_kde(X_syn, X_ref, X_eval, y_eval, max_dim=10, seed=0):
    """DOMIAS (van Breugel et al., AISTATS 2023), KDE density variant:
    score(x) = p_syn(x) / p_ref(x), estimated in a PCA space fit on X_ref."""
    from sklearn.decomposition import PCA
    k = int(min(max_dim, X_ref.shape[1], len(X_ref) - 1))
    pca = PCA(n_components=k, random_state=seed).fit(X_ref)
    zs, zr, ze = pca.transform(X_syn), pca.transform(X_ref), pca.transform(X_eval)
    # tiny jitter keeps the KDE covariance non-singular on discrete data
    rng = np.random.default_rng(seed)
    zs = zs + 1e-6 * rng.standard_normal(zs.shape)
    zr = zr + 1e-6 * rng.standard_normal(zr.shape)
    try:
        ps = stats.gaussian_kde(zs.T)(ze.T)
        pr = stats.gaussian_kde(zr.T)(ze.T)
    except np.linalg.LinAlgError:
        return {"mia_domias_auc": np.nan, "mia_domias_tpr@fpr0.1": np.nan}
    auc, tpr = _auc_tpr(ps / (pr + 1e-300), y_eval)
    return {"mia_domias_auc": auc, "mia_domias_tpr@fpr0.1": tpr}


def anonymeter_risks(train, syn, test, meta, n_attacks=300, seed=0):
    """Attribute-inference and linkability risks (Giomi et al., PoPETs 2023),
    each corrected by the control (non-member) baseline."""
    from anonymeter.evaluators import InferenceEvaluator, LinkabilityEvaluator
    cols = meta["columns"]
    n = int(min(n_attacks, len(test), len(syn)))
    out = {}
    tgt = meta["target"]
    aux = [c for c in cols if c != tgt]
    try:
        ev = InferenceEvaluator(ori=train[cols], syn=syn[cols], control=test[cols],
                                aux_cols=aux, secret=tgt, n_attacks=n)
        ev.evaluate(n_jobs=1)
        r = ev.risk()
        out["aia_risk"] = float(r.value)
        res = ev.results()
        out["aia_attack_rate"] = float(res.attack_rate.value)
        out["aia_control_rate"] = float(res.control_rate.value)
    except Exception as e:  # pragma: no cover - reported, not fatal
        out["aia_error"] = f"{type(e).__name__}: {e}"[:200]
    try:
        rng = np.random.default_rng(seed)
        perm = list(rng.permutation(aux))
        half = len(perm) // 2
        ev = LinkabilityEvaluator(ori=train[cols], syn=syn[cols], control=test[cols],
                                  n_attacks=n, aux_cols=(perm[:half], perm[half:]), n_neighbors=10)
        ev.evaluate(n_jobs=1)
        out["link_risk"] = float(ev.risk().value)
    except Exception as e:  # pragma: no cover
        out["link_error"] = f"{type(e).__name__}: {e}"[:200]
    return out


# ---------------------------------------------------------------- fairness & diversity

MIN_GROUP = 30  # subgroup levels need >= this many real rows to be scored


def _shares(s):
    return s.astype(str).value_counts(normalize=True)


def js_divergence_mean(real, syn, meta, bins=20):
    """Mean per-column Jensen-Shannon divergence (base 2, in [0, 1]); numericals
    on shared equal-width bins over the pooled range, categoricals on levels."""
    from scipy.spatial.distance import jensenshannon
    out = []
    for c in meta["numerical"]:
        r, s = real[c].dropna().values, syn[c].dropna().values
        if not len(r) or not len(s):
            continue
        lo, hi = min(r.min(), s.min()), max(r.max(), s.max())
        if hi == lo:
            out.append(0.0)
            continue
        edges = np.linspace(lo, hi, bins + 1)
        p, q = np.histogram(r, edges)[0] + 1e-12, np.histogram(s, edges)[0] + 1e-12
        out.append(jensenshannon(p / p.sum(), q / q.sum(), base=2) ** 2)
    for c in meta["categorical"] + [meta["target"]]:
        p, q = _shares(real[c]), _shares(syn[c])
        lv = p.index.union(q.index)
        out.append(jensenshannon(p.reindex(lv, fill_value=0) + 1e-12, q.reindex(lv, fill_value=0) + 1e-12,
                                 base=2) ** 2)
    return float(np.mean(out)) if out else np.nan


def representation_metrics(real, syn, meta):
    """Subgroup representation: every level (n_real >= MIN_GROUP) of every
    categorical covariate is a subgroup; the target classes are scored separately."""
    ratios, gaps = [], []
    for c in meta["categorical"]:
        p, q = _shares(real[c]), _shares(syn[c])
        cnt = real[c].astype(str).value_counts()
        for lv in cnt[cnt >= MIN_GROUP].index:
            ps, qs = p.get(lv, 0.0), q.get(lv, 0.0)
            ratios.append(min(qs / ps, ps / qs) if qs > 0 else 0.0)
            gaps.append(abs(qs - ps))
    pt, qt = _shares(real[meta["target"]]), _shares(syn[meta["target"]])
    minority = pt.idxmin()
    return {"repr_parity": float(np.mean(ratios)) if ratios else np.nan,
            "stat_parity_diff_max": float(np.max(gaps)) if gaps else np.nan,
            "n_subgroups": len(ratios),
            "minority_class_ratio": float(qt.get(minority, 0.0) / pt[minority])}


def subgroup_utility(syn, train, test, meta, enc_x, seed=0, trtr_cache=None):
    """Worst-subgroup utility: balanced accuracy of a HistGB trained on
    synthetic vs. on real train, both tested on the real test split, within
    every categorical-covariate subgroup with >= MIN_GROUP test rows and >= 2
    classes (HistGB with 100 iterations on both sides of the ratio). Returns the worst and mean ratio TSTR_g / TRTR_g."""
    from sklearn.metrics import balanced_accuracy_score
    groups = []
    for c in meta["categorical"]:
        vals = test[c].astype(str)
        for lv, n in vals.value_counts().items():
            idx = np.nonzero((vals == lv).values)[0]
            if n >= MIN_GROUP and test[meta["target"]].iloc[idx].nunique() >= 2:
                groups.append((c, lv, idx))
    if not groups:
        return {"subgroup_util_ratio_min": np.nan, "subgroup_util_ratio_mean": np.nan, "n_util_subgroups": 0}
    Xte, yte = _xy(test, enc_x, meta)
    k = len(meta["classes"])
    if trtr_cache is not None and "pred" in trtr_cache:
        p_real = trtr_cache["pred"]
    else:
        Xtr, ytr = _xy(train, enc_x, meta)
        p_real = _fit_predict("hgb_fast", Xtr, ytr, Xte, k, seed).argmax(1)
        if trtr_cache is not None:
            trtr_cache["pred"] = p_real
    Xs, ys = _xy(syn, enc_x, meta)
    p_syn = _fit_predict("hgb_fast", Xs, ys, Xte, k, seed).argmax(1)
    ratios = []
    for _, _, idx in groups:
        b_real = balanced_accuracy_score(yte[idx], p_real[idx])
        b_syn = balanced_accuracy_score(yte[idx], p_syn[idx])
        ratios.append(min(1.5, b_syn / max(b_real, 1e-6)))
    return {"subgroup_util_ratio_min": float(np.min(ratios)), "subgroup_util_ratio_mean": float(np.mean(ratios)),
            "n_util_subgroups": len(ratios)}


def diversity_metrics(real, syn, meta, rare=0.05, bins=10):
    cols = meta["columns"]
    keys = exact_keys(syn, meta)
    out = {"uniqueness": float(len(set(keys.tolist())) / max(1, len(keys)))}
    cov, rare_cov = [], []
    for c in meta["categorical"] + [meta["target"]]:
        p = _shares(real[c])
        present = set(syn[c].astype(str))
        cov.append(np.mean([lv in present for lv in p.index]))
        rare_lv = p[p < rare].index
        if len(rare_lv):
            rare_cov.append(np.mean([lv in present for lv in rare_lv]))
    out["category_coverage"] = float(np.mean(cov)) if cov else np.nan
    out["rare_category_coverage"] = float(np.mean(rare_cov)) if rare_cov else np.nan

    def entropy(x):
        p = x / x.sum()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())
    ent, tails = [], []
    for c in cols:
        if c in meta["numerical"]:
            r, s = real[c].dropna().values, syn[c].dropna().values
            if not len(r) or not len(s):
                continue
            edges = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
            if len(edges) < 3:
                continue
            edges[0], edges[-1] = -np.inf, np.inf
            hr, hs = entropy(np.histogram(r, edges)[0]), entropy(np.histogram(s, edges)[0])
            lo, hi = np.quantile(r, [0.05, 0.95])
            real_tail = ((r < lo) | (r > hi)).mean()
            # heavily tied columns can have (almost) no mass beyond their own
            # 5th/95th percentiles; the ratio is undefined there
            if hi > lo and real_tail >= 0.02:
                tails.append(((s < lo) | (s > hi)).mean() / real_tail)
        else:
            hr, hs = entropy(real[c].astype(str).value_counts().values), entropy(syn[c].astype(str).value_counts().values)
        if hr > 0 and hs > 0:
            ent.append(min(hs / hr, hr / hs))
        elif hr == 0 and hs == 0:
            ent.append(1.0)
        else:
            ent.append(0.0)
    out["entropy_preservation"] = float(np.mean(ent)) if ent else np.nan
    # share of synthetic mass in the real 5%/95% tails, relative to real (1 = tails kept)
    out["tail_preservation"] = float(np.mean(tails)) if tails else np.nan
    return out


def fairness_diversity(syn, train, test, meta, enc_x, seed=0, trtr_cache=None):
    """Representation and diversity are judged against the real train split
    (what the generator was asked to reproduce); utility against real test."""
    out = {"js_mean": js_divergence_mean(test, syn, meta)}
    out.update(representation_metrics(train, syn, meta))
    out.update(subgroup_utility(syn, train, test, meta, enc_x, seed=seed, trtr_cache=trtr_cache))
    out.update(diversity_metrics(train, syn, meta))
    return out
