"""Full evaluation of one released synthetic dataset against the test split."""
import time

from . import metrics as M
from .encoding import exact_keys

TSTR_MODELS = ("catboost", "hgb")  # catboost is the pre-declared primary model


def full_eval(syn, train, val, test, meta, enc, enc_x, seed=0, attacks=True, trtr_cache=None):
    t0 = time.time()
    out = {"n_syn": len(syn)}
    X_tr, X_te, X_va = enc.transform(train), enc.transform(test), enc.transform(val)
    X_s = enc.transform(syn)

    # fidelity (real reference = unseen test data)
    out.update(M.marginal_scores(test, syn, meta))
    out["wasserstein_norm"] = M.wasserstein_norm(test, syn, meta, ref=train)
    out["assoc_err"] = M.association_error(test, syn, meta)
    out["c2st_auc"] = M.c2st_auc(X_te, X_s, seed=seed)
    out.update(M.prdc(X_te, X_s, seed=seed))
    out.update(M.alpha_beta_auth(X_te, X_s, seed=seed))

    # utility: TSTR on the real test split
    for m in TSTR_MODELS:
        for k, v in M.tstr(syn, test, meta, enc_x, model=m, seed=seed).items():
            out[f"tstr_{m}_{k}"] = v

    # fairness (subgroup representation / utility), diversity, mean JS divergence
    out.update(M.fairness_diversity(syn, train, test, meta, enc_x, seed=seed, trtr_cache=trtr_cache))

    # privacy: proximity vs. holdout baseline
    rep = M.privacy_report(X_s, X_tr, X_te, exact_keys(syn, meta), exact_keys(train, meta),
                           exact_keys(test, meta), seed=seed)
    out.update(rep)
    out.update(M.privacy_score(rep))

    if attacks:
        Xe, ye = M.mia_members(X_tr, X_te, seed=seed)
        out.update(M.distance_mia(X_s, Xe, ye))
        out.update(M.domias_kde(X_s, X_va, Xe, ye, seed=seed))
        out.update(M.anonymeter_risks(train, syn, test, meta, seed=seed))
    out["eval_seconds"] = round(time.time() - t0, 1)
    return out


def trtr(train, test, meta, enc_x, seed=0):
    out = {}
    for m in TSTR_MODELS:
        for k, v in M.tstr(train, test, meta, enc_x, model=m, seed=seed).items():
            out[f"trtr_{m}_{k}"] = v
    return out
