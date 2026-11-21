"""TAPAS-style shadow-modelling membership inference (Houssiau et al., 2022).

Threat model: the attacker knows the generator's training algorithm and
hyper-parameters, holds auxiliary data from the same population, and targets
specific records x*. For each "world" a training set of size m is drawn and
each target is inserted independently with probability 1/2, so a single
generator fit yields one labelled example per target.

  shadow worlds : base rows from the attacker's auxiliary pool (val + test)
  eval worlds   : base rows from the custodian's real train split
Targets are the most isolated train records (mean 5-NN distance), the
records most at risk (Meeus et al., 2023).

Features of world w for target x* (TAPAS "closest-distance" family plus
density counts): sorted distances from x* to its 10 nearest synthetic rows and
the number of synthetic rows within radii r_q (aux-calibrated quantiles).
A random forest per target is trained on the shadow worlds and scored (AUC)
on the eval worlds. The same fitted world is released raw and after
PSyGenTAB, so both pipelines face the identical attack.
"""
import argparse
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from psygentab import core
from psygentab.datasets import load_split
from psygentab.encoding import Encoder
from psygentab.generators import make

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "tapas")


def world_features(X_syn, X_targets, radii):
    nn = NearestNeighbors(n_neighbors=min(10, len(X_syn))).fit(X_syn)
    d, _ = nn.kneighbors(X_targets)
    dall = NearestNeighbors(n_neighbors=min(200, len(X_syn))).fit(X_syn).kneighbors(X_targets)[0]
    counts = np.stack([(dall <= r).sum(1) for r in radii], axis=1)
    return np.hstack([d, counts])


def psygentab_release(gen, train_w, val, meta, enc, enc_x, n_out, p_min, seed):
    pools = [(c, gen.sample(int(2.0 * n_out), c)) for c in gen.configs()]
    ctx = core.RealContext(train_w, val, meta, enc, enc_x, seed=seed)
    tab, subsets = core.candidate_table(pools, n_out, ctx, ctx, seed=seed)
    d_idx = next(i for i, (c, _) in enumerate(pools) if c.get("temperature", 1.0) == 1.0)
    raw = subsets[core.select_default(tab, d_idx)]
    i, _ = core.select_alm(tab, p_min)
    return raw, subsets[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="heart_failure")
    ap.add_argument("--generator", required=True)
    ap.add_argument("--n_targets", type=int, default=5)
    ap.add_argument("--n_shadow", type=int, default=40)
    ap.add_argument("--n_eval", type=int, default=20)
    ap.add_argument("--m", type=int, default=150, help="world training-set size")
    ap.add_argument("--p_min", type=float, default=0.90)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = f"{a.dataset}__{a.generator}"
    feat_path = os.path.join(OUT, f"{tag}__features.npz")

    train, val, test, meta = load_split(a.dataset, 0)
    enc = Encoder(meta).fit(train)
    enc_x = Encoder(meta, include_target=False).fit(train)
    X_tr = enc.transform(train)
    iso = NearestNeighbors(n_neighbors=6).fit(X_tr).kneighbors(X_tr)[0][:, 1:].mean(1)
    t_idx = np.argsort(-iso)[: a.n_targets]
    targets = train.iloc[t_idx].reset_index(drop=True)
    X_t = enc.transform(targets)
    aux = pd.concat([val, test]).reset_index(drop=True)
    base_eval = train.drop(index=t_idx).reset_index(drop=True)
    X_aux = enc.transform(aux)
    dd = NearestNeighbors(n_neighbors=2).fit(X_aux).kneighbors(X_aux)[0][:, 1]
    radii = np.quantile(dd, [0.05, 0.25, 0.5])
    m = min(a.m, len(aux) - a.n_targets, len(base_eval) - a.n_targets)

    rng = np.random.default_rng(12345)
    done = dict(np.load(feat_path, allow_pickle=True)) if os.path.exists(feat_path) else {}
    rows = done.get("rows", np.zeros((0, 3 + 2 * a.n_targets * (10 + len(radii)))))
    rows = list(rows)
    n_total = a.n_shadow + a.n_eval
    for w in range(n_total):
        kind = 0 if w < a.n_shadow else 1  # 0 = shadow, 1 = eval
        inc = rng.random(a.n_targets) < 0.5
        pool = aux if kind == 0 else base_eval
        base = pool.sample(m - int(inc.sum()), random_state=int(rng.integers(1 << 31)))
        if w < len(rows):
            continue
        train_w = pd.concat([base, targets[inc]]).sample(frac=1, random_state=w).reset_index(drop=True)
        gen = make(a.generator)
        if a.epochs and hasattr(gen, "epochs"):
            gen.epochs = a.epochs
        if a.epochs and hasattr(gen, "n_iter"):
            gen.n_iter = a.epochs
        t0 = time.time()
        gen.fit(train_w, meta, seed=w)
        raw, prot = psygentab_release(gen, train_w, val, meta, enc, enc_x, len(train_w), a.p_min, seed=w)
        f_raw = world_features(enc.transform(raw), X_t, radii)
        f_prot = world_features(enc.transform(prot), X_t, radii)
        rows.append(np.r_[w, kind, time.time() - t0, inc.astype(float), f_raw.ravel(), f_prot.ravel()])
        np.savez(feat_path, rows=np.array(rows))
        print(f"world {w + 1}/{n_total} kind={'shadow' if kind == 0 else 'eval'} "
              f"in={inc.astype(int).tolist()} {time.time() - t0:.0f}s", flush=True)

    R = np.array(rows)
    kinds, labels = R[:, 1], R[:, 3:3 + a.n_targets]
    nf = 10 + len(radii)
    F = R[:, 3 + a.n_targets:]
    F_raw = F[:, : a.n_targets * nf].reshape(len(R), a.n_targets, nf)
    F_prot = F[:, a.n_targets * nf:].reshape(len(R), a.n_targets, nf)
    res = {"dataset": a.dataset, "generator": a.generator, "m": m, "n_shadow": a.n_shadow,
           "n_eval": a.n_eval, "p_min": a.p_min, "targets": t_idx.tolist()}
    for name, Fx in [("raw", F_raw), ("psygentab", F_prot)]:
        aucs = []
        for t in range(a.n_targets):
            sh, ev = kinds == 0, kinds == 1
            y_sh, y_ev = labels[sh, t], labels[ev, t]
            if len(np.unique(y_sh)) < 2 or len(np.unique(y_ev)) < 2:
                continue
            clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=t)
            clf.fit(Fx[sh, t], y_sh)
            aucs.append(roc_auc_score(y_ev, clf.predict_proba(Fx[ev, t])[:, 1]))
        res[f"{name}_auc_per_target"] = aucs
        res[f"{name}_auc_mean"] = float(np.mean(aucs)) if aucs else None
    json.dump(res, open(os.path.join(OUT, f"{tag}.json"), "w"), indent=1)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
