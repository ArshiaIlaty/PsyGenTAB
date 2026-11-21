"""Recompute test-split proximity metrics after the equal-reference-size fix.

`privacy_report` compares each synthetic record's distance to a train subset
and to a holdout, and must use reference sets of equal size. It subsampled
train to the holdout size but never the holdout to the train size, so when the
holdout was larger (test splits of Adult, Diabetes and VN Banking: 3,000 rows
vs. the 2,000-row train cap) the share of records closer to train was biased
below 0.5 and test P was biased down. Calibration (val, 1,500 rows) was always
smaller than train, so every selection is unchanged; only test-side numbers
move. No generator is refit:

  * released datasets (<method>.csv.gz) -> proximity keys of <method>.json
  * candidate pools (pool<k>.csv.gz)    -> test_* proximity columns of
    candidates.csv (candidate rows are rebuilt with the original RNG streams
    and verified against the stored val-split scores before anything is written)

Originals are copied to results/_pre_rescore/ first.

    python -m scripts.rescore_holdout_size [--dataset adult ...]
"""
import argparse
import glob
import json
import os
import shutil
import types

import numpy as np
import pandas as pd

from psygentab import core
from psygentab import metrics as M
from psygentab.datasets import harmonise, load_split
from psygentab.encoding import Encoder, exact_keys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
BACKUP = os.path.join(RES, "_pre_rescore")
AFFECTED = ["adult", "diabetes", "vn_banking"]
PROX = ["dcr_syn_median", "dcr_syn_p5", "dcr_hold_median", "dcr_hold_p5", "dcr_ratio", "nndr_syn_median",
        "nndr_hold_median", "nndr_ratio", "share_closer_train", "exact_match_syn", "exact_match_hold",
        "P_dcr", "P_nndr", "P_em"]


class Mismatch(RuntimeError):
    pass


def ctx(train, hold, meta, enc, seed):
    c = types.SimpleNamespace(meta=meta, X_train=enc.transform(train), X_hold=enc.transform(hold),
                              k_train=exact_keys(train, meta), k_hold=exact_keys(hold, meta), seed=seed)
    rng = np.random.default_rng(seed)
    c.tref_idx = rng.choice(len(train), min(len(hold), len(train)), replace=False)
    return c


def backup(path):
    dst = os.path.join(BACKUP, os.path.relpath(path, RES))
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(path, dst)


def rescore_run(run_dir, split, seed, train, test, val, meta, enc):
    eval_seed = 100 * split + seed
    X_tr, X_te = enc.transform(train), enc.transform(test)
    k_tr, k_te = exact_keys(train, meta), exact_keys(test, meta)
    changed = []
    changed = []
    # 2. candidate table (non-DP runs)
    cpath = os.path.join(run_dir, "candidates.csv")
    pools = sorted(glob.glob(os.path.join(run_dir, "pool*.csv.gz")))
    if os.path.exists(cpath) and pools:
        tab = pd.read_csv(cpath)
        vctx, tctx = ctx(train, val, meta, enc, eval_seed), ctx(train, test, meta, enc, eval_seed)
        n_out = len(train)
        new = {}
        for ci, p in enumerate(pools):
            pool = harmonise(pd.read_csv(p, float_precision="round_trip"), meta)
            Xp = enc.transform(pool)
            rho = M._nn(vctx.X_train[vctx.tref_idx], Xp, 1)[:, 0] / (M._nn(vctx.X_hold, Xp, 1)[:, 0] + 1e-12)
            for _, r in tab[tab.config_idx == ci].iterrows():
                q = float(r.tau_q)
                acc = np.nonzero(rho >= float(np.quantile(rho, q)))[0] if q > 0 else np.arange(len(pool))
                rng = np.random.default_rng(eval_seed + 7919 * ci + int(1000 * q))
                pick = rng.choice(acc, n_out, replace=False)
                syn = pool.iloc[pick].reset_index(drop=True)
                Xs = Xp[pick]
                v = core.RealContext.privacy_components(vctx, syn, Xs)
                # exact reproduction is required. Integer-valued data have many exactly
                # tied distances, which other CPUs/BLAS builds can break differently, so
                # a run must be rescored on the machine that scored it (see --only)
                if not np.isclose(v["share_closer_train"], r.share_closer_train, rtol=0, atol=1e-9):
                    raise Mismatch(f"{run_dir} cand {r.cand}: rebuilt candidate does not match "
                                       f"(val share {v['share_closer_train']} vs {r.share_closer_train})")
                t = core.RealContext.privacy_components(tctx, syn, enc.transform(syn))
                new[int(r.cand)] = {f"test_{k}": t[k] for k in PROX}
        if len(new) != len(tab):
            raise RuntimeError(f"{run_dir}: rebuilt {len(new)} of {len(tab)} candidates")
        backup(cpath)
        upd = pd.DataFrame.from_dict(new, orient="index")
        tab = tab.set_index("cand")
        tab.loc[upd.index, upd.columns] = upd
        tab.reset_index().to_csv(cpath, index=False)
    # 1. released datasets
    for js in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        name = os.path.basename(js)[:-5]
        csv = os.path.join(run_dir, name + ".csv.gz")
        if name in ("fit", "alm_trace") or not os.path.exists(csv):
            continue
        res = json.load(open(js))
        syn = harmonise(pd.read_csv(csv, float_precision="round_trip"), meta)
        rep = M.privacy_report(enc.transform(syn), X_tr, X_te, exact_keys(syn, meta), k_tr, k_te, seed=eval_seed)
        rep.update(M.privacy_score(rep))
        old = res.get("share_closer_train")
        backup(js)
        res.update(rep)
        with open(js, "w") as f:
            json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        changed.append((name, old, rep["share_closer_train"]))
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", default=AFFECTED)
    ap.add_argument("--only", help="file listing run directories (relative to results/) to rescore")
    ap.add_argument("--failed", default=os.path.join(RES, "_pre_rescore", "failed.txt"))
    a = ap.parse_args()
    only = set(open(a.only).read().split()) if a.only else None
    failed = []
    for d in a.dataset:
        cache = {}
        for run_dir in sorted(glob.glob(os.path.join(RES, "runs", d, "*", "split*_seed*"))):
            if not os.path.exists(os.path.join(run_dir, "DONE")):
                continue
            rel = os.path.relpath(run_dir, RES)
            if only is not None and rel not in only:
                continue
            tag = os.path.basename(run_dir)
            split, seed = int(tag.split("_")[0][5:]), int(tag.split("seed")[1])
            if split not in cache:
                train, val, test, meta = load_split(d, split)
                cache[split] = (train, val, test, meta, Encoder(meta).fit(train))
            train, val, test, meta, enc = cache[split]
            try:
                ch = rescore_run(run_dir, split, seed, train, test, val, meta, enc)
            except Mismatch as e:
                print("MISMATCH (left untouched):", e, flush=True)
                failed.append(rel)
                continue
            s = "; ".join(f"{n} {o:.3f}->{x:.3f}" for n, o, x in ch[:2] if o is not None)
            print(f"{d}/{os.path.relpath(run_dir, os.path.join(RES, 'runs', d))}: {len(ch)} releases. {s}", flush=True)
    os.makedirs(os.path.dirname(a.failed) or ".", exist_ok=True)
    with open(a.failed, "w") as f:
        f.write("\n".join(failed) + ("\n" if failed else ""))
    print(f"rescored; {len(failed)} runs need the machine that scored them -> {a.failed}", flush=True)


if __name__ == "__main__":
    main()
