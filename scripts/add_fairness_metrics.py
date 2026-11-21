"""Post-hoc: add fairness / diversity / mean-JS metrics to every saved release.

Reads results/runs/<ds>/<gen>/split*_seed*/<method>.csv.gz, recomputes the
run's split, and merges the new keys into <method>.json (idempotent: files
that already have them are skipped). New runs compute them in full_eval.
    python -m scripts.add_fairness_metrics --workers 4
"""
import argparse
import glob
import json
import os
import re
import warnings
from multiprocessing import Pool

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = "js_mean"


FORCE = False


def do_run(run_dir):
    import pandas as pd
    from psygentab import metrics as M
    from psygentab.datasets import harmonise, load_split
    from psygentab.encoding import Encoder
    ds = run_dir.split(os.sep)[-3]
    split, seed = map(int, re.match(r"split(\d+)_seed(\d+)", os.path.basename(run_dir)).groups())
    todo = []
    for jp in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        if os.path.basename(jp) in ("fit.json", "alm_trace.json"):
            continue
        d = json.load(open(jp))
        if (FORCE or KEY not in d) and os.path.exists(jp[:-5] + ".csv.gz"):
            todo.append((jp, d))
    if not todo:
        return run_dir, 0
    train, _, test, meta = load_split(ds, split)
    enc_x = Encoder(meta, include_target=False).fit(train)
    cache = {}
    for jp, d in todo:
        syn = harmonise(pd.read_csv(jp[:-5] + ".csv.gz"), meta)
        d.update(M.fairness_diversity(syn, train, test, meta, enc_x, seed=100 * split + seed, trtr_cache=cache))
        with open(jp, "w") as f:
            json.dump(d, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return run_dir, len(todo)


def _init(force):
    global FORCE
    FORCE = force


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="recompute files that already have the metrics")
    a = ap.parse_args()
    global FORCE
    FORCE = a.force
    runs = sorted(glob.glob(os.path.join(ROOT, "results", "runs", "*", "*", "split*_seed*")))
    total = 0
    with Pool(a.workers, initializer=_init, initargs=(a.force,)) as pool:
        for k, (r, n) in enumerate(pool.imap_unordered(do_run, runs)):
            total += n
            print(f"{k + 1}/{len(runs)} {os.path.relpath(r, ROOT)} +{n}", flush=True)
    print("updated", total, "release files")


if __name__ == "__main__":
    main()
