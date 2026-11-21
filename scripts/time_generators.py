"""Time fit + sampling of each generator on one dataset split (budget planning)."""
import argparse
import json
import time
import warnings

warnings.filterwarnings("ignore")

from psygentab.datasets import load_split
from psygentab.generators import make

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="adult")
ap.add_argument("--methods", nargs="+", required=True)
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--n_sample", type=int, default=2000)
args = ap.parse_args()

tr, va, te, meta = load_split(args.dataset, 0)
for m in args.methods:
    g = make(m)
    if args.epochs is not None and hasattr(g, "epochs"):
        g.epochs = args.epochs
    if args.epochs is not None and hasattr(g, "n_iter"):
        g.n_iter = args.epochs
    t = time.time()
    try:
        g.fit(tr, meta, seed=0)
        ft = time.time() - t
        t = time.time()
        s = g.sample(args.n_sample, g.configs()[0])
        st = time.time() - t
        rec = dict(method=m, dataset=args.dataset, n_train=len(tr), fit_s=round(ft, 1),
                   sample_s=round(st, 1), n_out=len(s), info=getattr(g, "fit_info", {}))
    except Exception as e:
        rec = dict(method=m, error=f"{type(e).__name__}: {e}"[:300], elapsed=round(time.time() - t, 1))
    print("TIMING", json.dumps(rec, default=str), flush=True)
