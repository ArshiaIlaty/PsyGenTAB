"""Recompute only `tail_preservation` for every saved release.

The metric now skips columns whose real 5%/95% tails hold < 2% of rows (heavily
tied columns made the ratio explode); all other metrics are unchanged.
    python -m scripts.fix_tail_metric
"""
import glob
import json
import os
import re
import warnings
from multiprocessing import Pool

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def do(run_dir):
    import pandas as pd
    from psygentab import metrics as M
    from psygentab.datasets import harmonise, load_split
    ds = run_dir.split(os.sep)[-3]
    split, _ = map(int, re.match(r"split(\d+)_seed(\d+)", os.path.basename(run_dir)).groups())
    js = [p for p in glob.glob(os.path.join(run_dir, "*.json"))
          if os.path.basename(p) not in ("fit.json", "alm_trace.json") and os.path.exists(p[:-5] + ".csv.gz")]
    if not js:
        return 0
    train, _, _, meta = load_split(ds, split)
    for p in js:
        d = json.load(open(p))
        syn = harmonise(pd.read_csv(p[:-5] + ".csv.gz"), meta)
        d["tail_preservation"] = M.diversity_metrics(train, syn, meta)["tail_preservation"]
        with open(p, "w") as f:
            json.dump(d, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return len(js)


if __name__ == "__main__":
    runs = sorted(glob.glob(os.path.join(ROOT, "results", "runs", "*", "*", "split*_seed*")))
    with Pool(6) as pool:
        print("updated", sum(pool.imap_unordered(do, runs)), "release files")
