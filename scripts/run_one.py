"""One (dataset, split, generator-seed, generator) run of the PSyGenTAB study.

Outputs under results/runs/<dataset>/<generator>/split<s>_seed<k>/:
  fit.json            generator fit info and timings
  candidates.csv      every (config, tau) candidate scored on val and test
  alm_trace.json      ALM iterates for the main P_min
  <method>.json       full test-split evaluation of each released dataset
  <method>.csv.gz     released synthetic data
  DONE                completion marker (runs are skipped when present)

Methods: <gen> (default sampling), <gen>+filter (DCR filter), <gen>+psygentab
(ALM at P_min), <gen>+psygentab_min (minimal intervention: filter only if the
unfiltered output misses P_min on val). DP generators only release their raw output: any filtering
against the training data would void the DP guarantee.
"""
import argparse
import json
import os
import time
import traceback
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from psygentab import core
from psygentab.datasets import load_split
from psygentab.encoding import Encoder
from psygentab.evaluate import full_eval, trtr
from psygentab.generators import make

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
P_MIN = 0.90
POOL_FACTOR = 2.0
SAVE_MODELS = os.environ.get("PSY_SAVE_MODELS", "1") == "1"


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def ensure_trtr(dataset, split, train, test, meta, enc_x):
    os.makedirs(os.path.join(RES, "trtr"), exist_ok=True)
    p = os.path.join(RES, "trtr", f"{dataset}_split{split}.json")
    if not os.path.exists(p):
        dump(dict(dataset=dataset, split=split, **trtr(train, test, meta, enc_x, seed=split)), p)


def run(dataset, split, seed, gen_name, attacks=True):
    out_dir = os.path.join(RES, "runs", dataset, gen_name, f"split{split}_seed{seed}")
    if os.path.exists(os.path.join(out_dir, "DONE")):
        print("skip", out_dir, flush=True)
        return
    os.makedirs(out_dir, exist_ok=True)
    train, val, test, meta = load_split(dataset, split)
    enc = Encoder(meta).fit(train)
    enc_x = Encoder(meta, include_target=False).fit(train)
    ensure_trtr(dataset, split, train, test, meta, enc_x)
    eval_seed = 100 * split + seed
    n_out = len(train)

    gen = make(gen_name)
    t0 = time.time()
    gen.fit(train, meta, seed=seed)
    fit_s = time.time() - t0
    if SAVE_MODELS:
        # final fitted model kept for release to reviewers; non-fatal on failure
        try:
            gen.save(os.path.join(RES, "models", dataset, gen_name, f"split{split}_seed{seed}"))
        except NotImplementedError:
            pass
        except Exception as e:  # noqa: BLE001
            print(f"model save failed: {type(e).__name__}: {e}", flush=True)
    base = dict(dataset=dataset, split=split, seed=seed, generator=gen_name,
                dp_epsilon=gen.dp_epsilon, n_train=len(train), n_val=len(val), n_test=len(test))

    t0 = time.time()
    if gen.dp_epsilon is not None:
        pools = [(gen.configs()[0], gen.sample(n_out, gen.configs()[0]))]
    else:
        pools = [(c, gen.sample(int(POOL_FACTOR * n_out), c)) for c in gen.configs()]
    sample_s = time.time() - t0
    sample_info = getattr(gen, "sample_info", None)
    dump(dict(base, fit_seconds=round(fit_s, 1), sample_seconds=round(sample_s, 1),
              pool_sizes=[len(p) for _, p in pools], fit_info=getattr(gen, "fit_info", {}),
              sample_info=sample_info),
         os.path.join(out_dir, "fit.json"))

    trtr_cache = {}  # real-train subgroup predictions, shared by all releases of this run

    def release(method, syn, extra, reuse=None):
        res = dict(base, method=method, **extra)
        # an identical released dataset has identical metrics: reuse them
        ev = reuse if reuse is not None else full_eval(syn, train, val, test, meta, enc, enc_x,
                                                       seed=eval_seed, attacks=attacks, trtr_cache=trtr_cache)
        res.update(ev)
        dump(res, os.path.join(out_dir, f"{method}.json"))
        syn.to_csv(os.path.join(out_dir, f"{method}.csv.gz"), index=False)
        return ev

    if gen.dp_epsilon is not None:
        release(gen_name, pools[0][1].iloc[:n_out].reset_index(drop=True), {"family": "dp"})
    else:
        val_ctx = core.RealContext(train, val, meta, enc, enc_x, seed=eval_seed)
        test_ctx = core.RealContext(train, test, meta, enc, enc_x, seed=eval_seed)
        tab, subsets = core.candidate_table(pools, n_out, val_ctx, test_ctx, seed=eval_seed)
        tab.to_csv(os.path.join(out_dir, "candidates.csv"), index=False)
        # pools are kept so any future selector can be evaluated without refitting
        for k, (_, pool) in enumerate(pools):
            pool.to_csv(os.path.join(out_dir, f"pool{k}.csv.gz"), index=False)
        default_idx = next(i for i, (c, _) in enumerate(pools) if c.get("temperature", 1.0) == 1.0)

        i_def = core.select_default(tab, default_idx)
        release(gen_name, subsets[i_def], {"family": "base", "cand": i_def})

        syn_f, info_f = core.filter_baseline(pools[default_idx][1], n_out, val_ctx, seed=eval_seed)
        release(f"{gen_name}+filter", syn_f, dict(family="filter", **info_f))

        i_alm, feas, trace = core.select_alm(tab, P_MIN, return_trace=True)
        dump(trace, os.path.join(out_dir, "alm_trace.json"))

        def sel_info(i, feasible, family):
            r = tab.iloc[i]
            return dict(family=family, cand=i, p_min=P_MIN, feasible_val=feasible, config=r["config"],
                        tau_q=float(r["tau_q"]), accept_rate=float(r["accept_rate"]),
                        P_val=float(core.privacy(r)), U_val=float(core.utility(r)))

        ev_alm = release(f"{gen_name}+psygentab", subsets[i_alm], sel_info(i_alm, feas, "psygentab"))
        i_min, feas_min = core.select_min_intervention(tab, P_MIN)
        release(f"{gen_name}+psygentab_min", subsets[i_min], sel_info(i_min, feas_min, "psygentab_min"),
                reuse=ev_alm if i_min == i_alm else None)
    open(os.path.join(out_dir, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"done {dataset} {gen_name} split{split} seed{seed} fit={fit_s:.0f}s sample={sample_s:.0f}s "
          f"total={time.time() - t0 + fit_s:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--no_attacks", action="store_true")
    a = ap.parse_args()
    try:
        run(a.dataset, a.split, a.seed, a.generator, attacks=not a.no_attacks)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
