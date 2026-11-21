"""Job scheduler for the full PSyGenTAB study (resumable).

Tiers (agreed protocol):
  core : rtf, ctabgan_plus, DP-CTAB-GAN+ (weight clipping "dp" and gradient
         penalty "dpgp") eps{1,3,10}                    -> 5 splits x 3 seeds
  zoo  : DISCO-TAB generators + PATE-GAN eps{1,3,10}      -> 5 splits x 1 seed

Lanes: "gpu" jobs (MPS) and "cpu" jobs run in separate worker pools. Jobs are
ordered seed-major then split-major, so partial results already cover every
dataset. Each job is a fresh subprocess; completed runs are skipped.

  python -m scripts.run_all --lane gpu --workers 2
  python -m scripts.run_all --lane cpu --workers 3
"""
import argparse
import itertools
import os
import threading
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from psygentab.datasets import ALL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs", "jobs")
SPLITS = range(5)

CORE = ["rtf", "ctabgan_plus"] + [f"ctabgan_plus_{v}_eps{e}" for v in ("dp", "dpgp") for e in (1, 3, 10)]
ZOO = ["bayesian_network", "arf", "tvae", "ctgan", "tabddpm", "pategan_eps1", "pategan_eps3", "pategan_eps10"]
GPU = set(CORE)


def jobs(lane, datasets, generators=None):
    out = []
    for seed in range(3):
        for split in SPLITS:
            for ds in datasets:
                for g in CORE + ZOO:
                    if generators and g not in generators:
                        continue
                    if (g in GPU) != (lane == "gpu"):
                        continue
                    if g in ZOO and seed > 0:
                        continue
                    out.append((ds, split, seed, g))
    return out


def done(ds, split, seed, g):
    return os.path.exists(os.path.join(ROOT, "results", "runs", ds, g, f"split{split}_seed{seed}", "DONE"))


_gpu_lock = threading.Lock()
_gpu_next = itertools.count()


def run_job(job, threads, timeout, gpus=None):
    ds, split, seed, g = job
    if done(*job):
        return job, "skip", 0
    log = os.path.join(LOGS, f"{ds}__{g}__split{split}_seed{seed}.log")
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), PSY_THREADS=str(threads),
               OMP_WAIT_POLICY="PASSIVE",
               PYTHONWARNINGS="ignore", TOKENIZERS_PARALLELISM="false")
    if gpus:
        with _gpu_lock:
            env["CUDA_VISIBLE_DEVICES"] = gpus[next(_gpu_next) % len(gpus)]
    cmd = [sys.executable, "-m", "scripts.run_one", "--dataset", ds, "--split", str(split),
           "--seed", str(seed), "--generator", g]
    t0 = time.time()
    status = "fail"
    for attempt in range(2):
        with open(log, "a") as f:
            f.write(f"\n=== attempt {attempt + 1} {time.ctime()} ===\n")
            f.flush()
            try:
                rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT,
                                    timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        if rc == 0 and done(*job):
            status = "ok"
            break
        status = f"fail({rc})"
    return job, status, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", choices=["gpu", "cpu"], required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=4 * 3600)
    ap.add_argument("--datasets", nargs="+", default=ALL)
    ap.add_argument("--generators", nargs="+", default=None)
    ap.add_argument("--gpus", default=None, help="comma-separated CUDA device ids, assigned round-robin")
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    todo = [j for j in jobs(a.lane, a.datasets, a.generators) if not done(*j)]
    print(f"[{time.ctime()}] lane={a.lane} pending={len(todo)}", flush=True)
    status_log = os.path.join(ROOT, "logs", f"status_{a.lane}.log")
    with ThreadPoolExecutor(a.workers) as ex:
        gpus = a.gpus.split(",") if a.gpus else None
        for k, (job, status, dt) in enumerate(ex.map(lambda j: run_job(j, a.threads, a.timeout, gpus), todo)):
            line = f"[{time.ctime()}] {k + 1}/{len(todo)} {status:10s} {dt / 60:6.1f} min  {job}"
            print(line, flush=True)
            with open(status_log, "a") as f:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
