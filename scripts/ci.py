"""Confidence intervals used throughout the manuscript (all 95%).

* Means over datasets: two-level (hierarchical) percentile bootstrap that
  resamples datasets and, within each sampled dataset, its runs, so the interval
  reflects both between-dataset heterogeneity and run-to-run variability.
* Paired effects (PSyGenTAB minus raw generator): the same bootstrap applied to
  run-level paired differences (pairs share dataset, split and seed).
* Per-dataset means over runs: Student-t interval.
* Proportions (constraint satisfaction): Wilson score interval.
"""
import numpy as np
import pandas as pd
from scipy import stats

B = 5000
SEED = 20261116


def _groups(df, value_col, group_col="dataset"):
    return [g[value_col].dropna().to_numpy(float) for _, g in df.groupby(group_col)]


def hier_boot_ci(groups, B=B, seed=SEED, stat=np.mean):
    """Percentile CI of the mean of per-group means; groups: list of 1-D arrays."""
    groups = [g for g in groups if len(g)]
    if not groups:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    point = float(np.mean([stat(g) for g in groups]))
    # per-group bootstrap distribution of the within-group mean (B values each)
    within = np.stack([g[rng.integers(0, len(g), size=(B, len(g)))].mean(1) if len(g) > 1
                       else np.repeat(g.mean(), B) for g in groups])        # (G, B)
    pick = rng.integers(0, len(groups), size=(B, len(groups)))              # (B, G)
    boots = within[pick, np.arange(B)[:, None]].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def mean_ci(df, method, metric, transform=None):
    sub = df[df.method == method]
    if transform is not None:
        sub = sub.assign(**{metric: transform(sub[metric])})
    return hier_boot_ci(_groups(sub, metric))


def paired_diff_ci(df, ref, other, metric, direction):
    """CI of the paired improvement of `ref` over `other` in the metric's ideal
    direction (+1 higher, -1 lower, 0 closer to 0.5, 2 closer to 1)."""
    key = ["dataset", "split", "seed"]
    a = df[df.method == ref].set_index(key)[metric]
    b = df[df.method == other].set_index(key)[metric]
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if direction == 0:
        d = (j.b - 0.5).abs() - (j.a - 0.5).abs()
    elif direction == 2:
        d = (j.b - 1).abs() - (j.a - 1).abs()
    else:
        d = direction * (j.a - j.b)
    return hier_boot_ci(_groups(d.rename("d").reset_index(), "d"))


def t_ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float(np.mean(x)) if len(x) else np.nan, np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def wilson(k, n, z=1.959964):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def pm(point, lo, hi, d=3, small=True):
    """LaTeX 'mean ± half-width' with the half-width typeset small."""
    if np.isnan(point):
        return "--"
    if np.isnan(lo):
        return f"{point:.{d}f}"
    h = (hi - lo) / 2
    hs = f"{h:.{d}f}".lstrip("0") if h < 1 else f"{h:.{d}f}"
    return f"{point:.{d}f}" + (f"{{\\scriptsize$\\pm${hs}}}" if small else f"$\\pm${hs}")


def bracket(point, lo, hi, d=3, sign=False):
    if np.isnan(point):
        return "--"
    f = f"{{:+.{d}f}}" if sign else f"{{:.{d}f}}"
    return f.format(point) + f"\\,{{\\scriptsize[{f.format(lo)}, {f.format(hi)}]}}"
