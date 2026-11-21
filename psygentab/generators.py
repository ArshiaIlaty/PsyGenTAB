"""Uniform generator interface: fit(train, meta, seed); sample(n, config).

`configs()` lists the sampling-time configurations PSyGenTAB may choose from.
REaLTabFormer exposes temperature; GAN/VAE/diffusion models expose a single
default configuration, so for them PSyGenTAB acts through record acceptance.
"""
import contextlib
import os
import shutil
import random
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from .datasets import harmonise

HERE = os.path.dirname(os.path.abspath(__file__))
CTAB_DIR = os.path.join(os.path.dirname(HERE), "third_party", "CTAB-GAN-Plus")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)


def epoch_schedule(n):
    """Epochs by training-set size (same schedule as DISCO-TAB)."""
    if n <= 500:
        return 300
    if n <= 2000:
        return 200
    return 150


def finalize(syn, meta):
    """Align a raw synthetic frame to the real schema and class set."""
    syn = syn.copy()
    for c in meta["columns"]:
        if c not in syn.columns:
            syn[c] = np.nan
    syn = harmonise(syn, meta)
    classes = [str(c) for c in meta["classes"]]
    t = meta["target"]
    bad = ~syn[t].isin(classes)
    if bad.any():
        # numeric-looking labels such as "1.0" -> nearest class
        def fix(v):
            try:
                fv = float(v)
                num = [float(c) for c in classes]
                return classes[int(np.argmin([abs(fv - x) for x in num]))]
            except ValueError:
                return None
        syn.loc[bad, t] = syn.loc[bad, t].map(fix)
        syn = syn.dropna(subset=[t])
    return syn.reset_index(drop=True)


class Generator:
    name = "base"
    dp_epsilon = None

    def configs(self):
        return [{}]

    def fit(self, train, meta, seed):
        raise NotImplementedError

    def sample(self, n, config=None):
        raise NotImplementedError

    def save(self, path):
        """Persist the fitted model (final weights only) for release."""
        raise NotImplementedError


# ---------------------------------------------------------------- REaLTabFormer

# REaLTabFormer truncates decimal columns to `numeric_max_len` characters and
# requires it to exceed the integer width; computed on each full dataset. All
# other datasets fit the library default of 10.
RTF_NUMERIC_MAX_LEN = {"pir_vision": 11, "vn_banking": 11}
RTF_TMP = os.environ.get("PSY_TMP", os.path.join(os.path.dirname(HERE), "tmp_rtf"))


class RTFGen(Generator):
    name = "rtf"

    def __init__(self, epochs=None, batch_size=64, device=None, temperatures=(0.8, 1.0, 1.2)):
        self.epochs, self.batch_size = epochs, batch_size
        self.temperatures = temperatures
        import torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else
                                 "cuda" if torch.cuda.is_available() else "cpu")

    def configs(self):
        return [{"temperature": t} for t in self.temperatures]

    def fit(self, train, meta, seed):
        import torch
        from realtabformer import REaLTabFormer
        from transformers.utils import logging as tl
        tl.set_verbosity_error()
        torch.set_num_threads(int(os.environ.get("PSY_THREADS", 4)))
        seed_everything(seed)
        self.meta = meta
        # checkpoints go to disk (not RAM-backed /tmp) and are deleted after fitting:
        # the library saves ~3 GB of checkpoints per run even with save_strategy="no"
        os.makedirs(RTF_TMP, exist_ok=True)
        self.tmp = tempfile.mkdtemp(prefix="rtf_", dir=RTF_TMP)
        n = len(train)
        ep = self.epochs or (100 if n <= 500 else 60 if n <= 1000 else 40)
        self.model = REaLTabFormer(model_type="tabular", epochs=ep, batch_size=self.batch_size,
                                   random_state=seed, train_size=1.0,
                                   checkpoints_dir=os.path.join(self.tmp, "ck"),
                                   samples_save_dir=os.path.join(self.tmp, "s"),
                                   numeric_max_len=RTF_NUMERIC_MAX_LEN.get(meta["dataset"], 10),
                                   logging_steps=10 ** 9, save_strategy="no")
        try:
            self.model.fit(train[meta["columns"]], n_critic=0, device=self.device)
        finally:
            shutil.rmtree(self.tmp, ignore_errors=True)
        self.fit_info = {"epochs": ep}
        return self

    def save(self, path):
        """REaLTabFormer.save JSON-dumps its attributes; it converts its two
        directory Paths itself but not Paths elsewhere, so stringify those
        temporarily."""
        from pathlib import Path
        os.makedirs(path, exist_ok=True)
        keep = ("checkpoints_dir", "samples_save_dir")

        def strip(v):
            if isinstance(v, Path):
                return v.as_posix()
            if isinstance(v, dict):
                return {k: strip(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return type(v)(strip(x) for x in v)
            return v
        attrs = self.model.__dict__
        saved = {k: v for k, v in attrs.items() if k not in keep and strip(v) is not v}
        try:
            for k, v in saved.items():
                attrs[k] = strip(v)
            self.model.save(path, allow_overwrite=True)
        finally:
            attrs.update(saved)

    def sample(self, n, config=None):
        # on Apple MPS, autoregressive decoding is ~5x faster on CPU; on CUDA use the GPU
        config = config or {}
        dev = "cpu" if self.device == "mps" else self.device
        self.model.model.to(dev)
        # generation batch only bounds memory (rows decoded at once), not the samples;
        # long rows (e.g. 31-column Breast Cancer) need a smaller one on a shared 16 GB GPU
        gen_batch = 256 if dev.startswith("cuda") else 1024
        out = self.model.sample(n_samples=n, gen_batch=gen_batch, device=dev,
                                temperature=config.get("temperature", 1.0))
        return finalize(out, self.meta)


# ---------------------------------------------------------------- CTAB-GAN+

@contextlib.contextmanager
def _ctab_path():
    sys.path.insert(0, CTAB_DIR)
    try:
        yield
    finally:
        sys.path.remove(CTAB_DIR)


def ctab_column_spec(train, meta):
    cats = meta["categorical"] + [meta["target"]]
    mixed, integer = {}, []
    for c in meta["numerical"]:
        x = train[c].dropna()
        if len(x) and x.value_counts(normalize=True).iloc[0] >= 0.3:
            mixed[c] = [float(x.value_counts().index[0])]
        if len(x) and np.all(np.isclose(x, np.round(x))):
            integer.append(c)
    return cats, mixed, integer


class CTABGANPlusGen(Generator):
    """Official CTAB-GAN+ (Zhao et al., 2024), optionally with DP-SGD.

    DP mode (dp_epsilon set): the critic is trained with DP-SGD on Poisson-
    subsampled real rows (per-sample clipping of the real-data loss incl. the
    gradient penalty, Gaussian noise, RDP accounting). The auxiliary
    classifier and the information loss, which read real data outside the
    critic, are disabled. As in DP-CTGAN, the mode-specific normalisation and
    category frequencies are fitted non-privately.
    """

    def __init__(self, epochs=None, batch_size=500, dp_epsilon=None, dp_delta=None,
                 max_grad_norm=1.0, lipschitz="clip"):
        self.epochs, self.batch_size = epochs, batch_size
        self.dp_epsilon, self.dp_delta, self.C = dp_epsilon, dp_delta, max_grad_norm
        self.lipschitz = lipschitz
        tag = "dp" if lipschitz == "clip" else "dpgp"
        self.name = "ctabgan_plus" if dp_epsilon is None else f"ctabgan_plus_{tag}_eps{dp_epsilon:g}"

    def fit(self, train, meta, seed):
        import torch
        torch.set_num_threads(int(os.environ.get("PSY_THREADS", 4)))
        seed_everything(seed)
        self.meta = meta
        cats, mixed, integer = ctab_column_spec(train, meta)
        with _ctab_path():
            from model.pipeline import data_preparation as dp_mod
            from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer
            # Our split contract replaces the internal 80/20 split.
            orig = dp_mod.model_selection.train_test_split
            dp_mod.model_selection.train_test_split = (
                lambda X, y, **kw: (X.copy(), X.iloc[:0], y.copy(), y.iloc[:0]))
            try:
                self.prep = dp_mod.DataPrep(train[meta["columns"]].copy(), cats, [], dict(mixed), [], [],
                                            integer, {"Classification": meta["target"]}, 0.2)
            finally:
                dp_mod.model_selection.train_test_split = orig
            ep = self.epochs or epoch_schedule(len(train))
            bs = min(self.batch_size, max(50, (len(train) // 2) // 10 * 10))
            self.synth = CTABGANSynthesizer(epochs=ep, batch_size=bs)
            # CPU convolutions in this torch build are ~25x slower than MPS
            if torch.backends.mps.is_available():
                self.synth.device = torch.device("mps")
            ct = self.prep.column_types
            # CTAB-GAN+ fits a 10-component mixture per numerical column; tiny training
            # sets (e.g. 150-record shadow worlds) can have fewer usable values than that
            nc = self._n_clusters(self.prep.df, ct)
            import model.synthesizer.ctabgan_synthesizer as cs
            orig_dt = cs.DataTransformer
            if nc < 10:
                import functools
                cs.DataTransformer = functools.partial(orig_dt, n_clusters=nc)
                self.fit_info_extra = {"n_clusters": nc}
            try:
                self._fit_inner(meta, ct, ep, bs, train)
            finally:
                cs.DataTransformer = orig_dt
        return self

    def _fit_inner(self, meta, ct, ep, bs, train):
        with _ctab_path():
            if self.dp_epsilon is None:
                self.synth.fit(train_data=self.prep.df, categorical=ct["categorical"], mixed=ct["mixed"],
                               general=ct["general"], non_categorical=ct["non_categorical"],
                               type={"Classification": meta["target"]})
                self.fit_info = {"epochs": ep, "batch_size": bs}
            else:
                from .ctabgan_dp import fit_dp
                delta = self.dp_delta or 1.0 / (len(train) ** 1.1)
                self.fit_info = fit_dp(self.synth, self.prep.df, ct, self.dp_epsilon, delta, self.C,
                                       lipschitz=self.lipschitz)
                self.fit_info.update(epochs=ep, batch_size=bs)
        return self

    @staticmethod
    def _n_clusters(df, ct, default=10):
        """Largest mixture size every continuous/mixed column can support."""
        cats = set(ct["categorical"])
        nc = default
        for j, c in enumerate(df.columns):
            if j in cats:
                continue
            x = pd.to_numeric(df[c], errors="coerce").dropna()
            modes = ct["mixed"].get(j, [])
            n = int((~x.isin(modes)).sum()) if modes else int(x.nunique())
            nc = min(nc, max(1, n))
        return nc

    def save(self, path):
        import torch
        os.makedirs(path, exist_ok=True)
        with _ctab_path():
            torch.save({"synthesizer": self.synth, "data_prep": self.prep, "meta": self.meta,
                        "fit_info": self.fit_info}, os.path.join(path, "ctabgan_plus.pt"))

    def sample(self, n, config=None):
        with _ctab_path():
            raw = self._sample_bounded(n)
            out = self.prep.inverse_prep(raw)
        return finalize(out, self.meta)

    def _sample_bounded(self, n, max_rounds=50):
        """CTAB-GAN+ rejection sampling (rows with any numeric value outside the
        training range are discarded), capped at `max_rounds` extra batches.
        Heavily noised (DP) generators on all-numeric data can produce almost no
        valid rows, which makes the original loop run forever; any shortfall is
        filled with generated rows whose numeric values are clipped to the
        training range. `self.sample_info` records the valid-row rate."""
        import torch
        from model.synthesizer.ctabgan_synthesizer import apply_activate
        sy = self.synth
        sy.generator.eval()

        def batch():
            z = torch.randn(sy.batch_size, sy.random_dim, device=sy.device)
            c = torch.from_numpy(sy.cond_generator.sample(sy.batch_size)).to(sy.device)
            z = torch.cat([z, c], dim=1).view(sy.batch_size, sy.random_dim + sy.cond_generator.n_opt, 1, 1)
            with torch.no_grad():
                fake = apply_activate(sy.Gtransformer.inverse_transform(sy.generator(z)),
                                      sy.transformer.output_info)
            return fake.cpu().numpy()

        valid, generated, rounds = [], 0, 0
        need_rounds = n // sy.batch_size + 1
        while sum(len(v) for v in valid) < n and rounds < need_rounds + max_rounds:
            raw = batch()
            res, _ = sy.transformer.inverse_transform(raw)
            valid.append(res)
            generated += len(raw)
            rounds += 1
        out = np.concatenate(valid, axis=0) if valid else np.zeros((0, len(sy.transformer.meta)))
        n_valid_total = len(out)
        n_clipped = 0
        if len(out) < n:
            # widen the validity bounds, invert, then clip to the true bounds
            meta = sy.transformer.meta
            bounds = [(m.get("min"), m.get("max")) for m in meta]
            for m in meta:
                if "min" in m:
                    m["min"], m["max"] = -np.inf, np.inf
            try:
                extra = []
                while sum(len(e) for e in extra) < n - len(out):
                    extra.append(sy.transformer.inverse_transform(batch())[0])
            finally:
                for m, (lo, hi) in zip(meta, bounds):
                    if lo is not None:
                        m["min"], m["max"] = lo, hi
            extra = np.concatenate(extra, axis=0)[: n - len(out)].astype(float)
            for j, (lo, hi) in enumerate(bounds):
                if lo is not None:
                    extra[:, j] = np.clip(extra[:, j], lo, hi)
            n_clipped = len(extra)
            out = np.concatenate([out, extra], axis=0)
        self.sample_info = {"valid_rate": n_valid_total / max(1, generated), "n_clipped": int(n_clipped),
                            "n_requested": int(n)}
        return out[:n]


# ---------------------------------------------------------------- synthcity zoo

DDPM_STEPS, DDPM_BATCH = 20000, 256

SYNTHCITY = {
    "ctgan": "ctgan", "tvae": "tvae", "tabddpm": "ddpm", "arf": "arf",
    "bayesian_network": "bayesian_network", "adsgan": "adsgan",
    "pategan": "pategan", "dpgan": "dpgan", "aim": "aim",
}


class SynthcityGen(Generator):
    def __init__(self, label, dp_epsilon=None, n_iter=None):
        self.label, self.plugin = label, SYNTHCITY[label]
        self.dp_epsilon, self.n_iter = dp_epsilon, n_iter
        self.name = label if dp_epsilon is None else f"{label}_eps{dp_epsilon:g}"

    def fit(self, train, meta, seed):
        from synthcity.plugins import Plugins
        from synthcity.plugins.core.dataloader import GenericDataLoader
        import torch
        torch.set_num_threads(int(os.environ.get("PSY_THREADS", 4)))
        seed_everything(seed)
        self.meta = meta
        kw = {"random_state": seed}
        if self.plugin not in ("bayesian_network", "arf", "aim"):
            kw["n_iter"] = self.n_iter or epoch_schedule(len(train))
        if self.plugin in ("ctgan", "tvae", "ddpm", "adsgan", "pategan", "dpgan"):
            # synthcity's DDPM mixes its global default device (CUDA if present) with
            # this argument, so on a CUDA machine everything must use the GPU
            kw["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.plugin == "ddpm":
            # synthcity's n_iter counts epochs; budget TabDDPM in gradient steps
            # (Kotelnikov et al., 2023 train for tens of thousands of steps)
            kw["is_classification"] = True
            kw["batch_size"] = DDPM_BATCH
            kw["n_iter"] = self.n_iter or int(np.ceil(DDPM_STEPS / np.ceil(len(train) / DDPM_BATCH)))
        if self.plugin == "pategan":
            # logistic-regression teachers as in Jordon et al. (2019); the xgboost
            # teachers segfault in this environment
            kw["epsilon"] = self.dp_epsilon
            kw["teacher_template"] = "linear"
        if self.plugin == "dpgan":
            kw["dp_epsilon"] = self.dp_epsilon
        if self.plugin == "aim":
            kw["epsilon"] = self.dp_epsilon
        df = train[meta["columns"]].copy()
        # synthcity's encoder breaks on columns that are constant in train;
        # they carry no information, so drop them and restore on sampling.
        self.constant = {c: df[c].iloc[0] for c in df.columns
                         if c != meta["target"] and df[c].nunique(dropna=False) == 1}
        df = df.drop(columns=list(self.constant))
        disc = [c for c in meta["categorical"] if c not in self.constant] + [meta["target"]]
        loader = GenericDataLoader(df, target_column=meta["target"], discrete_columns=disc)
        self.model = Plugins().get(self.plugin, **kw)
        self.model.fit(loader)
        self.fit_info = {k: v for k, v in kw.items() if k != "device"}
        return self

    def sample(self, n, config=None):
        out = self.model.generate(count=n).dataframe()
        for c, v in self.constant.items():
            out[c] = v
        return finalize(out, self.meta)


def make(name):
    """Factory from a method name such as 'ctabgan_plus_dp_eps3' or 'pategan_eps1'."""
    if name == "rtf":
        return RTFGen()
    if name == "ctabgan_plus":
        return CTABGANPlusGen()
    if name.startswith("ctabgan_plus_dp_eps"):
        return CTABGANPlusGen(dp_epsilon=float(name.split("eps")[1]), lipschitz="clip")
    if name.startswith("ctabgan_plus_dpgp_eps"):
        return CTABGANPlusGen(dp_epsilon=float(name.split("eps")[1]), lipschitz="gp")
    if "_eps" in name:
        base, eps = name.rsplit("_eps", 1)
        return SynthcityGen(base, dp_epsilon=float(eps))
    return SynthcityGen(name)
