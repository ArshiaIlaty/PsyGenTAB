"""DP-SGD training loop for the official CTAB-GAN+ synthesizer.

Mirrors `CTABGANSynthesizer.fit` (third_party/CTAB-GAN-Plus) with these
changes, required for a valid (epsilon, delta) guarantee on the critic:
  * real rows are Poisson-subsampled with rate q = batch / n, and each real
    row's conditional vector is derived from the row itself;
  * the real-data part of the critic loss gets per-sample gradients
    (torch.func), clipped to norm C, summed and perturbed with N(0, sigma^2 C^2);
  * the critic's Lipschitz constraint is either
      lipschitz="clip": weight clipping, as in DPGAN (Xie et al., 2018), or
      lipschitz="gp":   CTAB-GAN+'s slerp gradient penalty, computed per record
                        (real row i paired with fake row i) inside the clipped
                        per-sample gradient; needs a double backward per record;
  * the auxiliary classifier and the information loss are disabled;
  * sigma is calibrated with the repo's RDP accountant for the total number
    of critic steps.
Must be called with third_party/CTAB-GAN-Plus on sys.path.
"""
import functools
import json
import os

import numpy as np
import torch
from torch.func import functional_call, grad, vmap
from torch.optim import Adam

from model.privacy_utils.rdp_accountant import compute_rdp, get_privacy_spent
from model.synthesizer.ctabgan_synthesizer import (
    Cond, Discriminator, Generator, apply_activate, cond_loss, determine_layers_disc,
    determine_layers_gen, slerp, weights_init)
from model.synthesizer.transformer import DataTransformer, ImageTransformer

ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)) + [128, 256]


def epsilon_for(q, sigma, steps, delta):
    rdp = compute_rdp(q, sigma, steps, ORDERS)
    return get_privacy_spent(ORDERS, rdp, target_delta=delta)[0]


SIGMA_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "dp_sigma_cache.json")


def calibrate_sigma(q, steps, eps, delta):
    """Smallest noise multiplier reaching (eps, delta); cached on disk because
    (q, steps, eps, delta) repeat across the splits and seeds of a dataset."""
    key = f"{q:.8f}|{steps}|{eps:g}|{delta:.6e}"
    try:
        cache = json.load(open(SIGMA_CACHE))
    except (OSError, ValueError):
        cache = {}
    if key in cache:
        return cache[key]
    sigma = _bisect_sigma(q, steps, eps, delta)
    try:
        cache = json.load(open(SIGMA_CACHE))
    except (OSError, ValueError):
        cache = {}
    cache[key] = sigma
    tmp = f"{SIGMA_CACHE}.{os.getpid()}.tmp"
    os.makedirs(os.path.dirname(SIGMA_CACHE), exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=1)
    os.replace(tmp, SIGMA_CACHE)
    return sigma


def _bisect_sigma(q, steps, eps, delta):
    lo, hi = 0.3, 200.0
    if epsilon_for(q, hi, steps, delta) > eps:
        raise ValueError("target epsilon unreachable")
    for _ in range(60):
        mid = (lo + hi) / 2
        if epsilon_for(q, mid, steps, delta) > eps:
            lo = mid
        else:
            hi = mid
    return hi


def _row_condvec(cond, rows):
    """Conditional vectors built from the real rows' own categories."""
    B = rows.shape[0]
    idx = np.random.choice(np.arange(cond.n_col), B)
    vec = np.zeros((B, cond.n_opt), dtype="float32")
    for i in range(B):
        opt = cond.model[idx[i]][rows[i]]
        vec[i, cond.interval[idx[i], 0] + opt] = 1
    return vec


def fit_dp(self, train_df, ct, target_eps, delta, C=1.0, ci=5, micro=None, clip_w=0.01,
           lipschitz="clip", gp_lambda=10.0):
    if lipschitz not in ("clip", "gp"):
        raise ValueError(lipschitz)
    micro = micro or (64 if lipschitz == "gp" else 256)
    dev = self.device
    import model.synthesizer.ctabgan_synthesizer as cs
    DT = cs.DataTransformer if isinstance(cs.DataTransformer, functools.partial) else DataTransformer
    self.transformer = DT(train_data=train_df, categorical_list=ct["categorical"],
                                       mixed_dict=ct["mixed"], general_list=ct["general"],
                                       non_categorical_list=ct["non_categorical"])
    self.transformer.fit()
    data = self.transformer.transform(train_df.values)
    n = len(data)
    data_dim = self.transformer.output_dim
    self.cond_generator = Cond(data, self.transformer.output_info)
    for s in [4, 8, 16, 24, 32, 64]:
        if s * s >= data_dim + self.cond_generator.n_opt:
            self.dside = s
            break
    for s in [4, 8, 16, 24, 32, 64]:
        if s * s >= data_dim:
            self.gside = s
            break
    self.generator = Generator(self.gside, determine_layers_gen(
        self.gside, self.random_dim + self.cond_generator.n_opt, self.num_channels)).to(dev)
    D = Discriminator(self.dside, determine_layers_disc(self.dside, self.num_channels)).to(dev)
    # in-place activations are not vmap-safe
    for m in D.modules():
        if hasattr(m, "inplace"):
            m.inplace = False
    opt_kw = dict(lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
    optG, optD = Adam(self.generator.parameters(), **opt_kw), Adam(D.parameters(), **opt_kw)
    self.generator.apply(weights_init)
    D.apply(weights_init)
    self.Gtransformer, self.Dtransformer = ImageTransformer(self.gside), ImageTransformer(self.dside)
    Dt = self.Dtransformer

    B = self.batch_size
    q = B / n
    steps_per_epoch = max(1, n // B)
    total_d_steps = self.epochs * steps_per_epoch * ci
    sigma = calibrate_sigma(q, total_d_steps, target_eps, delta)

    def gen_fake(bsz):
        z = torch.randn(bsz, self.random_dim, device=dev)
        c, m, col, opt = self.cond_generator.sample_train(bsz)
        c = torch.from_numpy(c).to(dev)
        z = torch.cat([z, c], dim=1).view(bsz, self.random_dim + self.cond_generator.n_opt, 1, 1)
        faket = self.Gtransformer.inverse_transform(self.generator(z))
        fake = apply_activate(faket, self.transformer.output_info)
        return fake, c, torch.from_numpy(m).to(dev), faket

    names = [k for k, _ in D.named_parameters()]

    def real_loss(params, x_real):
        d_real = functional_call(D, params, (Dt.transform(x_real.unsqueeze(0)),))[0]
        return -d_real.mean()

    def real_loss_gp(params, x_real, x_fake, a):
        xr, xf = x_real.unsqueeze(0), x_fake.unsqueeze(0)
        d_real = functional_call(D, params, (Dt.transform(xr),))[0]
        inter = slerp(a.view(1, 1), xr, xf)

        def d_sum(x):
            return functional_call(D, params, (Dt.transform(x),))[0].sum()
        gnorm = grad(d_sum)(inter).view(1, -1).norm(2, dim=1)
        return -d_real.mean() + gp_lambda * ((gnorm - 1) ** 2).mean()

    if lipschitz == "gp":
        per_sample = vmap(grad(real_loss_gp), in_dims=(None, 0, 0, 0))
    else:
        per_sample = vmap(grad(real_loss), in_dims=(None, 0))
    cat_rows = np.stack(self.cond_generator.model, axis=1) if self.cond_generator.n_col else None

    for _ in range(self.epochs):
        for _ in range(steps_per_epoch):
            for _ in range(ci):
                mask = np.random.rand(n) < q
                idx = np.nonzero(mask)[0]
                if len(idx) == 0:
                    continue
                Bt = len(idx)
                real = torch.from_numpy(data[idx].astype("float32")).to(dev)
                c_real = torch.from_numpy(_row_condvec(self.cond_generator, cat_rows[idx])).to(dev)
                with torch.no_grad():
                    fake, c_fake, _, _ = gen_fake(Bt)
                real_cat = torch.cat([real, c_real], dim=1)
                fake_cat = torch.cat([fake, c_fake], dim=1)
                params = {k: v.detach() for k, v in D.named_parameters()}
                alpha = torch.rand(Bt, device=dev)
                # clipped per-sample gradient sums, in micro-batches to bound memory
                gsum = {k: torch.zeros_like(v) for k, v in params.items()}
                for s0 in range(0, Bt, micro):
                    sl = slice(s0, s0 + micro)
                    g = (per_sample(params, real_cat[sl], fake_cat[sl], alpha[sl]) if lipschitz == "gp"
                         else per_sample(params, real_cat[sl]))
                    b = g[names[0]].shape[0]
                    flat = torch.cat([g[k].reshape(b, -1) for k in names], dim=1)
                    factor = (C / (flat.norm(2, dim=1) + 1e-12)).clamp(max=1.0)
                    for k in names:
                        gsum[k] += (g[k] * factor.view(-1, *([1] * (g[k].dim() - 1)))).sum(0)
                    del g, flat
                optD.zero_grad()
                # non-private term: the critic score on generated rows
                d_fake = D(Dt.transform(fake_cat))[0].mean()
                d_fake.backward()
                for k, p in D.named_parameters():
                    gk = (gsum[k] + sigma * C * torch.randn_like(gsum[k])) / (q * n)
                    p.grad = p.grad + gk if p.grad is not None else gk
                optD.step()
                if lipschitz == "clip":
                    with torch.no_grad():
                        for p in D.parameters():
                            p.clamp_(-clip_w, clip_w)
            # generator step (post-processing of the DP critic)
            optG.zero_grad()
            fake, c, m, faket = gen_fake(B)
            y_fake, _ = D(Dt.transform(torch.cat([fake, c], dim=1)))
            # conditional loss uses only the generator's own logits
            loss_g = -torch.mean(y_fake) + cond_loss(faket, self.transformer.output_info, c, m)
            loss_g.backward()
            optG.step()
    eps_spent = epsilon_for(q, sigma, total_d_steps, delta)
    return {"dp_epsilon": float(eps_spent), "dp_delta": float(delta), "noise_multiplier": float(sigma),
            "sample_rate": float(q), "critic_steps": int(total_d_steps), "max_grad_norm": C,
            "lipschitz": lipschitz, "weight_clip": clip_w if lipschitz == "clip" else None,
            "gp_lambda": gp_lambda if lipschitz == "gp" else None}
