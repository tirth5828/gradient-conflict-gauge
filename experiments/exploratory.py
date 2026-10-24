"""Exploratory pool (key: exploratory). Probes nobody asked for, each at 50 seeds where affordable.

  (a) depth 2..8 and width 16..512: mixed-cell fraction and conflict rate (is mixedness monotone in depth?)
  (b) biased trunk: exact symmetry W_l -> c_l W_l, b_l -> P_l b_l (P_l = prod_{k<=l} c_k), eta_l -> c_l^2 eta_l,
      eta_{b,l} -> P_l^2 eta_{b,l}; conjugacy check; sign census with biases (torch-default and zero bias init)
  (c) Adam: with eps = 0 the update m_hat/sqrt(v_hat) is invariant to g -> g/c, so the compensation is
      eta_l -> c_l eta_l (not c_l^2); conjugacy check at eps = 0 and the size of the break at eps = 1e-8;
      censuses under Adam (eps = 0) at two learning rates
  (d) K = 3 and 5 tasks: pairwise mixed-cell fractions, pairwise conflict, and the MGDA simplex solution
      (exact by support enumeration, Frank-Wolfe cross-checked) at c = 1 vs random per-layer gauges
  (e) softmax cross-entropy classification (random-teacher labels, 4 classes) instead of regression
  (f) LayerNorm trunk (LN after each linear map, no affine): every per-layer scale is a symmetry of the
      FUNCTION alone (no product constraint, heads untouched); conjugacy of GD still needs eta_l -> c_l^2 eta_l;
      census under LN
  (g) neuron-wise nearest single-neuron flip, by layer, on the width-64 depth-4 census
  (h) whatever turned up (see the report)

All new code lives here; the reference model functions (make_tasks, init_net, gd_run, grads,
layer_inner_products) are imported from conjugation_check.py and used unchanged for the bias-free GD
regression censuses of probe (a). The general model below (biases / LayerNorm / K heads / softmax /
Adam) reproduces the reference gradients exactly in the base case (self-test in `selftest`).

Seeds: rng = default_rng(1000 + s), s = 10..59 (disjoint from the development seeds s = 0..9), inputs / teachers /
W_0 shared across rho inside a seed (the paired design). Checkpoints 0, 99, 199, 299, 399.
Bootstrap CIs are cluster bootstraps over seeds (cells within a seed are correlated), 2000 resamples.

Run:  python exploratory.py            (all probes; ~1-1.5 h with 3 workers)
      python exploratory.py selftest   (conjugacy checks + gradient self-test only, ~1 min)
Writes exploratory.json, exploratory_cells.npz, ../figures/exploratory_*.png, and progress to exploratory.log.
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"

import itertools
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from conjugation_check import make_tasks, init_net, gd_run, grads, layer_inner_products  # noqa: E402

N_JOBS = 3
SEED0, N_SEEDS = 10, 50
RHOS = (-0.5, -0.2, 0.0, 0.3, 0.7)
CKPTS = (0, 99, 199, 299, 399)
LOG = os.path.join(HERE, "exploratory.log")


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


# ============================================================================ general model
def ln_fwd(z, eps):
    mu = z.mean(1, keepdims=True)
    zc = z - mu
    sig = np.sqrt((zc ** 2).mean(1, keepdims=True) + eps)
    return zc / sig, sig


def ln_bwd(dy, y, sig):
    return (dy - dy.mean(1, keepdims=True) - y * (dy * y).mean(1, keepdims=True)) / sig


def fwd(X, W, b, H, ln=False, ln_eps=0.0):
    hs, cache, h = [X], [], X
    for l, Wl in enumerate(W):
        z = h @ Wl
        if b is not None:
            z = z + b[l]
        if ln:
            y, sig = ln_fwd(z, ln_eps)
        else:
            y, sig = z, None
        h = np.maximum(y, 0.0)
        cache.append((y > 0, y, sig))
        hs.append(h)
    return hs, cache, [h @ Hk for Hk in H]


def loss_and_dout(out, Y, task):
    n = out.shape[0]
    if task == "reg":
        return 0.5 * np.mean((out - Y) ** 2), (out - Y) / n
    # softmax cross-entropy, Y = integer labels
    z = out - out.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    loss = -np.mean(np.log(p[np.arange(n), Y] + 0.0))
    d = p.copy(); d[np.arange(n), Y] -= 1.0
    return loss, d / n


def grads_gen(X, Ys, W, b, H, task="reg", ln=False, ln_eps=0.0):
    """Per-task gradients for K tasks. Returns losses[K], gW[k][l], gb[k][l] (None if b is None), gH[k]
    (gradient of task k w.r.t. its own head; other heads get exactly zero)."""
    hs, cache, outs = fwd(X, W, b, H, ln, ln_eps)
    L, K = len(W), len(H)
    losses, gW, gb, gH = [], [], [], []
    for k in range(K):
        loss, d = loss_and_dout(outs[k], Ys[k], task)
        gH.append(hs[-1].T @ d)
        delta = d @ H[k].T
        gWk, gbk = [None] * L, ([None] * L if b is not None else None)
        for l in range(L - 1, -1, -1):
            mask, y, sig = cache[l]
            delta = delta * mask
            if ln:
                delta = ln_bwd(delta, y, sig)
            if b is not None:
                gbk[l] = delta.sum(0)
            gWk[l] = hs[l].T @ delta
            delta = delta @ W[l].T
        losses.append(float(loss)); gW.append(gWk); gb.append(gbk)
    return losses, gW, gb, gH


def train(X, Ys, W, b, H, etaW, etab, etaH, steps, ckpts, task="reg", ln=False, ln_eps=0.0,
          opt="gd", adam_eps=1e-8, betas=(0.9, 0.999)):
    """Full-batch GD or Adam on the summed loss with per-layer step sizes (etaW[l] for W_l, etab[l] for b_l,
    etaH for every head). Returns per-step losses, parameter copies at the checkpoints (before the update
    at that step, as gd_run does) and the final parameters."""
    W = [w.copy() for w in W]; H = [h.copy() for h in H]
    b = [v.copy() for v in b] if b is not None else None
    K = len(H)
    if opt == "adam":
        mW = [np.zeros_like(w) for w in W]; vW = [np.zeros_like(w) for w in W]
        mH = [np.zeros_like(h) for h in H]; vH = [np.zeros_like(h) for h in H]
        mb = [np.zeros_like(v) for v in b] if b is not None else None
        vb = [np.zeros_like(v) for v in b] if b is not None else None
    b1, b2 = betas
    hist = {"loss": [], "ck": {}}

    def adam_step(p, g, m, v, eta, t):
        m *= b1; m += (1 - b1) * g
        v *= b2; v += (1 - b2) * g * g
        mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
        den = np.sqrt(vh) + adam_eps
        upd = np.divide(mh, den, out=np.zeros_like(mh), where=den > 0)
        p -= eta * upd

    for t in range(steps):
        losses, gW, gb, gH = grads_gen(X, Ys, W, b, H, task, ln, ln_eps)
        hist["loss"].append(losses)
        if t in ckpts:
            hist["ck"][t] = ([w.copy() for w in W], [v.copy() for v in b] if b is not None else None, [h.copy() for h in H])
        for l in range(len(W)):
            g = sum(gW[k][l] for k in range(K))
            if opt == "gd":
                W[l] -= etaW[l] * g
            else:
                adam_step(W[l], g, mW[l], vW[l], etaW[l], t + 1)
            if b is not None:
                gg = sum(gb[k][l] for k in range(K))
                if opt == "gd":
                    b[l] -= etab[l] * gg
                else:
                    adam_step(b[l], gg, mb[l], vb[l], etab[l], t + 1)
        for k in range(K):
            if opt == "gd":
                H[k] -= etaH * gH[k]
            else:
                adam_step(H[k], gH[k], mH[k], vH[k], etaH, t + 1)
    losses, _, _, _ = grads_gen(X, Ys, W, b, H, task, ln, ln_eps)
    hist["loss"].append(losses)
    hist["final"] = (W, b, H)
    return hist


# ============================================================================ tasks
def make_tasks_K(rng, n, d_in, rho, K, teacher_width=32):
    """K regression targets with (approximately) equal pairwise correlation rho >= 0:
    Y_i = sqrt(rho) f_0 + sqrt(1 - rho) f_i, f_i independent standardized random ReLU teachers."""
    X = rng.standard_normal((n, d_in))

    def teacher():
        A = rng.standard_normal((d_in, teacher_width)) * np.sqrt(2.0 / d_in)
        bb = rng.standard_normal((teacher_width, 1)) * np.sqrt(1.0 / teacher_width)
        y = np.maximum(X @ A, 0) @ bb
        return (y - y.mean()) / y.std()

    f0 = teacher()
    Ys = []
    for _ in range(K):
        fi = teacher()
        Ys.append(np.sqrt(rho) * f0 + np.sqrt(1 - rho) * fi)
    return X, Ys


def make_tasks_cls(rng, n, d_in, rho, C=4, teacher_width=32):
    """Two classification tasks: labels = argmax of random-teacher logits; the second teacher's logits are
    mixed with the first's at correlation rho before the argmax (same construction as the regression pair)."""
    X = rng.standard_normal((n, d_in))

    def teacher():
        A = rng.standard_normal((d_in, teacher_width)) * np.sqrt(2.0 / d_in)
        Bm = rng.standard_normal((teacher_width, C)) * np.sqrt(1.0 / teacher_width)
        z = np.maximum(X @ A, 0) @ Bm
        return (z - z.mean(0)) / z.std(0)

    T1, T2 = teacher(), teacher()
    Z2 = rho * T1 + np.sqrt(max(0.0, 1 - rho ** 2)) * T2
    return X, [T1.argmax(1), Z2.argmax(1)]


def init_bias(rng, W, mode):
    if mode == "zero":
        return [np.zeros(w.shape[1]) for w in W]
    # torch.nn.Linear default: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
    return [rng.uniform(-1, 1, size=w.shape[1]) / np.sqrt(w.shape[0]) for w in W]


# ============================================================================ per-cell statistics
def gram_of(gW, gb=None):
    """Per-layer Gram tensors G[i, j, l] = <g_i^l, g_j^l> for the weights (and biases if given)."""
    K, L = len(gW), len(gW[0])
    G = np.zeros((K, K, L))
    for i in range(K):
        for j in range(i, K):
            for l in range(L):
                G[i, j, l] = G[j, i, l] = np.sum(gW[i][l] * gW[j][l])
    Gb = None
    if gb is not None and gb[0] is not None:
        Gb = np.zeros((K, K, L))
        for i in range(K):
            for j in range(i, K):
                for l in range(L):
                    Gb[i, j, l] = Gb[j, i, l] = np.sum(gb[i][l] * gb[j][l])
    return G, Gb


def neuron_flips_by_layer(P, S):
    """Nearest single-neuron flip factor per hidden layer l (column of W_l = fan-in, row of W_{l+1} = fan-out):
    u = c^2 solves r u^2 + (S - b - r) u + b = 0 (split by layer)."""
    L = len(P)
    out = np.full(L - 1, np.inf)
    for l in range(L - 1):
        b = P[l].sum(0); r = P[l + 1].sum(1)
        best = np.inf
        for j in range(P[l].shape[1]):
            rest = S - b[j] - r[j]
            for u in np.roots([r[j], rest, b[j]]):
                if abs(u.imag) < 1e-12 and u.real > 0:
                    best = min(best, abs(np.log(u.real)) / 2)
        out[l] = np.exp(best)
    return out


# ============================================================================ one seed of one config
CFG_DEFAULTS = dict(depth=4, width=64, d_in=16, n=256, eta=0.05, steps=400, K=2, task="reg", bias=None,
                    ln=False, ln_eps=1e-5, opt="gd", adam_eps=0.0, rhos=RHOS, n_seeds=N_SEEDS, neuron=False,
                    C=4, use_reference=False)


def run_seed(cfg, s):
    """All rho for one seed. Returns a list of per-cell dicts."""
    c = dict(CFG_DEFAULTS); c.update(cfg)
    L, K = c["depth"], c["K"]
    rows = []
    for rho in c["rhos"]:
        rng = np.random.default_rng(1000 + s)
        if c["task"] == "cls":
            X, Ys = make_tasks_cls(rng, c["n"], c["d_in"], rho, C=c["C"])
            W, H = init_net(rng, c["d_in"], c["width"], L, d_out=c["C"])
        elif K == 2:
            X, Ys = make_tasks(rng, c["n"], c["d_in"], rho)
            W, H = init_net(rng, c["d_in"], c["width"], L)
        else:
            X, Ys = make_tasks_K(rng, c["n"], c["d_in"], rho, K)
            W, H = init_net(rng, c["d_in"], c["width"], L)
            H = H + [rng.standard_normal((c["width"], 1)) * np.sqrt(1.0 / c["width"]) for _ in range(K - 2)]
        b = init_bias(rng, W, c["bias"]) if c["bias"] else None
        etas = [c["eta"]] * L
        if c["use_reference"]:
            hist = gd_run(X, Ys, W, H, etas, c["eta"], c["steps"], record_every=1)
            ck = {k: (hist["W"][k], None, hist["H"][k]) for k in CKPTS}
            losses_all = hist["loss"]
        else:
            hist = train(X, Ys, W, b, H, etas, etas, c["eta"], c["steps"], set(CKPTS), task=c["task"], ln=c["ln"],
                         ln_eps=c["ln_eps"], opt=c["opt"], adam_eps=c["adam_eps"])
            ck = hist["ck"]; losses_all = hist["loss"]
        for k in CKPTS:
            Wk, bk, Hk = ck[k]
            if c["use_reference"]:
                ls, gW, gH = grads(X, Ys, Wk, Hk)
                gb = None
            else:
                ls, gW, gb, gH = grads_gen(X, Ys, Wk, bk, Hk, c["task"], c["ln"], c["ln_eps"])
            row = {"seed": s, "rho": rho, "step": k, "loss": np.array(ls, float),
                   "layer_fro": np.array([np.linalg.norm(w) for w in Wk])}
            if not np.all(np.isfinite(row["loss"])) or not all(np.all(np.isfinite(g)) for gk in gW for g in gk):
                row["diverged"] = True
                row["gram"] = np.full((K, K, L), np.nan); row["gram_b"] = np.full((K, K, L), np.nan) if b is not None else None
                rows.append(row); continue
            row["diverged"] = False
            G, Gb = gram_of(gW, gb)
            row["gram"], row["gram_b"] = G, Gb
            if c["neuron"] and K == 2 and b is None:
                P = [gW[0][l] * gW[1][l] for l in range(L)]
                S = float(sum(p.sum() for p in P))
                row["neuron_flip_by_layer"] = neuron_flips_by_layer(P, S)
                row["frac_neg_incoming_by_layer"] = np.array([(p.sum(0) < 0).mean() for p in P])
                row["frac_neg_outgoing_by_layer"] = np.array([(p.sum(1) < 0).mean() for p in P])
            if c["task"] == "cls" and k == 0:
                row["label_agreement"] = float((Ys[0] == Ys[1]).mean())
            rows.append(row)
    return rows


# ============================================================================ census summaries
def boot_ci(values, seeds, stat=np.mean, n_boot=2000, seed=0):
    """Cluster bootstrap over seeds of stat(values) (values: 1-d array, seeds: cluster labels)."""
    values = np.asarray(values, float); seeds = np.asarray(seeds)
    us = np.unique(seeds)
    groups = [values[seeds == u] for u in us]
    rng = np.random.default_rng(seed)
    est = float(stat(values))
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(us), len(us))
        bs.append(stat(np.concatenate([groups[i] for i in idx])))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"est": est, "lo": float(lo), "hi": float(hi)}


def random_gauge_flips(A, n_draws=2000, seed=0):
    """Fraction of cells whose sign of sum_l c_l^-2 a_l differs from c = 1 under iid c_l ~ logU[1/b, b]
    (as conjugation_check.reachability, without the product constraint = head-absorbed group)."""
    A = np.asarray(A, float); L = A.shape[1]
    sign0 = np.sign(A.sum(1))
    rng = np.random.default_rng(seed)
    out = {}
    for lo, hi, key in ((1 / np.sqrt(2), np.sqrt(2), "sqrt2"), (0.5, 2.0, "2")):
        Cm = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(n_draws, L)))
        sc = np.sign(A @ (Cm ** -2.0).T)            # cells x draws
        fl = (sc != sign0[:, None])
        out[key] = {"mean_frac": float(fl.mean()), "max_frac": float(fl.mean(0).max()),
                    "cells_ever_flipped_frac": float(fl.any(1).mean())}
    return out


def box_nearest_flip(A):
    """Closed-form nearest flip over the box max_l |log c_l| <= t WITHOUT the product constraint:
    t* = (1/4) log(sum_maj |a_l| / sum_min |a_l|); returns factors exp(t*) (inf for single-sign cells)."""
    A = np.asarray(A, float)
    s = np.sign(A.sum(1))
    maj = np.where(np.sign(A) == s[:, None], np.abs(A), 0).sum(1)
    mino = np.where(np.sign(A) == -s[:, None], np.abs(A), 0).sum(1)
    with np.errstate(divide="ignore"):
        t = 0.25 * np.log(maj / mino)
    return np.exp(t)


def summarize_two_task(rows, name):
    """Mixed fraction, conflict rate (all / post-init), with cluster-bootstrap CIs, by step and by rho,
    random-gauge flips, box nearest flip. For biased configs: several mixedness definitions."""
    ok = np.array([not r["diverged"] for r in rows])
    seeds = np.array([r["seed"] for r in rows]); steps = np.array([r["step"] for r in rows]); rhos = np.array([r["rho"] for r in rows])
    out = {"name": name, "n_cells": int(len(rows)), "n_diverged": int((~ok).sum()), "n_seeds": int(len(np.unique(seeds)))}
    if ok.sum() == 0:
        return out
    G = np.stack([r["gram"] for r in rows])               # cells x 2 x 2 x L
    A = G[:, 0, 1, :]
    has_b = rows[0].get("gram_b") is not None
    Ab = np.stack([r["gram_b"][0, 1, :] for r in rows]) if has_b else None
    S = np.nansum(A, 1) + (np.nansum(Ab, 1) if has_b else 0.0)
    N1 = G[:, 0, 0, :].sum(1) + (np.stack([r["gram_b"][0, 0, :] for r in rows]).sum(1) if has_b else 0)
    N2 = G[:, 1, 1, :].sum(1) + (np.stack([r["gram_b"][1, 1, :] for r in rows]).sum(1) if has_b else 0)
    cos = S / np.sqrt(N1 * N2)

    def mixed_of(T):
        return (np.nanmax(T, 1) > 0) & (np.nanmin(T, 1) < 0)
    mixedW = mixed_of(A)
    conflict = S < 0
    post = steps > 0
    res = {}
    for key, mask in (("all", ok), ("post_init", ok & post)):
        res[f"mixed_frac_{key}"] = boot_ci(mixedW[mask], seeds[mask])
        res[f"conflict_rate_{key}"] = boot_ci(conflict[mask], seeds[mask])
        res[f"median_abs_cos_{key}"] = boot_ci(np.abs(cos[mask]), seeds[mask], stat=np.median)
        res[f"median_cos_{key}"] = boot_ci(cos[mask], seeds[mask], stat=np.median)
    res["mixed_by_step"] = {int(k): float(mixedW[ok & (steps == k)].mean()) for k in CKPTS}
    res["conflict_by_step"] = {int(k): float(conflict[ok & (steps == k)].mean()) for k in CKPTS}
    res["mixed_by_rho"] = {str(r): float(mixedW[ok & (rhos == r)].mean()) for r in sorted(set(rhos.tolist()))}
    res["conflict_by_rho"] = {str(r): float(conflict[ok & (rhos == r)].mean()) for r in sorted(set(rhos.tolist()))}
    res["median_cos_by_rho_post_init"] = {str(r): float(np.median(cos[ok & post & (rhos == r)])) for r in sorted(set(rhos.tolist()))}
    res["median_final_summed_loss"] = float(np.median([r["loss"].sum() for r in rows if r["step"] == CKPTS[-1] and not r["diverged"]]))
    res["median_initial_summed_loss"] = float(np.median([r["loss"].sum() for r in rows if r["step"] == 0 and not r["diverged"]]))
    res["layer_fro_maxmin_median"] = float(np.median([r["layer_fro"].max() / r["layer_fro"].min() for r in rows if not r["diverged"]]))
    # per-layer sign statistics: fraction of cells where a_l < 0, by layer
    res["frac_a_l_negative_by_layer"] = [float((A[ok, l] < 0).mean()) for l in range(A.shape[1])]
    res["frac_a_l_negative_by_layer_post_init"] = [float((A[ok & post, l] < 0).mean()) for l in range(A.shape[1])]
    # per-layer share of |a_l| in sum |a_l|
    res["mean_share_abs_a_l_by_layer"] = [float(np.mean(np.abs(A[ok, l]) / np.abs(A[ok]).sum(1))) for l in range(A.shape[1])]
    res["random_gauge_flips_W_only"] = random_gauge_flips(A[ok])
    bnf = box_nearest_flip(A[ok])
    fin = np.isfinite(bnf)
    res["box_nearest_flip_unconstrained"] = {"median_of_mixed": float(np.median(bnf[fin])) if fin.any() else None,
                                             "frac_cells_within_sqrt2": float((bnf <= np.sqrt(2)).mean()),
                                             "frac_cells_within_2": float((bnf <= 2).mean())}
    if has_b:
        L = A.shape[1]
        # product-one group: W_1 and b_1 always share the coefficient omega_1; b_L has coefficient 1
        terms_p1 = np.concatenate([(A[:, :1] + Ab[:, :1]), A[:, 1:], Ab[:, 1:]], 1)
        terms_headabs = np.concatenate([(A[:, :1] + Ab[:, :1]), A[:, 1:], Ab[:, 1:]], 1)  # same terms; b_L isolable
        terms_p1_noLast = np.concatenate([(A[:, :1] + Ab[:, :1]), A[:, 1:], Ab[:, 1:-1]], 1)
        mixed_all = mixed_of(terms_p1); mixed_noLast = mixed_of(terms_p1_noLast)
        mixed_b = mixed_of(Ab)
        for key, mask in (("all", ok), ("post_init", ok & post)):
            res[f"mixed_frac_W_and_b_terms_{key}"] = boot_ci(mixed_all[mask], seeds[mask])
            res[f"mixed_frac_W_and_b_terms_excl_last_bias_{key}"] = boot_ci(mixed_noLast[mask], seeds[mask])
            res[f"mixed_frac_bias_terms_only_{key}"] = boot_ci(mixed_b[mask], seeds[mask])
        res["bias_share_of_abs_numerator_median"] = float(np.median(np.abs(Ab[ok]).sum(1) / (np.abs(A[ok]).sum(1) + np.abs(Ab[ok]).sum(1))))
        res["frac_b_l_negative_by_layer"] = [float((Ab[ok, l] < 0).mean()) for l in range(L)]
        res["conflict_rate_W_only_all"] = boot_ci((A[ok].sum(1) < 0), seeds[ok])
        res["sign_W_only_vs_full_disagree_frac"] = float((np.sign(A[ok].sum(1)) != np.sign(S[ok])).mean())
        res["random_gauge_flips_W_and_b_terms"] = random_gauge_flips_biased(A[ok], Ab[ok])
    if "neuron_flip_by_layer" in rows[0]:
        NF = np.stack([r["neuron_flip_by_layer"] for r in rows if not r["diverged"]])   # cells x (L-1)
        best = NF.min(1); arg = NF.argmin(1)
        res["neuron"] = {
            "nearest_flip_median_all_layers": float(np.median(best)),
            "nearest_flip_q25_q75": [float(np.percentile(best, 25)), float(np.percentile(best, 75))],
            "within_sqrt2_frac": float((best <= np.sqrt(2)).mean()), "within_2_frac": float((best <= 2).mean()),
            "within_3_frac": float((best <= 3).mean()),
            "nearest_layer_histogram": [int((arg == l).sum()) for l in range(NF.shape[1])],
            "nearest_layer_histogram_post_init": [int((arg[post[ok]] == l).sum()) for l in range(NF.shape[1])],
            "median_flip_by_layer": [float(np.median(NF[:, l])) for l in range(NF.shape[1])],
            "within_2_frac_by_layer": [float((NF[:, l] <= 2).mean()) for l in range(NF.shape[1])],
            "within_sqrt2_frac_by_layer": [float((NF[:, l] <= np.sqrt(2)).mean()) for l in range(NF.shape[1])],
            "median_frac_neg_incoming_by_layer": [float(v) for v in np.median(np.stack([r["frac_neg_incoming_by_layer"] for r in rows if not r["diverged"]]), 0)],
            "median_frac_neg_outgoing_by_layer": [float(v) for v in np.median(np.stack([r["frac_neg_outgoing_by_layer"] for r in rows if not r["diverged"]]), 0)],
            "median_flip_by_step": {int(k): float(np.median(best[steps[ok] == k])) for k in CKPTS},
        }
    if "label_agreement" in rows[0]:
        la = np.array([r["label_agreement"] for r in rows if "label_agreement" in r]); rr = np.array([r["rho"] for r in rows if "label_agreement" in r])
        res["label_agreement_by_rho"] = {str(r): float(la[rr == r].mean()) for r in sorted(set(rr.tolist()))}
    out.update(res)
    return out


def random_gauge_flips_biased(A, Ab, n_draws=2000, seed=0):
    """Random product-one per-layer gauges applied to the biased numerator sum_l c_l^-2 a_l^W + P_l^-2 a_l^b."""
    L = A.shape[1]; rng = np.random.default_rng(seed)
    S0 = np.sign(A.sum(1) + Ab.sum(1))
    out = {}
    for lo, hi, key in ((1 / np.sqrt(2), np.sqrt(2), "sqrt2"), (0.5, 2.0, "2")):
        lc = rng.uniform(np.log(lo), np.log(hi), size=(n_draws, L))
        lc -= lc.mean(1, keepdims=True)                      # product one (largest factor shrinks accordingly)
        P = np.cumsum(lc, 1)
        N = A @ np.exp(-2 * lc).T + Ab @ np.exp(-2 * P).T
        fl = np.sign(N) != S0[:, None]
        out[key] = {"mean_frac": float(fl.mean()), "max_frac": float(fl.mean(0).max()),
                    "realized_max_factor_median": float(np.median(np.exp(np.abs(lc).max(1))))}
    return out


# ---------------------------------------------------------------------------- MGDA (probe d)
def mgda_exact(M):
    """min_{alpha in simplex} alpha^T M alpha by enumerating supports (K <= 6). Returns alpha."""
    K = M.shape[0]; best, best_a = np.inf, None
    for r in range(1, K + 1):
        for Ssup in itertools.combinations(range(K), r):
            idx = list(Ssup)
            Ms = M[np.ix_(idx, idx)]
            try:
                x = np.linalg.solve(Ms, np.ones(r))
            except np.linalg.LinAlgError:
                continue
            if x.sum() <= 0 or np.any(x < -1e-12):
                continue
            a = np.zeros(K); a[idx] = x / x.sum()
            val = a @ M @ a
            if val < best:
                best, best_a = val, a
    return best_a


def mgda_fw(M, iters=2000):
    K = M.shape[0]; a = np.ones(K) / K
    for _ in range(iters):
        Ma = M @ a; t = int(np.argmin(Ma)); dd = a @ Ma
        den = dd - 2 * Ma[t] + M[t, t]
        g = 1.0 if den <= 0 else float(np.clip((dd - Ma[t]) / den, 0, 1))
        a = (1 - g) * a; a[t] += g
    return a


def summarize_K_task(rows, name, n_draws=300, seed=0):
    ok = np.array([not r["diverged"] for r in rows])
    seeds = np.array([r["seed"] for r in rows]); steps = np.array([r["step"] for r in rows]); rhos = np.array([r["rho"] for r in rows])
    G = np.stack([r["gram"] for r in rows])          # cells x K x K x L
    K, L = G.shape[1], G.shape[3]
    out = {"name": name, "K": K, "n_cells": int(len(rows)), "n_diverged": int((~ok).sum()), "n_seeds": int(len(np.unique(seeds)))}
    pairs = list(itertools.combinations(range(K), 2))
    A = np.stack([G[:, i, j, :] for i, j in pairs], 1)  # cells x P x L
    mixed_pair = (np.nanmax(A, 2) > 0) & (np.nanmin(A, 2) < 0)
    Ssum = G.sum(3)                                     # cells x K x K
    d = np.sqrt(np.einsum("cii->ci", Ssum))
    cosm = Ssum / (d[:, :, None] * d[:, None, :])
    conf_pair = np.stack([cosm[:, i, j] < 0 for i, j in pairs], 1)
    post = steps > 0
    for key, mask in (("all", ok), ("post_init", ok & post)):
        out[f"pairwise_mixed_frac_{key}"] = boot_ci(mixed_pair[mask].ravel(), np.repeat(seeds[mask], len(pairs)))
        out[f"any_pair_mixed_frac_{key}"] = boot_ci(mixed_pair[mask].any(1), seeds[mask])
        out[f"all_pairs_single_sign_frac_{key}"] = boot_ci(~mixed_pair[mask].any(1), seeds[mask])
        out[f"pairwise_conflict_rate_{key}"] = boot_ci(conf_pair[mask].ravel(), np.repeat(seeds[mask], len(pairs)))
        out[f"any_pair_conflict_frac_{key}"] = boot_ci(conf_pair[mask].any(1), seeds[mask])
    out["pairwise_mixed_by_rho"] = {str(r): float(mixed_pair[ok & (rhos == r)].mean()) for r in sorted(set(rhos.tolist()))}
    out["pairwise_conflict_by_rho"] = {str(r): float(conf_pair[ok & (rhos == r)].mean()) for r in sorted(set(rhos.tolist()))}
    out["pairwise_conflict_by_step"] = {int(k): float(conf_pair[ok & (steps == k)].mean()) for k in CKPTS}
    out["median_pairwise_cos_post_init"] = float(np.median(np.stack([cosm[:, i, j] for i, j in pairs], 1)[ok & post]))
    # MGDA under the gauge
    rng = np.random.default_rng(seed)
    idx = np.where(ok)[0]
    a1 = np.stack([mgda_exact(G[c].sum(2)) for c in idx])
    fw_dev = max(np.abs(mgda_fw(G[c].sum(2)) - a1[i]).max() for i, c in enumerate(idx[:100]))
    out["mgda_fw_vs_exact_max_abs_dev_first100"] = float(fw_dev)
    out["mgda_alpha_at_1"] = {"median_max_weight": float(np.median(a1.max(1))), "frac_boundary_(some_alpha_0)": float((a1.min(1) < 1e-10).mean()),
                              "frac_single_task": float((a1.max(1) > 1 - 1e-10).mean()),
                              "mean_support_size": float((a1 > 1e-10).sum(1).mean())}
    for lo, hi, key in ((1 / np.sqrt(2), np.sqrt(2), "sqrt2"), (0.5, 2.0, "2"), (0.25, 4.0, "4")):
        dmax, supp_change, argmax_change, dl1 = [], [], [], []
        for _ in range(n_draws):
            cvec = np.exp(rng.uniform(np.log(lo), np.log(hi), size=L)); w = cvec ** -2.0
            for i, cidx in enumerate(idx):
                ac = mgda_exact((G[cidx] * w).sum(2))
                dmax.append(np.abs(ac - a1[i]).max()); dl1.append(np.abs(ac - a1[i]).sum())
                supp_change.append(bool(np.any((ac > 1e-10) != (a1[i] > 1e-10))))
                argmax_change.append(bool(ac.argmax() != a1[i].argmax()))
        dmax = np.array(dmax); dl1 = np.array(dl1)
        out[f"mgda_gauge_{key}"] = {"max_abs_dalpha_median": float(np.median(dmax)), "max_abs_dalpha_q90": float(np.percentile(dmax, 90)),
                                    "max_abs_dalpha_max": float(dmax.max()), "l1_dalpha_median": float(np.median(dl1)),
                                    "support_change_frac": float(np.mean(supp_change)), "argmax_change_frac": float(np.mean(argmax_change)),
                                    "frac_pairs_dalpha_gt_0.1": float((dmax > 0.1).mean()), "frac_pairs_dalpha_gt_0.25": float((dmax > 0.25).mean()),
                                    "n_draws": n_draws}
    return out


# ============================================================================ conjugacy checks
def check_bias_conjugacy(c, seed=10, steps=50, eta=0.05, bias="torch", depth=4, width=64):
    rng = np.random.default_rng(1000 + seed)
    X, Ys = make_tasks(rng, 256, 16, 0.3); W, H = init_net(rng, 16, width, depth); b = init_bias(rng, W, bias)
    c = np.asarray(c, float); P = np.cumprod(c)
    assert abs(np.prod(c) - 1) < 1e-12
    A = train(X, Ys, W, b, H, [eta] * depth, [eta] * depth, eta, steps, set())
    B = train(X, Ys, [cl * w for cl, w in zip(c, W)], [pl * v for pl, v in zip(P, b)], H,
              [eta * cl ** 2 for cl in c], [eta * pl ** 2 for pl in P], eta, steps, set())
    lA, lB = np.array(A["loss"]), np.array(B["loss"])
    WA, bA, HA = A["final"]; WB, bB, HB = B["final"]
    return {"c": c.tolist(), "P": P.tolist(), "max_rel_loss_diff": float(np.max(np.abs(lA - lB) / np.abs(lA))),
            "max_abs_W_dev": float(max(np.max(np.abs(wb - cl * wa)) for cl, wa, wb in zip(c, WA, WB))),
            "max_abs_b_dev": float(max(np.max(np.abs(vb - pl * va)) for pl, va, vb in zip(P, bA, bB))),
            "max_abs_H_dev": float(max(np.max(np.abs(hb - ha)) for ha, hb in zip(HA, HB))),
            "final_loss": lA[-1].tolist()}


def check_adam_conjugacy(c, seed=10, steps=50, eta=1e-2, adam_eps=0.0, power=1.0, depth=4, width=64):
    """Run A: Adam(eta) from W. Run B: Adam(c_l^power eta) from c_l W. power = 1 is the claimed compensation."""
    rng = np.random.default_rng(1000 + seed)
    X, Ys = make_tasks(rng, 256, 16, 0.3); W, H = init_net(rng, 16, width, depth)
    c = np.asarray(c, float)
    A = train(X, Ys, W, None, H, [eta] * depth, None, eta, steps, set(), opt="adam", adam_eps=adam_eps)
    B = train(X, Ys, [cl * w for cl, w in zip(c, W)], None, H, [eta * cl ** power for cl in c], None, eta, steps, set(),
              opt="adam", adam_eps=adam_eps)
    lA, lB = np.array(A["loss"]), np.array(B["loss"])
    WA, _, HA = A["final"]; WB, _, HB = B["final"]
    return {"c": c.tolist(), "eta": eta, "adam_eps": adam_eps, "power": power, "steps": steps,
            "max_rel_loss_diff": float(np.max(np.abs(lA - lB) / np.abs(lA))),
            "final_rel_loss_diff": float(np.max(np.abs(lA[-1] - lB[-1]) / np.abs(lA[-1]))),
            "max_abs_W_dev": float(max(np.max(np.abs(wb - cl * wa)) for cl, wa, wb in zip(c, WA, WB))),
            "max_rel_W_dev": float(max(np.max(np.abs(wb - cl * wa)) / np.max(np.abs(cl * wa)) for cl, wa, wb in zip(c, WA, WB))),
            "final_loss": lA[-1].tolist()}


def check_ln_conjugacy(c, seed=10, steps=50, eta=0.05, ln_eps=0.0, depth=4, width=64):
    rng = np.random.default_rng(1000 + seed)
    X, Ys = make_tasks(rng, 256, 16, 0.3); W, H = init_net(rng, 16, width, depth)
    c = np.asarray(c, float)
    # function-alone symmetry at fixed weights (no product constraint, heads untouched)
    o0 = fwd(X, W, None, H, ln=True, ln_eps=ln_eps)[2]; o1 = fwd(X, [cl * w for cl, w in zip(c, W)], None, H, ln=True, ln_eps=ln_eps)[2]
    fdev = max(float(np.max(np.abs(a - b2))) for a, b2 in zip(o0, o1))
    # GD trajectories with and without compensation
    A = train(X, Ys, W, None, H, [eta] * depth, None, eta, steps, set(), ln=True, ln_eps=ln_eps)
    B = train(X, Ys, [cl * w for cl, w in zip(c, W)], None, H, [eta * cl ** 2 for cl in c], None, eta, steps, set(), ln=True, ln_eps=ln_eps)
    Bu = train(X, Ys, [cl * w for cl, w in zip(c, W)], None, H, [eta] * depth, None, eta, steps, set(), ln=True, ln_eps=ln_eps)
    lA, lB, lBu = np.array(A["loss"]), np.array(B["loss"]), np.array(Bu["loss"])
    WA, _, _ = A["final"]; WB, _, _ = B["final"]
    return {"c": c.tolist(), "prod_c": float(np.prod(c)), "ln_eps": ln_eps, "function_max_abs_output_dev_at_fixed_weights": fdev,
            "compensated_max_rel_loss_diff": float(np.max(np.abs(lA - lB) / np.abs(lA))),
            "compensated_max_abs_W_dev": float(max(np.max(np.abs(wb - cl * wa)) for cl, wa, wb in zip(c, WA, WB))),
            "uncompensated_max_rel_loss_diff": float(np.max(np.abs(lA - lBu) / np.abs(lA))),
            "uncompensated_final_loss_ratio": (lBu[-1] / lA[-1]).tolist(), "final_loss": lA[-1].tolist()}


def selftest():
    rng = np.random.default_rng(1010)
    X, Ys = make_tasks(rng, 256, 16, 0.3); W, H = init_net(rng, 16, 64, 4)
    ls0, gW0, gH0 = grads(X, Ys, W, H)
    ls1, gW1, gb1, gH1 = grads_gen(X, Ys, W, None, H)
    dev = max(np.max(np.abs(a - b2)) for k in range(2) for a, b2 in zip(gW0[k], gW1[k]))
    devH = max(np.max(np.abs(gH0[k][k] - gH1[k])) for k in range(2))
    h0 = gd_run(X, Ys, W, H, [0.05] * 4, 0.05, 30)
    h1 = train(X, Ys, W, None, H, [0.05] * 4, None, 0.05, 30, set())
    devT = max(np.max(np.abs(a - b2)) for a, b2 in zip(h0["final_W"], h1["final"][0]))
    out = {"grad_max_abs_dev_vs_reference": float(dev), "head_grad_dev": float(devH), "gd_30_steps_max_abs_W_dev_vs_reference": float(devT),
           "loss_dev": float(np.max(np.abs(np.array(ls0) - np.array(ls1))))}
    # finite-difference check of the LN + bias + softmax gradient
    rng = np.random.default_rng(1011)
    Xc, Yc = make_tasks_cls(rng, 64, 16, 0.3); Wc, Hc = init_net(rng, 16, 16, 3, d_out=4); bc = init_bias(rng, Wc, "torch")
    ls, gW, gb, gH = grads_gen(Xc, Yc, Wc, bc, Hc, task="cls", ln=True, ln_eps=1e-5)
    eps = 1e-6; errs = []
    for trial in range(20):
        l = rng.integers(0, 3); i = rng.integers(0, Wc[l].shape[0]); j = rng.integers(0, Wc[l].shape[1])
        Wp = [w.copy() for w in Wc]; Wm = [w.copy() for w in Wc]; Wp[l][i, j] += eps; Wm[l][i, j] -= eps
        fd = (sum(grads_gen(Xc, Yc, Wp, bc, Hc, "cls", True, 1e-5)[0]) - sum(grads_gen(Xc, Yc, Wm, bc, Hc, "cls", True, 1e-5)[0])) / (2 * eps)
        errs.append(abs(fd - (gW[0][l][i, j] + gW[1][l][i, j])))
        bp = [v.copy() for v in bc]; bm = [v.copy() for v in bc]; bp[l][j] += eps; bm[l][j] -= eps
        fd = (sum(grads_gen(Xc, Yc, Wc, bp, Hc, "cls", True, 1e-5)[0]) - sum(grads_gen(Xc, Yc, Wc, bm, Hc, "cls", True, 1e-5)[0])) / (2 * eps)
        errs.append(abs(fd - (gb[0][l][j] + gb[1][l][j])))
    out["ln_bias_softmax_finite_difference_max_abs_err"] = float(max(errs))
    return out


def conjugacy_block():
    out = {"selftest": selftest()}
    out["bias_gd"] = [check_bias_conjugacy([3, 1 / 3, 1, 1]), check_bias_conjugacy([2, 0.5, 2, 0.5]),
                      check_bias_conjugacy([2, 0.5, 2, 0.5], bias="zero"), check_bias_conjugacy([0.5, 4, 0.5, 1], steps=400)]
    out["adam"] = {}
    out["adam"]["eps0_power1"] = [check_adam_conjugacy([3, 1 / 3, 1, 1]), check_adam_conjugacy([2, 0.5, 2, 0.5], steps=400),
                                  check_adam_conjugacy([10, 0.1, 1, 1], steps=400, eta=1e-3)]
    out["adam"]["eps0_power2_(wrong)"] = [check_adam_conjugacy([3, 1 / 3, 1, 1], power=2.0)]
    out["adam"]["eps1e-8_power1"] = [check_adam_conjugacy(cc, steps=st, adam_eps=1e-8, eta=et)
                                     for cc in ([2, 0.5, 1, 1], [10, 0.1, 1, 1]) for st in (50, 400) for et in (1e-3, 1e-2)]
    out["ln_gd"] = [check_ln_conjugacy([3, 0.5, 1, 2], ln_eps=0.0), check_ln_conjugacy([3, 0.5, 1, 2], ln_eps=1e-5),
                    check_ln_conjugacy([3, 0.5, 1, 2], ln_eps=1e-5, steps=400), check_ln_conjugacy([0.1, 1, 1, 10], ln_eps=0.0, steps=400)]
    return out


# ============================================================================ configs
def configs():
    C = {}
    # (a) depth and width (bias-free GD, reference functions)
    for d in (2, 3, 4, 5, 6, 8):
        C[f"a_depth{d}_w64"] = dict(depth=d, use_reference=True, neuron=(d == 4))
    for w in (16, 32, 128):
        C[f"a_depth4_w{w}"] = dict(width=w, use_reference=True)
    C["a_depth4_w256_eta0.05"] = dict(width=256, use_reference=True, n_seeds=10)   # expected to diverge; divergence count is the datum
    C["a_depth4_w256_eta0.0125"] = dict(width=256, eta=0.0125, use_reference=True, n_seeds=20)
    C["a_depth4_w512_eta0.00625"] = dict(width=512, eta=0.00625, use_reference=True, n_seeds=10)
    C["a_depth4_w16_eta0.2"] = dict(width=16, eta=0.2, use_reference=True)
    # (b) biases
    C["b_bias_torch_eta0.05"] = dict(bias="torch")
    C["b_bias_zero_eta0.05"] = dict(bias="zero")
    C["b_bias_torch_eta0.01"] = dict(bias="torch", eta=0.01)
    # (c) Adam, eps = 0
    C["c_adam_eps0_lr1e-3"] = dict(opt="adam", eta=1e-3, adam_eps=0.0)
    C["c_adam_eps0_lr1e-2"] = dict(opt="adam", eta=1e-2, adam_eps=0.0)
    C["c_adam_eps0_lr3e-2"] = dict(opt="adam", eta=3e-2, adam_eps=0.0)
    C["c_adam_eps1e-8_lr1e-3"] = dict(opt="adam", eta=1e-3, adam_eps=1e-8)
    # (d) K tasks
    for K in (3, 5):
        C[f"d_K{K}_tasks"] = dict(K=K, rhos=(0.0, 0.3, 0.7))
    # (e) classification
    C["e_cls_eta0.05"] = dict(task="cls")
    C["e_cls_eta0.5"] = dict(task="cls", eta=0.5)
    # (f) LayerNorm
    C["f_ln_eta0.05"] = dict(ln=True, ln_eps=1e-5)
    C["f_ln_eta0.2"] = dict(ln=True, ln_eps=1e-5, eta=0.2)   # probe: at eta >= 0.5 the LN net collapses to zero output (summed loss pinned at 1.0)
    return C


def run_config(name, cfg):
    from joblib import Parallel, delayed
    c = dict(CFG_DEFAULTS); c.update(cfg)
    t0 = time.time()
    seeds = list(range(SEED0, SEED0 + c["n_seeds"]))
    res = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(run_seed)(cfg, s) for s in seeds)
    rows = [r for rr in res for r in rr]
    summ = summarize_K_task(rows, name, n_draws=150) if c["K"] > 2 else summarize_two_task(rows, name)
    summ["cfg"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()}
    summ["runtime_s"] = time.time() - t0
    return rows, summ


def rows_to_arrays(rows, name):
    arr = {}
    keys = ["seed", "rho", "step", "loss", "layer_fro", "gram"]
    if rows[0].get("gram_b") is not None:
        keys.append("gram_b")
    for k in ("neuron_flip_by_layer", "frac_neg_incoming_by_layer", "frac_neg_outgoing_by_layer"):
        if k in rows[0]:
            keys.append(k)
    for k in keys:
        vals = []
        for r in rows:
            v = r.get(k)
            if v is None:
                v = np.full_like(np.asarray(rows[0][k], float), np.nan)
            vals.append(np.asarray(v, float))
        arr[f"{name}__{k}"] = np.stack(vals)
    arr[f"{name}__diverged"] = np.array([r["diverged"] for r in rows])
    return arr


def main():
    open(LOG, "w").close()
    t_start = time.time()
    out = {"seeds": f"rng default_rng(1000 + s), s in [{SEED0}, {SEED0 + N_SEEDS - 1}] (n_seeds per config in cfg)",
           "checkpoints": list(CKPTS), "rhos_default": list(RHOS), "n_jobs": N_JOBS}
    log("conjugacy checks ...")
    out["conjugacy"] = conjugacy_block()
    log(json.dumps(out["conjugacy"]["selftest"]))
    for k, v in out["conjugacy"].items():
        if k != "selftest":
            log(f"  {k}: {json.dumps(v)[:1500]}")
    arrays = {}
    out["configs"] = {}
    for name, cfg in configs().items():
        rows, summ = run_config(name, cfg)
        out["configs"][name] = summ
        arrays.update(rows_to_arrays(rows, name))
        if summ.get("cfg", {}).get("K", 2) > 2:
            log(f"{name:28s} n={summ['n_cells']} div={summ['n_diverged']} pair-mixed {summ['pairwise_mixed_frac_all']['est']:.3f} "
                f"any-mixed {summ['any_pair_mixed_frac_all']['est']:.3f} pair-conflict {summ['pairwise_conflict_rate_all']['est']:.3f} "
                f"MGDA sqrt2 dalpha med {summ['mgda_gauge_sqrt2']['max_abs_dalpha_median']:.4f} max {summ['mgda_gauge_sqrt2']['max_abs_dalpha_max']:.3f} "
                f"supp-change {summ['mgda_gauge_sqrt2']['support_change_frac']:.3f}  [{summ['runtime_s']:.0f}s]")
        elif "mixed_frac_all" in summ:
            m, cf = summ["mixed_frac_all"], summ["conflict_rate_all"]
            log(f"{name:28s} n={summ['n_cells']} div={summ['n_diverged']} mixed {m['est']:.3f} [{m['lo']:.3f},{m['hi']:.3f}] "
                f"conflict {cf['est']:.3f} [{cf['lo']:.3f},{cf['hi']:.3f}] post-init conflict {summ['conflict_rate_post_init']['est']:.3f} "
                f"by step {[round(v, 2) for v in summ['conflict_by_step'].values()]} rand sqrt2 {summ['random_gauge_flips_W_only']['sqrt2']['mean_frac']:.3f} "
                f"final loss {summ['median_final_summed_loss']:.4f}  [{summ['runtime_s']:.0f}s]")
        else:
            log(f"{name:28s} n={summ['n_cells']} div={summ['n_diverged']} (no finite cells)  [{summ['runtime_s']:.0f}s]")
        json.dump(out, open(os.path.join(HERE, "exploratory.json"), "w"), indent=1, allow_nan=True)
        np.savez_compressed(os.path.join(HERE, "exploratory_cells.npz"), **arrays)
    out["total_runtime_min"] = (time.time() - t_start) / 60
    json.dump(out, open(os.path.join(HERE, "exploratory.json"), "w"), indent=1, allow_nan=True)
    log(f"done in {out['total_runtime_min']:.1f} min")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        print(json.dumps(conjugacy_block(), indent=1))
    else:
        main()
