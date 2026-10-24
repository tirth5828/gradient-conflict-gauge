"""methods_scale.py -- PCGrad, CAGrad (exact two-task), MGDA at scale, and their gauge dependence.

Everything reuses the model/data/GD code of conjugation_check.py (imported, not copied).
Setup as in the abstract: bias-free ReLU trunk (width 64, depth 4, d_in 16), two linear heads,
two regression targets at correlation rho, n = 256, full-batch GD on the summed loss, float64.

Parts
  1/2. From step 0, on seeds 10..10+N-1 (rng streams 1000 + s, disjoint from the development seeds 0-9;
       inputs / teachers / W_0 shared across rho inside a seed, as in the census) and
       rho in {-0.5,-0.2,0,0.3,0.7}: summed GD, PCGrad (symmetric trunk projection when the
       trunk cosine is negative), CAGrad (exact two-task closed form, c in {0.5, 0.2, 0.8}; two
       normalisations, see below), MGDA (two-task min-norm weight, sum-scaled). eta = 0.05 x 400
       steps, and eta = 0.01 x 400 steps on the first 100 seeds. Per run: final losses, losses
       averaged over the last 10 steps, fraction of steps on which the method's direction differs
       from the summed direction, relative deviation ||u - u_sum||/||u_sum|| per step, angle to
       u_sum, norm ratio, trunk cosine per step, and at checkpoints 0/99/199/299/399 the
       per-layer a_l = <g_1^l, g_2^l> and per-layer squared gradient norms.
  2b.  Gauge dependence of the Gram-matrix methods in closed form at the abstract's Fig. 1 cell
       (seed 4, rho 0.3, step 99) along (c, 1/c, 1, 1) and over the 52 mixed cells of the base
       census (seeds 0-4, replicated deliberately) under random per-layer gauges within sqrt2
       and within 2: relative change of the INDUCED update in the original coordinates,
       v_l(c) = c_l u_l^(c), where u^(c) is the method's update computed from the gauge
       gradients g_i^l / c_l (exact: the conjugate run's gradients are exactly g_i^l / c_l).
  3.   Continuation at scale: for every cell of the eta = 0.05 census (steps 0..299) whose sign
       the nearest product-one per-layer gauge flips within factor 2, continue in the flipped
       gauge with PCGrad vs summed GD vs same-norm random kicks vs the reversed correction
       (anti-PCGrad), and PCGrad vs summed at c = 1.

CAGrad normalisation. The official code (rescale = 1, its default) returns
  d / (1 + c^2),  d = g_0 + (c ||g_0|| / ||g_w*||) g_w*,  g_0 = (g_1 + g_2)/2,
which for g_1 = g_2 = g is ((1+c)/(1+c^2)) g, i.e. NOT the plain-GD step on the summed loss;
we scale it by 2 (sum instead of mean) and call it 'off' (official). The step-matched
normalisation 2 d / (1 + c) (the code's rescale = 2 option) coincides with the summed update
whenever g_1 = g_2, so that any difference from summed GD is the Gram re-weighting and not a
longer step; it is the primary variant here.

Run: python methods_scale.py [--n_seeds 300 --n_seeds_slow 100 --n_jobs 3]
Writes methods_scale.json, methods_scale_cells.npz, ../figures/methods_scale_*.png.
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from conjugation_check import (make_tasks, init_net, gd_run, grads, layer_inner_products,  # noqa: E402
                               pcgrad_trunk, pcgrad_run, perturbed_continuation)

RHOS = (-0.5, -0.2, 0.0, 0.3, 0.7)
CKPTS = (0, 99, 199, 299, 399)
METHODS = ["sum", "pcgrad", "ca0.5", "ca0.2", "ca0.8", "ca0.5_off", "mgda"]
DEPTH, WIDTH, D_IN, N = 4, 64, 16, 256
SUB = None  # filled in main: subsampled step indices for the persisted per-step arrays


# ----------------------------------------------------------------------------- two-task closed forms
def cagrad_w(G11, G12, G22, c):
    """Exact two-task CAGrad inner minimiser w* in [0, 1] of
        F(w) = <g_w, g_0> + c ||g_0|| ||g_w||,   g_w = w g_1 + (1 - w) g_2,   g_0 = (g_1 + g_2)/2.
    F is convex (linear plus the norm of an affine function), so the minimum is at a stationary
    point (a root of the quadratic obtained by squaring F'(w) = 0, filtered by the sign condition)
    or at an endpoint. Validated against bounded Brent and a 20001-point grid on 400 trajectory
    iterates x 3 values of c (relative excess <= 1.3e-15) and 20000 random Grams."""
    a = G11 - 2 * G12 + G22; b = 2 * (G12 - G22); d = G22
    A = 0.5 * (G11 - G22); B = 0.5 * (G12 + G22)
    k = c * np.sqrt(max(0.25 * (G11 + 2 * G12 + G22), 0.0))

    def F(w):
        return A * w + B + k * np.sqrt(max(a * w * w + b * w + d, 0.0))
    cands = [0.0, 1.0]
    qa = 4 * a * a * k * k - 4 * a * A * A; qb = 4 * a * b * k * k - 4 * A * A * b; qc = k * k * b * b - 4 * A * A * d
    if abs(qa) > 1e-300:
        disc = qb * qb - 4 * qa * qc
        if disc >= 0:
            for r in ((-qb + np.sqrt(disc)) / (2 * qa), (-qb - np.sqrt(disc)) / (2 * qa)):
                if 0.0 <= r <= 1.0 and k * (2 * a * r + b) * (-A) >= -1e-12 * abs(A * k * (2 * a * r + b)) - 1e-300:
                    cands.append(float(r))
    elif abs(qb) > 1e-300:
        r = -qc / qb
        if 0.0 <= r <= 1.0:
            cands.append(float(r))
    w = min(cands, key=F)
    return w, F(w)


def method_betas(G11, G12, G22, method):
    """Update u = beta1 g_1 + beta2 g_2 of each method as a function of the trunk Gram matrix,
    plus an 'active' flag (the direction differs from the summed one) and the method's weight."""
    if method == "sum":
        return 1.0, 1.0, False, np.nan
    if method == "pcgrad":
        if G12 < 0:
            return 1.0 - G12 / G11, 1.0 - G12 / G22, True, np.nan
        return 1.0, 1.0, False, np.nan
    if method == "mgda":
        den = G11 + G22 - 2 * G12
        alpha = float(np.clip((G22 - G12) / den, 0.0, 1.0)) if den > 0 else 0.5
        return 2 * alpha, 2 * (1 - alpha), abs(alpha - 0.5) > 1e-9, alpha
    if method.startswith("ca"):
        c = float(method[2:].replace("_off", ""))
        w, _ = cagrad_w(G11, G12, G22, c)
        g0n = np.sqrt(max(0.25 * (G11 + 2 * G12 + G22), 0.0))
        gwn = np.sqrt(max(w * w * G11 + 2 * w * (1 - w) * G12 + (1 - w) ** 2 * G22, 0.0))
        lam = c * g0n / gwn if gwn > 0 else 0.0
        scale = 2.0 / (1 + c * c) if method.endswith("_off") else 2.0 / (1 + c)
        return scale * (0.5 + lam * w), scale * (0.5 + lam * (1 - w)), abs(w - 0.5) > 1e-9, w
    raise ValueError(method)


def nash_two_task(G11, G12, G22):
    """Exact two-task Nash-MTL solution of G alpha = 1/alpha (elementwise), alpha > 0:
    subtracting the two equations gives ||g_1|| alpha_1 = ||g_2|| alpha_2, whence
    alpha_i = 1 / (||g_i|| sqrt(1 + cos)); the update d = alpha_1 g_1 + alpha_2 g_2 is the
    bisector of the unit gradients, of norm sqrt2, and exists iff cos > -1."""
    n1, n2 = np.sqrt(G11), np.sqrt(G22); cos = G12 / (n1 * n2)
    if cos <= -1 + 1e-15:
        return np.nan, np.nan
    s = np.sqrt(1 + cos)
    return 1 / (n1 * s), 1 / (n2 * s)


# ----------------------------------------------------------------------------- method runs from step 0
def gram_from_layers(gW):
    G11 = sum(float(np.vdot(g, g)) for g in gW[0]); G22 = sum(float(np.vdot(g, g)) for g in gW[1])
    a = [float(np.vdot(p, q)) for p, q in zip(gW[0], gW[1])]
    return G11, sum(a), G22, a


def method_run(X, Ys, W, H, eta, steps, method, ckpts=CKPTS):
    """Run `method` on the trunk (heads take the summed step) for `steps` steps at uniform eta.
    Returns per-step diagnostics and checkpoint layer statistics."""
    W = [w.copy() for w in W]; H = [h.copy() for h in H]
    L = len(W)
    cos = np.full(steps, np.nan); dev = np.full(steps, np.nan); ang = np.full(steps, np.nan)
    nrm = np.full(steps, np.nan); act = np.zeros(steps, bool); wgt = np.full(steps, np.nan)
    loss = np.full((steps + 1, 2), np.nan)
    ck_a = np.full((len(ckpts), L), np.nan); ck_g1 = np.full((len(ckpts), L), np.nan); ck_g2 = np.full((len(ckpts), L), np.nan)
    diverged = False
    for t in range(steps):
        losses, gW, gH = grads(X, Ys, W, H)
        loss[t] = losses
        G11, G12, G22, a = gram_from_layers(gW)
        if not (np.isfinite(G11) and np.isfinite(G22) and np.isfinite(G12) and np.all(np.isfinite(losses))):
            diverged = True
            break
        if t in ckpts:
            i = ckpts.index(t)
            ck_a[i] = a; ck_g1[i] = [float(np.vdot(g, g)) for g in gW[0]]; ck_g2[i] = [float(np.vdot(g, g)) for g in gW[1]]
        cos[t] = G12 / np.sqrt(G11 * G22)
        b1, b2, active, wv = method_betas(G11, G12, G22, method)
        act[t] = active; wgt[t] = wv
        usum2 = G11 + 2 * G12 + G22
        uu = b1 * b1 * G11 + 2 * b1 * b2 * G12 + b2 * b2 * G22
        ud = (b1 - 1) ** 2 * G11 + 2 * (b1 - 1) * (b2 - 1) * G12 + (b2 - 1) ** 2 * G22
        us = b1 * (G11 + G12) + b2 * (G12 + G22)
        dev[t] = np.sqrt(max(ud, 0.0) / usum2); nrm[t] = np.sqrt(uu / usum2)
        ang[t] = float(np.degrees(np.arccos(np.clip(us / np.sqrt(uu * usum2), -1, 1))))
        for l in range(L):
            W[l] = W[l] - eta * (b1 * gW[0][l] + b2 * gW[1][l])
        for i in range(2):
            H[i] = H[i] - eta * (gH[0][i] + gH[1][i])
    if not diverged:
        losses, _, _ = grads(X, Ys, W, H)
        loss[steps] = losses
        if not np.all(np.isfinite(losses)):
            diverged = True
    return {"cos": cos, "dev": dev, "ang": ang, "nrm": nrm, "act": act, "wgt": wgt, "loss": loss,
            "ck_a": ck_a, "ck_g1": ck_g1, "ck_g2": ck_g2, "diverged": diverged}


def run_seed(s, eta, steps, methods=METHODS, rhos=RHOS):
    """All methods x rhos for one seed. rng stream 1000 + s (inputs/teachers/W_0 shared across rho)."""
    out = {}
    for ri, rho in enumerate(rhos):
        rng = np.random.default_rng(1000 + s)
        X, Ys = make_tasks(rng, N, D_IN, rho)
        W, H = init_net(rng, D_IN, WIDTH, DEPTH)
        for m in methods:
            out[(ri, m)] = method_run(X, Ys, W, H, eta, steps, m)
    return s, out


def collect(results, seeds, methods, steps, sub):
    """Pack per-seed dicts into arrays (seed, rho, method, ...)."""
    S, R, M = len(seeds), len(RHOS), len(methods)
    P = {"final": np.full((S, R, M, 2), np.nan), "last10": np.full((S, R, M, 2), np.nan),
         "last10_min": np.full((S, R, M, 2), np.nan), "last10_max": np.full((S, R, M, 2), np.nan),
         "act_frac": np.full((S, R, M), np.nan), "dev_median": np.full((S, R, M), np.nan), "dev_mean": np.full((S, R, M), np.nan),
         "dev_max": np.full((S, R, M), np.nan), "ang_median": np.full((S, R, M), np.nan), "ang_max": np.full((S, R, M), np.nan),
         "nrm_median": np.full((S, R, M), np.nan), "nrm_max": np.full((S, R, M), np.nan), "diverged": np.zeros((S, R, M), bool),
         "conflict_frac": np.full((S, R, M), np.nan),
         "ck_a": np.full((S, R, M, len(CKPTS), DEPTH), np.nan), "ck_g1": np.full((S, R, M, len(CKPTS), DEPTH), np.nan),
         "ck_g2": np.full((S, R, M, len(CKPTS), DEPTH), np.nan),
         "cos_sub": np.full((S, R, M, len(sub)), np.nan, np.float32), "dev_sub": np.full((S, R, M, len(sub)), np.nan, np.float32),
         "loss_sub": np.full((S, R, M, len(sub), 2), np.nan, np.float32), "wgt_sub": np.full((S, R, M, len(sub)), np.nan, np.float32)}
    for si, s in enumerate(seeds):
        d = results[s]
        for ri in range(R):
            for mi, m in enumerate(methods):
                r = d[(ri, m)]
                P["diverged"][si, ri, mi] = r["diverged"]
                P["final"][si, ri, mi] = r["loss"][steps]
                P["last10"][si, ri, mi] = r["loss"][steps - 9:steps + 1].mean(0)
                P["last10_min"][si, ri, mi] = r["loss"][steps - 9:steps + 1].min(0)
                P["last10_max"][si, ri, mi] = r["loss"][steps - 9:steps + 1].max(0)
                fin = np.isfinite(r["cos"])
                P["act_frac"][si, ri, mi] = r["act"].mean()
                P["conflict_frac"][si, ri, mi] = (r["cos"][fin] < 0).mean() if fin.any() else np.nan
                if fin.any():
                    P["dev_median"][si, ri, mi] = np.nanmedian(r["dev"]); P["dev_mean"][si, ri, mi] = np.nanmean(r["dev"]); P["dev_max"][si, ri, mi] = np.nanmax(r["dev"])
                    P["ang_median"][si, ri, mi] = np.nanmedian(r["ang"]); P["ang_max"][si, ri, mi] = np.nanmax(r["ang"])
                    P["nrm_median"][si, ri, mi] = np.nanmedian(r["nrm"]); P["nrm_max"][si, ri, mi] = np.nanmax(r["nrm"])
                P["ck_a"][si, ri, mi] = r["ck_a"]; P["ck_g1"][si, ri, mi] = r["ck_g1"]; P["ck_g2"][si, ri, mi] = r["ck_g2"]
                P["cos_sub"][si, ri, mi] = r["cos"][sub]; P["dev_sub"][si, ri, mi] = r["dev"][sub]
                P["loss_sub"][si, ri, mi] = r["loss"][sub]; P["wgt_sub"][si, ri, mi] = r["wgt"][sub]
    return P


# ----------------------------------------------------------------------------- statistics
def boot_ci(values, clusters, stat=np.median, n_boot=2000, seed=0):
    """Cluster (seed) bootstrap percentile 95% CI of `stat` over the finite entries of `values`."""
    values = np.asarray(values, float); clusters = np.asarray(clusters)
    ok = np.isfinite(values); values, clusters = values[ok], clusters[ok]
    if values.size == 0:
        return {"value": None, "ci": [None, None], "n": 0}
    uc = np.unique(clusters); idx = {c: np.where(clusters == c)[0] for c in uc}
    rng = np.random.default_rng(seed); bs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uc, size=uc.size, replace=True)
        bs[b] = stat(values[np.concatenate([idx[c] for c in pick])])
    return {"value": float(stat(values)), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": int(values.size)}


def summarize_from0(P, seeds, methods, key="final"):
    """Ratios method/summed of the final (or last-10-averaged) losses, pooled and by rho."""
    S, R, M = P["act_frac"].shape
    seed_id = np.repeat(np.array(seeds)[:, None], R, 1)
    out = {}
    ref = P[key][:, :, methods.index("sum"), :]
    for mi, m in enumerate(methods):
        rat = P[key][:, :, mi, :] / ref
        rat[P["diverged"][:, :, mi] | P["diverged"][:, :, methods.index("sum")]] = np.nan
        d = {"n_diverged": int(P["diverged"][:, :, mi].sum()),
             "ratio_task1": boot_ci(rat[..., 0].ravel(), seed_id.ravel()), "ratio_task2": boot_ci(rat[..., 1].ravel(), seed_id.ravel()),
             "ratio_task1_mean": boot_ci(rat[..., 0].ravel(), seed_id.ravel(), np.mean), "ratio_task2_mean": boot_ci(rat[..., 1].ravel(), seed_id.ravel(), np.mean),
             "geomean_ratio_median": boot_ci(np.sqrt(rat[..., 0] * rat[..., 1]).ravel(), seed_id.ravel()),
             "frac_both_below_1": boot_ci(((rat[..., 0] < 1) & (rat[..., 1] < 1)).astype(float).ravel(), seed_id.ravel(), np.mean),
             "frac_sum_below_1": boot_ci(((rat[..., 0] + rat[..., 1]) < 2).astype(float).ravel(), seed_id.ravel(), np.mean),
             "ratio_task1_q10_q90": [float(np.nanpercentile(rat[..., 0], 10)), float(np.nanpercentile(rat[..., 0], 90))],
             "ratio_task2_q10_q90": [float(np.nanpercentile(rat[..., 1], 10)), float(np.nanpercentile(rat[..., 1], 90))],
             "active_frac_median": boot_ci(P["act_frac"][:, :, mi].ravel(), seed_id.ravel()),
             "active_frac_mean": boot_ci(P["act_frac"][:, :, mi].ravel(), seed_id.ravel(), np.mean),
             "dev_median_of_medians": boot_ci(P["dev_median"][:, :, mi].ravel(), seed_id.ravel()),
             "dev_mean_of_means": boot_ci(P["dev_mean"][:, :, mi].ravel(), seed_id.ravel(), np.mean),
             "dev_max_median": boot_ci(P["dev_max"][:, :, mi].ravel(), seed_id.ravel()),
             "angle_deg_median_of_medians": boot_ci(P["ang_median"][:, :, mi].ravel(), seed_id.ravel()),
             "angle_deg_max_median": boot_ci(P["ang_max"][:, :, mi].ravel(), seed_id.ravel()),
             "norm_ratio_median_of_medians": boot_ci(P["nrm_median"][:, :, mi].ravel(), seed_id.ravel()),
             "conflict_frac_of_steps_mean": boot_ci(P["conflict_frac"][:, :, mi].ravel(), seed_id.ravel(), np.mean),
             "by_rho": {}}
        for ri, rho in enumerate(RHOS):
            d["by_rho"][str(rho)] = {"ratio_task1": boot_ci(rat[:, ri, 0], seeds), "ratio_task2": boot_ci(rat[:, ri, 1], seeds),
                                     "active_frac_mean": boot_ci(P["act_frac"][:, ri, mi], seeds, np.mean),
                                     "conflict_frac_of_steps_mean": boot_ci(P["conflict_frac"][:, ri, mi], seeds, np.mean),
                                     "n_diverged": int(P["diverged"][:, ri, mi].sum())}
        out[m] = d
    return out


# ----------------------------------------------------------------------------- gauge coordinates (closed form)
def induced_updates(gW, c, methods=("sum", "pcgrad", "ca0.5", "ca0.2", "ca0.8", "mgda", "nash")):
    """In gauge c the gradients are g_i^l / c_l. Each method computes u^(c) from them; the step
    W'_l -= c_l^2 eta u^(c)_l maps back to W_l -= eta c_l u^(c)_l. Returns v_l = c_l u^(c)_l per
    method, concatenated over layers, and the gauge cosine."""
    c = np.asarray(c, float)
    g1 = [g / cl for g, cl in zip(gW[0], c)]; g2 = [g / cl for g, cl in zip(gW[1], c)]
    G11, G12, G22, _ = gram_from_layers([g1, g2])
    out = {}
    for m in methods:
        if m == "nash":
            b1, b2 = nash_two_task(G11, G12, G22)
        else:
            b1, b2, _, _ = method_betas(G11, G12, G22, m)
        out[m] = np.concatenate([(cl * (b1 * p + b2 * q)).ravel() for cl, p, q in zip(c, g1, g2)])
    return out, G12 / np.sqrt(G11 * G22)


def rel_change(v, v1):
    return float(np.linalg.norm(v - v1) / np.linalg.norm(v1))


def angle_deg(v, v1):
    return float(np.degrees(np.arccos(np.clip(v @ v1 / (np.linalg.norm(v) * np.linalg.norm(v1)), -1, 1))))


def cell_gradients(seed, rho, step, eta=0.05):
    rng = np.random.default_rng(1000 + seed)
    X, Ys = make_tasks(rng, N, D_IN, rho)
    W0, H0 = init_net(rng, D_IN, WIDTH, DEPTH)
    ref = gd_run(X, Ys, W0, H0, [eta] * DEPTH, eta, step + 1, record_every=1)
    Wk, Hk = ref["W"][step], ref["H"][step]
    ips, _, gW, gH = layer_inner_products(X, Ys, Wk, Hk)
    return X, Ys, Wk, Hk, gW, gH, ips


def gauge_coordinate_analysis(n_draws=200, seed=0):
    out = {}
    meths = ("sum", "pcgrad", "ca0.5", "ca0.2", "ca0.8", "mgda", "nash")
    # --- Fig. 1 cell along (c, 1/c, 1, 1)
    X, Ys, Wk, Hk, gW, gH, ips = cell_gradients(4, 0.3, 99)
    cs = np.logspace(-1, 1, 81)
    v1, cos1 = induced_updates(gW, np.ones(DEPTH), meths)
    path = {m: {"rel_change": [], "angle_deg": []} for m in meths}
    path["c"] = cs.tolist(); path["cos"] = []
    for cv in cs:
        c = np.ones(DEPTH); c[0], c[1] = cv, 1 / cv
        v, cosv = induced_updates(gW, c, meths); path["cos"].append(float(cosv))
        for m in meths:
            path[m]["rel_change"].append(rel_change(v[m], v1[m])); path[m]["angle_deg"].append(angle_deg(v[m], v1[m]))
    cosarr = np.array(path["cos"]); inwin = np.sign(cosarr) == np.sign(cos1)
    out["fig1_cell"] = {"seed": 4, "rho": 0.3, "step": 99, "cos_c1": float(cos1), "ips": ips, "path": path,
                        "max_rel_change_no_flip_window": {m: float(np.max(np.array(path[m]["rel_change"])[inwin])) for m in meths},
                        "max_rel_change_all": {m: float(np.max(path[m]["rel_change"])) for m in meths},
                        "rel_change_at_sqrt2": {m: float(np.interp(np.log(np.sqrt(2)), np.log(cs), path[m]["rel_change"])) for m in meths},
                        "rel_change_at_2": {m: float(np.interp(np.log(2.0), np.log(cs), path[m]["rel_change"])) for m in meths},
                        "angle_at_sqrt2": {m: float(np.interp(np.log(np.sqrt(2)), np.log(cs), path[m]["angle_deg"])) for m in meths},
                        "angle_at_2": {m: float(np.interp(np.log(2.0), np.log(cs), path[m]["angle_deg"])) for m in meths}}
    # eta-metric (invariant) CAGrad check at this cell: Gram computed as sum_l eta_l <.,.>_l is gauge-invariant,
    # so the induced update is flat along the path.
    def induced_eta_metric(gW, c, cval=0.5):
        c = np.asarray(c, float); etas = 0.05 * c ** 2
        g1 = [g / cl for g, cl in zip(gW[0], c)]; g2 = [g / cl for g, cl in zip(gW[1], c)]
        G11 = sum(e * float(np.vdot(g, g)) for e, g in zip(etas, g1)); G22 = sum(e * float(np.vdot(g, g)) for e, g in zip(etas, g2))
        G12 = sum(e * float(np.vdot(p, q)) for e, p, q in zip(etas, g1, g2))
        b1, b2, _, _ = method_betas(G11, G12, G22, f"ca{cval}")
        return np.concatenate([(cl * (b1 * p + b2 * q)).ravel() for cl, p, q in zip(c, g1, g2)])
    ve1 = induced_eta_metric(gW, np.ones(DEPTH))
    out["fig1_cell"]["eta_metric_cagrad_max_rel_change"] = float(max(rel_change(induced_eta_metric(gW, [cv, 1 / cv, 1, 1]), ve1) for cv in cs))

    # --- 52 mixed cells of the base census under random per-layer gauges
    res = json.load(open(os.path.join(HERE, "results.json")))
    rows = [r for r in res["census"]["rows"] if r["mixed"]]
    rng = np.random.default_rng(seed)
    fam = {}
    for lo, hi, key in ((1 / np.sqrt(2), np.sqrt(2), "logU[1/sqrt2,sqrt2]"), (0.5, 2.0, "logU[1/2,2]")):
        rec = {m: {"flip": [], "noflip": [], "ang_flip": [], "ang_noflip": []} for m in meths}
        cell_max = {m: [] for m in meths}
        n_flip = 0
        for r in rows:
            _, _, _, _, gWc, _, _ = cell_gradients(r["seed"], r["rho"], r["step"])
            v1c, cos1c = induced_updates(gWc, np.ones(DEPTH), meths)
            cm = {m: 0.0 for m in meths}
            for _ in range(n_draws):
                c = np.exp(rng.uniform(np.log(lo), np.log(hi), size=DEPTH)); c = c / np.prod(c) ** (1 / DEPTH)
                v, cosv = induced_updates(gWc, c, meths)
                flipped = np.sign(cosv) != np.sign(cos1c); n_flip += flipped
                for m in meths:
                    rc = rel_change(v[m], v1c[m]) if np.all(np.isfinite(v[m])) else np.nan
                    rec[m]["flip" if flipped else "noflip"].append(rc); cm[m] = max(cm[m], rc if np.isfinite(rc) else 0.0)
                    rec[m]["ang_flip" if flipped else "ang_noflip"].append(angle_deg(v[m], v1c[m]) if np.isfinite(rc) else np.nan)
            for m in meths:
                cell_max[m].append(cm[m])
        fam[key] = {"n_cells": len(rows), "n_draws_per_cell": n_draws, "frac_draws_flipped": n_flip / (len(rows) * n_draws),
                    "note": "random factors drawn logU then renormalised to product one (so the largest factor can slightly exceed the nominal bound)"}
        for m in meths:
            f = np.array(rec[m]["flip"]); nf = np.array(rec[m]["noflip"])
            fam[key][m] = {"rel_change_median_all": float(np.nanmedian(np.concatenate([f, nf]))),
                           "rel_change_q90_all": float(np.nanpercentile(np.concatenate([f, nf]), 90)),
                           "rel_change_max_all": float(np.nanmax(np.concatenate([f, nf]))),
                           "rel_change_median_noflip": float(np.nanmedian(nf)) if nf.size else None,
                           "rel_change_max_noflip": float(np.nanmax(nf)) if nf.size else None,
                           "rel_change_median_flip": float(np.nanmedian(f)) if f.size else None,
                           "rel_change_max_flip": float(np.nanmax(f)) if f.size else None,
                           "frac_noflip_draws_changed_above_1e-3": float(np.nanmean(nf > 1e-3)) if nf.size else None,
                           "per_cell_max_median": float(np.median(cell_max[m])), "per_cell_max_max": float(np.max(cell_max[m])),
                           "angle_deg_median_noflip": float(np.nanmedian(rec[m]["ang_noflip"])) if nf.size else None,
                           "angle_deg_max_noflip": float(np.nanmax(rec[m]["ang_noflip"])) if nf.size else None,
                           "angle_deg_median_flip": float(np.nanmedian(rec[m]["ang_flip"])) if f.size else None,
                           "angle_deg_max_flip": float(np.nanmax(rec[m]["ang_flip"])) if f.size else None}
    out["mixed_cells_random_gauges"] = fam
    return out


# ----------------------------------------------------------------------------- part 3: continuation at scale
_SLICE = None


def slice_grid(L=DEPTH):
    global _SLICE
    if _SLICE is None:
        n_g = max(9, int(round(2.3e5 ** (1.0 / (L - 1)))))
        u = np.linspace(-1.0, 1.0, n_g)
        grids = np.stack(np.meshgrid(*([u] * (L - 1)), indexing="ij"), -1).reshape(-1, L - 1)
        last = -grids.sum(1); feas = np.abs(last) <= 1 + 1e-12
        _SLICE = np.concatenate([grids[feas], last[feas, None]], 1)
    return _SLICE


def nearest_flip_prod1(a):
    """Nearest sign flip over product-one per-layer gauges by largest factor (as in conjugation_check.reachability):
    bisection on t over the slice grid; returns (t_flip, direction x with sum x = 0, max|x| <= 1) or (inf, None)."""
    a = np.asarray(a, float); s = np.sign(a.sum())
    if s == 0 or np.all(np.sign(a) == s) or np.all(np.sign(a) == -s):
        return np.inf, None
    Xg = slice_grid(len(a))
    maj = np.abs(a[np.sign(a) == s]).sum(); mino = np.abs(a[np.sign(a) == -s]).sum()
    lo = 0.25 * np.log(maj / mino)

    def vals(t):
        return (np.exp(-2.0 * t * Xg) * (s * a)).sum(1)
    v = vals(lo)
    if v.min() <= 0:
        return lo, Xg[int(np.argmin(v))]
    hi = np.log(1e3)
    if vals(hi).min() > 0:
        return np.inf, None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if vals(mid).min() <= 0:
            hi = mid
        else:
            lo = mid
    v = vals(hi)
    return hi, Xg[int(np.argmin(v))]


def continuation_cell(seed, rho, T, a, eta=0.05, total_steps=400, n_ctrl=4):
    t_flip, x = nearest_flip_prod1(a)
    rng = np.random.default_rng(1000 + seed)
    X, Ys = make_tasks(rng, N, D_IN, rho)
    W0, H0 = init_net(rng, D_IN, WIDTH, DEPTH)
    ref = gd_run(X, Ys, W0, H0, [eta] * DEPTH, eta, T + 1, record_every=1)
    Wk, Hk = ref["W"][T], ref["H"][T]
    ips, _, gWk, _ = layer_inner_products(X, Ys, Wk, Hk)
    s0 = np.sign(sum(ips))
    # gauge used: along the flipping ray, 1.25 x the flipping log-factor; if that is not flipped, the centre of the
    # first flipped segment on the ray within [t_flip, log 10]
    ts = np.linspace(t_flip, np.log(10.0), 400)
    signs = np.array([np.sign((np.exp(-2.0 * t * x) * a).sum()) for t in ts])
    t_use = 1.25 * t_flip
    if np.sign((np.exp(-2.0 * t_use * x) * a).sum()) == s0:
        fl = np.where(signs != s0)[0]
        segs = np.split(fl, np.where(np.diff(fl) > 1)[0] + 1)
        t_use = float(np.mean(ts[segs[0]])) if fl.size else t_flip * (1 + 1e-3)
    c = np.exp(t_use * x)
    K = total_steps - 1 - T
    rec = {"seed": seed, "rho": rho, "step": T, "t_flip": float(t_flip), "factor_flip": float(np.exp(t_flip)),
           "factor_used": float(np.exp(t_use)), "c": c.tolist(), "ips": [float(v) for v in ips]}
    for label, cc in (("c1", np.ones(DEPTH)), ("cstar", c)):
        etas = [eta * cl ** 2 for cl in cc]
        Wc = [cl * w for cl, w in zip(cc, Wk)]; Hc = [h.copy() for h in Hk]     # exact conjugate iterate (rescaling)
        _, gW, gH = grads(X, Ys, Wc, Hc)
        cos, u_pc, u_sum = pcgrad_trunk(gW)
        fin_pc, n_act = pcgrad_run(X, Ys, Wc, Hc, etas, eta, K)
        fin_sum = gd_run(X, Ys, Wc, Hc, etas, eta, K)["loss"][-1]
        d = {"cos": float(cos), "active_steps": int(n_act), "final_pc": [float(v) for v in fin_pc], "final_sum": [float(v) for v in fin_sum],
             "pc_over_sum": [float(fin_pc[i] / fin_sum[i]) for i in range(2)]}
        corr = float(np.sqrt(sum(np.sum((p - q) ** 2) for p, q in zip(u_pc, u_sum))))
        d["relative_perturbation"] = corr / float(np.sqrt(sum(np.sum(q ** 2) for q in u_sum)))
        if corr > 0:
            gHsum = [gH[0][i] + gH[1][i] for i in range(2)]
            ctrl = perturbed_continuation(X, Ys, Wc, Hc, etas, eta, u_sum, gHsum, corr, K, n_ctrl, seed=10 * T + seed)
            d["ctrl_over_sum"] = [[f[i] / fin_sum[i] for i in range(2)] for f in ctrl]
            d["ctrl_over_sum_median"] = [float(np.median([f[i] / fin_sum[i] for f in ctrl])) for i in range(2)]
            delta = [p - q for p, q in zip(u_pc, u_sum)]
            Wn = [w - e * (q - dd) for w, e, q, dd in zip(Wc, etas, u_sum, delta)]
            Hn = [h - eta * g for h, g in zip(Hc, gHsum)]
            anti = gd_run(X, Ys, Wn, Hn, etas, eta, K)["loss"][-1]
            d["anti_over_sum"] = [float(anti[i] / fin_sum[i]) for i in range(2)]
        rec[label] = d
    return rec


def summarize_continuation(recs):
    seeds = np.array([r["seed"] for r in recs])
    def arr(key, sub):
        return np.array([r[key].get(sub, [np.nan, np.nan]) for r in recs], float)
    pc = arr("cstar", "pc_over_sum"); ct = arr("cstar", "ctrl_over_sum_median"); an = arr("cstar", "anti_over_sum"); pc1 = arr("c1", "pc_over_sum")
    ctall = np.array([np.array(r["cstar"]["ctrl_over_sum"]) for r in recs if "ctrl_over_sum" in r["cstar"]])  # (cells, draws, 2)
    act1 = np.array([r["c1"]["active_steps"] for r in recs]); actc = np.array([r["cstar"]["active_steps"] for r in recs])
    cosc = np.array([r["cstar"]["cos"] for r in recs])
    strict = act1 == 0
    out = {"n_cells": len(recs), "n_seeds": int(np.unique(seeds).size), "by_step": {str(k): int(sum(1 for r in recs if r["step"] == k)) for k in CKPTS[:-1]},
           "factor_flip_median": float(np.median([r["factor_flip"] for r in recs])), "factor_used_median": float(np.median([r["factor_used"] for r in recs])),
           "factor_used_max": float(np.max([r["factor_used"] for r in recs])),
           "n_cos_negative_at_cstar": int((cosc < 0).sum()), "n_pc_active_somewhere_at_cstar": int((actc > 0).sum()),
           "n_strictly_switched_on (pc inactive on every step at c=1)": int(strict.sum()),
           "active_steps_at_cstar_median": float(np.median(actc)), "active_steps_at_c1_median": float(np.median(act1)),
           "relative_perturbation_at_cstar_median": float(np.median([r["cstar"]["relative_perturbation"] for r in recs])),
           "relative_perturbation_at_cstar_max": float(np.max([r["cstar"]["relative_perturbation"] for r in recs]))}
    for name, v in (("pcgrad_over_sum_at_cstar", pc), ("kick_median_over_sum_at_cstar", ct), ("anti_over_sum_at_cstar", an), ("pcgrad_over_sum_at_c1", pc1)):
        out[name] = {"task1": boot_ci(v[:, 0], seeds), "task2": boot_ci(v[:, 1], seeds),
                     "task1_mean": boot_ci(v[:, 0], seeds, np.mean), "task2_mean": boot_ci(v[:, 1], seeds, np.mean),
                     "min": [float(np.nanmin(v[:, 0])), float(np.nanmin(v[:, 1]))], "max": [float(np.nanmax(v[:, 0])), float(np.nanmax(v[:, 1]))],
                     "q10_q90_task1": [float(np.nanpercentile(v[:, 0], 10)), float(np.nanpercentile(v[:, 0], 90))],
                     "q10_q90_task2": [float(np.nanpercentile(v[:, 1], 10)), float(np.nanpercentile(v[:, 1], 90))]}
    if ctall.size:
        out["kick_all_draws_over_sum_at_cstar"] = {"task1": boot_ci(ctall[..., 0].ravel(), np.repeat(seeds[[i for i, r in enumerate(recs) if "ctrl_over_sum" in r["cstar"]]], ctall.shape[1])),
                                                    "task2": boot_ci(ctall[..., 1].ravel(), np.repeat(seeds[[i for i, r in enumerate(recs) if "ctrl_over_sum" in r["cstar"]]], ctall.shape[1]))}
    diff = pc - ct
    out["paired_pcgrad_minus_kick_at_cstar"] = {"task1": boot_ci(diff[:, 0], seeds), "task2": boot_ci(diff[:, 1], seeds),
                                                "task1_mean": boot_ci(diff[:, 0], seeds, np.mean), "task2_mean": boot_ci(diff[:, 1], seeds, np.mean),
                                                "frac_pcgrad_below_kick_task1": boot_ci((diff[:, 0] < 0).astype(float), seeds, np.mean),
                                                "frac_pcgrad_below_kick_task2": boot_ci((diff[:, 1] < 0).astype(float), seeds, np.mean)}
    diffa = pc - an
    out["paired_pcgrad_minus_anti_at_cstar"] = {"task1": boot_ci(diffa[:, 0], seeds), "task2": boot_ci(diffa[:, 1], seeds),
                                                "frac_pcgrad_below_anti_task1": boot_ci((diffa[:, 0] < 0).astype(float), seeds, np.mean),
                                                "frac_pcgrad_below_anti_task2": boot_ci((diffa[:, 1] < 0).astype(float), seeds, np.mean)}
    gm = np.sqrt(pc[:, 0] * pc[:, 1]); gmk = np.sqrt(ct[:, 0] * ct[:, 1])
    out["geomean_ratio"] = {"pcgrad": boot_ci(gm, seeds), "kick": boot_ci(gmk, seeds), "paired_diff": boot_ci(gm - gmk, seeds),
                            "frac_pcgrad_below_1": boot_ci((gm < 1).astype(float), seeds, np.mean), "frac_kick_below_1": boot_ci((gmk < 1).astype(float), seeds, np.mean)}
    if strict.any():
        out["strict_subset"] = {"n": int(strict.sum()), "pcgrad_task1": boot_ci(pc[strict, 0], seeds[strict]), "pcgrad_task2": boot_ci(pc[strict, 1], seeds[strict]),
                                "kick_task1": boot_ci(ct[strict, 0], seeds[strict]), "kick_task2": boot_ci(ct[strict, 1], seeds[strict])}
    for k in CKPTS[:-1]:
        m = np.array([r["step"] == k for r in recs])
        if m.any():
            out.setdefault("by_step_ratios", {})[str(k)] = {"n": int(m.sum()), "pcgrad_task1": boot_ci(pc[m, 0], seeds[m]), "pcgrad_task2": boot_ci(pc[m, 1], seeds[m]),
                                                             "kick_task1": boot_ci(ct[m, 0], seeds[m]), "kick_task2": boot_ci(ct[m, 1], seeds[m])}
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=300)
    ap.add_argument("--n_seeds_slow", type=int, default=100)
    ap.add_argument("--n_jobs", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=15)
    ap.add_argument("--max_cont", type=int, default=240, help="cap on continuation cells (taken in seed order, all eligible cells of each seed)")
    ap.add_argument("--skip_from0", action="store_true")
    ap.add_argument("--scratch", default=os.environ.get("METHODS_SCALE_SCRATCH", HERE))
    args = ap.parse_args()
    from joblib import Parallel, delayed
    t0 = time.time()
    global SUB
    steps = 400
    SUB = np.array(sorted(set(range(0, steps, 4)) | set(range(steps - 8, steps))))
    summary = {"config": {"depth": DEPTH, "width": WIDTH, "d_in": D_IN, "n": N, "rhos": RHOS, "ckpts": CKPTS, "methods": METHODS,
                          "seed_streams": "numpy default_rng(1000 + s), s = 10..10+n_seeds-1 (page used s = 0..9); inputs/teachers/W_0 shared across rho within a seed",
                          "n_jobs": args.n_jobs, "cagrad_normalisation": "ca<c>: 2 d/(1+c) (step-matched; = summed update when g1 = g2); ca<c>_off: 2 d/(1+c^2) (official rescale=1, sum-scaled)",
                          "bootstrap": "cluster bootstrap over seeds, 2000 resamples, percentile 95% CI"}, "timing_s": {}}
    npz = {}

    def run_block(name, seeds, eta):
        parts = []
        for k in range(0, len(seeds), args.chunk):
            chunk = seeds[k:k + args.chunk]
            cache = os.path.join(args.scratch, f"ms_cache_{name}_{chunk[0]}_{chunk[-1]}.npz")
            tc = time.time()
            if os.path.exists(cache):
                Z = np.load(cache)
                parts.append({k2: Z[k2] for k2 in Z.files})
                print(f"[{name}] seeds {chunk[0]}..{chunk[-1]} loaded from cache", flush=True)
                continue
            results = {}
            for s, out in Parallel(n_jobs=args.n_jobs)(delayed(run_seed)(s, eta, steps) for s in chunk):
                results[s] = out
            P = collect(results, chunk, METHODS, steps, SUB)
            np.savez_compressed(cache, **P)
            parts.append(P)
            print(f"[{name}] seeds {chunk[0]}..{chunk[-1]} done in {time.time() - tc:.0f}s; elapsed {time.time() - t0:.0f}s", flush=True)
        return {k2: np.concatenate([p[k2] for p in parts], axis=0) for k2 in parts[0]}

    blocks = []
    if not args.skip_from0:
        blocks = [("eta0.05_400", list(range(10, 10 + args.n_seeds)), 0.05), ("eta0.01_400", list(range(10, 10 + args.n_seeds_slow)), 0.01)]
    summary["from_step0"] = {}
    P05 = None
    for name, seeds, eta in blocks:
        tb = time.time()
        P = run_block(name, seeds, eta)
        summary["timing_s"][name] = time.time() - tb
        for key in ("final", "last10"):
            summary["from_step0"].setdefault(name, {})[key] = summarize_from0(P, seeds, METHODS, key)
        summary["from_step0"][name]["n_seeds"] = len(seeds); summary["from_step0"][name]["eta"] = eta
        # census statistics of the summed run (mixed / conflict rate at checkpoints), for the record
        mi = METHODS.index("sum")
        A = P["ck_a"][:, :, mi]                     # (S, R, 5, L)
        mixed = (np.sign(A).max(-1) != np.sign(A).min(-1)) & np.all(np.isfinite(A), -1)
        Ssum = A.sum(-1)
        summary["from_step0"][name]["summed_run_census"] = {
            "n_cells": int(np.isfinite(Ssum).sum()), "mixed_frac": float(mixed[np.isfinite(Ssum)].mean()),
            "mixed_frac_by_step": {str(k): float(mixed[:, :, i][np.isfinite(Ssum[:, :, i])].mean()) for i, k in enumerate(CKPTS)},
            "conflict_rate": float((Ssum[np.isfinite(Ssum)] < 0).mean()),
            "conflict_by_step": {str(k): float((Ssum[:, :, i][np.isfinite(Ssum[:, :, i])] < 0).mean()) for i, k in enumerate(CKPTS)},
            "conflict_by_rho": {str(r): float((Ssum[:, ri][np.isfinite(Ssum[:, ri])] < 0).mean()) for ri, r in enumerate(RHOS)}}
        for k, v in P.items():
            npz[f"{name}/{k}"] = v
        npz[f"{name}/seeds"] = np.array(seeds)
        if name == "eta0.05_400":
            P05 = P
        with open(os.path.join(HERE, "methods_scale.json"), "w") as f:
            json.dump(summary, f, indent=1)
        print(f"[{name}] block done, {time.time() - tb:.0f}s", flush=True)

    # ---- part 2b: gauge coordinates
    tb = time.time()
    summary["gauge_coordinates"] = gauge_coordinate_analysis()
    summary["timing_s"]["gauge_coordinates"] = time.time() - tb
    with open(os.path.join(HERE, "methods_scale.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"gauge coordinates done, {time.time() - tb:.0f}s", flush=True)

    # ---- part 3: continuation at scale on the eta = 0.05 census cells flipped within factor 2
    tb = time.time()
    if P05 is None:
        cache = os.path.join(HERE, "methods_scale_cells.npz")
        Z = np.load(cache); A = Z["eta0.05_400/ck_a"][:, :, METHODS.index("sum")]; seeds05 = Z["eta0.05_400/seeds"].tolist()
    else:
        A = P05["ck_a"][:, :, METHODS.index("sum")]; seeds05 = list(range(10, 10 + args.n_seeds))
    jobs = []
    n_mixed_eligible = 0
    for si, s in enumerate(seeds05):
        for ri, rho in enumerate(RHOS):
            for ki, T in enumerate(CKPTS[:-1]):
                a = A[si, ri, ki]
                if not np.all(np.isfinite(a)):
                    continue
                sg = np.sign(a)
                if sg.max() == sg.min():
                    continue
                n_mixed_eligible += 1
                t_flip, _ = nearest_flip_prod1(a)
                if np.exp(t_flip) <= 2.0 + 1e-12:
                    jobs.append((s, rho, T, a))
    n_jobs_all = len(jobs)
    jobs = jobs[:args.max_cont]
    print(f"continuation: {n_jobs_all} cells flipped within 2 among {n_mixed_eligible} mixed cells at steps 0..299; running {len(jobs)}", flush=True)
    recs = []
    for k in range(0, len(jobs), 30):
        chunk = jobs[k:k + 30]
        recs.extend(Parallel(n_jobs=args.n_jobs)(delayed(continuation_cell)(*j) for j in chunk))
        print(f"  continuation cells {k + len(chunk)}/{len(jobs)}; elapsed {time.time() - t0:.0f}s", flush=True)
    summary["continuation"] = summarize_continuation(recs) if recs else {"n_cells": 0}
    summary["continuation"]["n_mixed_eligible_cells_steps0_299"] = n_mixed_eligible
    summary["continuation"]["n_cells_flipped_within_2_total"] = n_jobs_all
    summary["continuation"]["n_cells_run"] = len(jobs)
    summary["continuation"]["records"] = recs
    summary["timing_s"]["continuation"] = time.time() - tb
    summary["timing_s"]["total"] = time.time() - t0
    with open(os.path.join(HERE, "methods_scale.json"), "w") as f:
        json.dump(summary, f, indent=1)
    npz["sub_steps"] = SUB
    np.savez_compressed(os.path.join(HERE, "methods_scale_cells.npz"), **npz)
    print(f"all done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
