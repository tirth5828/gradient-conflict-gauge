"""Gauge dependence of gradient conflict: every number in the abstract.

Bias-free ReLU trunk (depth L, width W) with two linear heads, two synthetic
regression tasks, full-batch gradient descent, float64 throughout.

  1. Exact conjugation: run A (W_l, eta) vs run B (c_l W_l, c_l^2 eta), prod c_l = 1.
     Also the head-absorbed form with prod c_l != 1 (H -> H/P, eta_head -> eta_head/P^2).
  2. Head cross-term is exactly zero (disjoint support).
  3. Sign census: seeds x correlations x checkpoints, sign pattern of the L layerwise
     inner products a_l = <g_1^l, g_2^l>; per-cell a_l, layer Frobenius norms persisted.
  4. Gauge sweep along (c, 1/c, 1, 1): the conjugate run is RETRAINED for every c from
     c_l W_0 with c_l^2 eta; losses, trunk cosine, eta-weighted invariant, PCGrad one-step
     loss effect (at eta, eta/10, eta/100 + first-order prediction), eta-metric PCGrad
     (equivariant), PCGrad continued from T to step 399 vs summed GD (final losses), a
     random-kick control of the same norm on the same step, directed controls (PCGrad's
     correction with the opposite sign; kicks along +/- u_sum), plus
     eta*lambda_max of the summed-loss Hessian at T and at the last step (edge-of-stability check).
     The figure's right panel is the census-wide CDF of the nearest flipping factor (over the
     whole product-one per-layer group, largest per-layer factor; two-layer and single-layer
     values are persisted too).
  5. Reachability: signs of sum_l c_l^-2 a_l at gauges that ordinary conventions induce
     (init profiles incl. the measured orthogonal/He profile, input scale, layerwise
     step-size decay read as c_l = sqrt(eta_l/eta)), random per-layer factors (+ |cos| on
     either side of a flip = PCGrad's correction, + the closed-form two-task MGDA weight),
     and the nearest sign-flipping gauge per mixed cell.
  6. PCGrad continuation on every census cell that a gauge in [0.1,10] flips (c* = centre of
     the flipped region nearest c = 1) vs summed GD, with the random-kick control.
  The census also records eta*lambda_max (power iteration on a finite-difference HVP,
  eps = 1e-6, inside one ReLU activation pattern) at every checkpoint.

Run:  python conjugation_check.py   (~12 min; writes results.json and ../figures/gauge_sweep.{png,pdf})
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "..", "figures")


# ----------------------------------------------------------------------------- model
def init_net(rng, d_in, width, depth, d_out=1, scheme="he"):
    """Trunk W[0..depth-1] (bias-free) and two heads H[0], H[1]."""
    W = []
    fan_in = d_in
    for _ in range(depth):
        if scheme == "he":
            W.append(rng.standard_normal((fan_in, width)) * np.sqrt(2.0 / fan_in))
        elif scheme == "lecun":
            W.append(rng.standard_normal((fan_in, width)) * np.sqrt(1.0 / fan_in))
        elif scheme == "orthogonal":                   # torch.nn.init.orthogonal_, gain 1
            A = rng.standard_normal((max(fan_in, width), min(fan_in, width)))
            Q, R = np.linalg.qr(A)
            Q = Q * np.sign(np.diag(R))
            W.append(Q if fan_in >= width else Q.T)
        else:
            raise ValueError(scheme)
        fan_in = width
    H = [rng.standard_normal((width, d_out)) * np.sqrt(1.0 / width) for _ in range(2)]
    return W, H


def forward(X, W, H):
    hs = [X]
    h = X
    for Wl in W:
        h = np.maximum(h @ Wl, 0.0)
        hs.append(h)
    outs = [h @ Hi for Hi in H]
    return hs, outs


def task_loss(out, Y):
    return 0.5 * np.mean((out - Y) ** 2)


def grads(X, Ys, W, H):
    """Per-task gradients. Returns (losses, gW[task][layer], gH[task][head])."""
    hs, outs = forward(X, W, H)
    n = X.shape[0]
    losses, gW, gH = [], [], []
    for i in range(2):
        d_out = (outs[i] - Ys[i]) / n                      # dL_i / d out_i
        gHi = [np.zeros_like(H[0]), np.zeros_like(H[1])]
        gHi[i] = hs[-1].T @ d_out
        delta = d_out @ H[i].T                            # dL_i / d h_L
        gWi = [None] * len(W)
        for l in range(len(W) - 1, -1, -1):
            delta = delta * (hs[l + 1] > 0)               # through ReLU
            gWi[l] = hs[l].T @ delta
            delta = delta @ W[l].T
        losses.append(task_loss(outs[i], Ys[i]))
        gW.append(gWi)
        gH.append(gHi)
    return losses, gW, gH


def gd_run(X, Ys, W, H, eta_layers, eta_head, steps, record_every=None):
    """Full-batch GD on the summed loss with per-layer step sizes. Returns history."""
    W = [w.copy() for w in W]
    H = [h.copy() for h in H]
    hist = {"loss": [], "W": [], "H": []}
    for t in range(steps):
        losses, gW, gH = grads(X, Ys, W, H)
        hist["loss"].append(losses)
        if record_every and t % record_every == 0:
            hist["W"].append([w.copy() for w in W])
            hist["H"].append([h.copy() for h in H])
        for l in range(len(W)):
            W[l] = W[l] - eta_layers[l] * (gW[0][l] + gW[1][l])
        for i in range(2):
            H[i] = H[i] - eta_head * (gH[0][i] + gH[1][i])
    losses, _, _ = grads(X, Ys, W, H)
    hist["loss"].append(losses)
    hist["final_W"], hist["final_H"] = W, H
    return hist


# ----------------------------------------------------------------------------- data
def make_tasks(rng, n, d_in, rho, teacher_width=32):
    """Two regression targets from two random ReLU teachers, mixed so that the
    second target correlates with the first at level rho."""
    X = rng.standard_normal((n, d_in))

    def teacher():
        A = rng.standard_normal((d_in, teacher_width)) * np.sqrt(2.0 / d_in)
        b = rng.standard_normal((teacher_width, 1)) * np.sqrt(1.0 / teacher_width)
        y = np.maximum(X @ A, 0) @ b
        return (y - y.mean()) / y.std()

    f1, f2 = teacher(), teacher()
    Y1 = f1
    Y2 = rho * f1 + np.sqrt(max(0.0, 1 - rho ** 2)) * f2
    return X, [Y1, Y2]


# ----------------------------------------------------------------------------- checks
def check_conjugation(depth=4, width=64, d_in=16, n=256, eta=0.05, steps=50, c=None, seed=0):
    """Run A (W, eta) vs run B (c_l W_l, c_l^2 eta). If prod c != 1 the heads absorb it:
    H -> H / P, eta_head -> eta_head / P^2, which is exact by the same homogeneity."""
    rng = np.random.default_rng(seed)
    X, Ys = make_tasks(rng, n, d_in, rho=0.3)
    W, H = init_net(rng, d_in, width, depth)
    if c is None:
        c = np.ones(depth); c[0], c[1] = 3.0, 1.0 / 3.0
    c = np.asarray(c, dtype=float)
    P = float(np.prod(c))
    A = gd_run(X, Ys, W, H, [eta] * depth, eta, steps)
    B = gd_run(X, Ys, [cl * w for cl, w in zip(c, W)], [h / P for h in H],
               [eta * cl ** 2 for cl in c], eta / P ** 2, steps)
    lossA, lossB = np.array(A["loss"]), np.array(B["loss"])
    rel = np.max(np.abs(lossA - lossB) / np.abs(lossA))
    dev = max(np.max(np.abs(wb - cl * wa)) for cl, wa, wb in zip(c, A["final_W"], B["final_W"]))
    devH = max(np.max(np.abs(hb - ha / P)) for ha, hb in zip(A["final_H"], B["final_H"]))
    return {
        "c": c.tolist(), "prod_c": P,
        "final_loss_A": [f"{v:.14f}" for v in lossA[-1]],
        "final_loss_B": [f"{v:.14f}" for v in lossB[-1]],
        "max_rel_loss_diff_all_steps": float(rel),
        "max_abs_weight_deviation": float(dev),
        "max_abs_head_deviation": float(devH),
    }


def layer_inner_products(X, Ys, W, H):
    _, gW, gH = grads(X, Ys, W, H)
    ips = [float(np.sum(gW[0][l] * gW[1][l])) for l in range(len(W))]
    head_cross = float(sum(np.sum(gH[0][i] * gH[1][i]) for i in range(2)))
    return ips, head_cross, gW, gH


def trunk_cosine(gW):
    g1 = np.concatenate([g.ravel() for g in gW[0]]); g2 = np.concatenate([g.ravel() for g in gW[1]])
    return float(g1 @ g2 / (np.linalg.norm(g1) * np.linalg.norm(g2)))


def sign_census(depth=4, width=64, d_in=16, n=256, eta=0.05, steps=400,
                seeds=range(5), rhos=(-0.5, -0.2, 0.0, 0.3, 0.7), n_ckpt=5, quiet=False,
                scheme="he", init_scale=None, eta_layers=None, sharpness=False, indep_streams=False):
    """One row per (seed, rho, checkpoint): layerwise a_l = <g1^l, g2^l>, sign pattern,
    trunk cosine at c = 1, per-layer Frobenius norms, per-layer squared gradient norms,
    losses. `scheme`, `init_scale` (per-layer multipliers on W_0) and `eta_layers`
    (per-layer step sizes, default uniform eta) are for censuses trained directly under
    those conventions. Diverged cells are counted, not silently dropped.
    `indep_streams=True` draws an independent (inputs, teachers, W_0) stream per (seed, rho)
    instead of sharing them across rho (a check of the paired design)."""
    ckpts = np.linspace(0, steps - 1, n_ckpt).astype(int)
    etas = [float(e) for e in eta_layers] if eta_layers is not None else [eta] * depth
    mixed, total, rows, head_max, diverged = 0, 0, [], 0.0, 0
    for s in seeds:
        for ri, rho in enumerate(rhos):
            # default: inputs, teachers, init shared across rho (paired design; cells within a seed are correlated)
            rng = np.random.default_rng(100000 + 100 * s + ri) if indep_streams else np.random.default_rng(1000 + s)
            X, Ys = make_tasks(rng, n, d_in, rho)
            W, H = init_net(rng, d_in, width, depth, scheme=scheme)
            if init_scale is not None:
                W = [float(sc) * w for sc, w in zip(init_scale, W)]
            hist = gd_run(X, Ys, W, H, etas, eta, steps, record_every=1)
            for k in ckpts:
                ips, hc, gW, _ = layer_inner_products(X, Ys, hist["W"][k], hist["H"][k])
                if not (np.all(np.isfinite(ips)) and np.all(np.isfinite(hist["loss"][k]))):
                    diverged += 1
                    continue
                head_max = max(head_max, abs(hc))
                signs = np.sign(ips)
                is_mixed = bool(signs.max() != signs.min())
                mixed += is_mixed
                total += 1
                # eta * lambda_max of the summed-loss Hessian: GD is stable below 2. With per-layer
                # step sizes the relevant matrix is diag(eta_l) H, whose spectrum is gauge-invariant;
                # here it is reported for the uniform-eta censuses only.
                sharp = float(eta * hessian_top_eig(X, Ys, hist["W"][k], hist["H"][k])) if (sharpness and eta_layers is None) else None
                sharp_s = float(eta * hessian_top_eig(X, Ys, hist["W"][k], hist["H"][k], eps=1e-3)) if (sharpness and eta_layers is None) else None
                rows.append({"seed": s, "rho": rho, "step": int(k), "ips": ips, "eta_lambda_max": sharp, "eta_lambda_max_kink_smoothed_eps1e-3": sharp_s,
                             "signs": signs.astype(int).tolist(), "mixed": is_mixed,
                             "cos_trunk": trunk_cosine(gW),
                             "layer_fro": [float(np.linalg.norm(w)) for w in hist["W"][k]],
                             "g1n2": [float(np.sum(g ** 2)) for g in gW[0]],
                             "g2n2": [float(np.sum(g ** 2)) for g in gW[1]],
                             "etas": etas,
                             "loss": [float(v) for v in hist["loss"][k]]})
    by_step = {}
    for r in rows:
        by_step.setdefault(r["step"], []).append(r["mixed"])
    by_rho = {}
    for r in rows:
        by_rho.setdefault(r["rho"], []).append(r["mixed"])
    S = np.array([sum(r["ips"]) for r in rows]) if rows else np.zeros(0)
    st = np.array([r["step"] for r in rows]) if rows else np.zeros(0, int)
    rh = np.array([r["rho"] for r in rows]) if rows else np.zeros(0)
    sh = np.array([r["eta_lambda_max"] for r in rows], float) if (rows and sharpness and eta_layers is None) else None
    cosv = np.array([r["cos_trunk"] for r in rows]) if rows else np.zeros(0)
    return {"mixed": mixed, "total": total, "frac": mixed / max(total, 1), "diverged": diverged,
            "max_abs_head_cross": head_max, "etas": etas,
            "mixed_by_step": {int(k): int(sum(v)) for k, v in by_step.items()},
            "mixed_by_rho": {str(k): int(sum(v)) for k, v in by_rho.items()},
            "conflict_rate": float((S < 0).mean()) if rows else None,
            "conflict_by_step": {int(k): float((S[st == k] < 0).mean()) for k in np.unique(st)},
            "conflict_by_rho": {str(k): float((S[rh == k] < 0).mean()) for k in sorted(set(rh.tolist()))},
            "conflict_by_rho_post_init": {str(k): float((S[(rh == k) & (st > 0)] < 0).mean()) for k in sorted(set(rh.tolist()))},
            "conflict_rate_post_init": float((S[st > 0] < 0).mean()) if rows else None,
            # the cosine's size by rho (its sign is the conflict rate above)
            "cos_trunk_median_by_rho_post_init": {str(k): float(np.median(cosv[(rh == k) & (st > 0)])) for k in sorted(set(rh.tolist()))},
            "cos_trunk_median_by_rho_all": {str(k): float(np.median(cosv[rh == k])) for k in sorted(set(rh.tolist()))},
            "eta_lambda_max_by_step": ({int(k): {"median": float(np.median(sh[st == k])), "min": float(sh[st == k].min()),
                                                 "max": float(sh[st == k].max())} for k in np.unique(st)} if sh is not None else None),
            "eta_lambda_max_frac_above_2_post_init": (float((sh[st > 0] > 2).mean()) if sh is not None else None),
            "rows": rows}


def pcgrad_trunk(gW):
    """Symmetric two-task PCGrad on the trunk gradients (as the methods do: shared params only).
    Returns (cos, u_pc as per-layer list, u_sum as per-layer list)."""
    shapes = [g.shape for g in gW[0]]
    g1 = np.concatenate([g.ravel() for g in gW[0]]); g2 = np.concatenate([g.ravel() for g in gW[1]])
    cos = float(g1 @ g2 / (np.linalg.norm(g1) * np.linalg.norm(g2)))
    p1, p2 = g1.copy(), g2.copy()
    if g1 @ g2 < 0:
        p1 = g1 - (g1 @ g2) / (g2 @ g2) * g2
        p2 = g2 - (g2 @ g1) / (g1 @ g1) * g1
    def split(v):
        out, k = [], 0
        for sh in shapes:
            m = int(np.prod(sh)); out.append(v[k:k + m].reshape(sh)); k += m
        return out
    return cos, split(p1 + p2), split(g1 + g2)


def ip_eta(a, b, etas):
    """Step-size inner product <a,b>_eta = sum_l eta_l <a_l, b_l> (gauge-invariant)."""
    return float(sum(e * np.sum(x * y) for e, x, y in zip(etas, a, b)))


def pcgrad_trunk_metric(gW, etas):
    """Two-task PCGrad with the projection taken in <.,.>_eta. Equivariant under the gauge
    (eta_l c_l^2 * c_l^-2 = eta_l); coincides with Euclidean PCGrad when eta_l is uniform.
    Returns (cos_eta, u as per-layer list)."""
    g1, g2 = gW[0], gW[1]
    n12, n11, n22 = ip_eta(g1, g2, etas), ip_eta(g1, g1, etas), ip_eta(g2, g2, etas)
    cos = n12 / np.sqrt(n11 * n22)
    if n12 < 0:
        p1 = [a - n12 / n22 * b for a, b in zip(g1, g2)]
        p2 = [b - n12 / n11 * a for a, b in zip(g1, g2)]
    else:
        p1, p2 = g1, g2
    return float(cos), [a + b for a, b in zip(p1, p2)]


def summed_grad_vec(X, Ys, W, H):
    _, gW, gH = grads(X, Ys, W, H)
    return np.concatenate([(gW[0][l] + gW[1][l]).ravel() for l in range(len(W))]
                          + [(gH[0][i] + gH[1][i]).ravel() for i in range(2)])


def hessian_top_eig(X, Ys, W, H, iters=40, eps=1e-6, seed=0):
    """Largest eigenvalue of the full-parameter Hessian (trunk + heads) of the summed loss,
    by power iteration on a central finite-difference Hessian-vector product. eps = 1e-6 stays
    inside one ReLU activation pattern (the local Hessian; 1e-7 agrees to 1e-9); eps = 1e-5..1e-4
    straddles kinks and inflates the value (12 vs 1.9 on one checkpoint), so is not used."""
    shapes = [w.shape for w in W] + [h.shape for h in H]
    theta = np.concatenate([w.ravel() for w in W] + [h.ravel() for h in H])

    def unpack(v):
        out, k = [], 0
        for sh in shapes:
            m = int(np.prod(sh)); out.append(v[k:k + m].reshape(sh)); k += m
        return out[:len(W)], out[len(W):]

    def hvp(v):
        Wp, Hp = unpack(theta + eps * v); Wm, Hm = unpack(theta - eps * v)
        return (summed_grad_vec(X, Ys, Wp, Hp) - summed_grad_vec(X, Ys, Wm, Hm)) / (2 * eps)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(theta.size); v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        hv = hvp(v)
        lam = float(v @ hv)
        nrm = np.linalg.norm(hv)
        if nrm == 0:
            break
        v = hv / nrm
    return lam


def perturbed_continuation(X, Ys, W, H, eta_layers, eta_head, u_sum, gHsum, norm, steps, n_draws, seed):
    """Control for the PCGrad continuation: at the current step apply the summed update plus a
    random per-layer perturbation r of Euclidean norm `norm` (= ||u_PC - u_sum|| in the same
    coordinates), then continue plain summed GD for `steps` steps. Returns final losses per draw."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_draws):
        r = [rng.standard_normal(w.shape) for w in W]
        s = np.sqrt(sum(np.sum(x ** 2) for x in r))
        r = [x * (norm / s) for x in r]
        Wn = [w - e * (u + x) for w, e, u, x in zip(W, eta_layers, u_sum, r)]
        Hn = [h - eta_head * g for h, g in zip(H, gHsum)]
        out.append([float(v) for v in gd_run(X, Ys, Wn, Hn, eta_layers, eta_head, steps)["loss"][-1]])
    return out


def pcgrad_run(X, Ys, W, H, eta_layers, eta_head, steps):
    """Euclidean PCGrad on the trunk (heads take the summed step) for `steps` steps.
    Returns (final losses, number of steps on which the projection was active)."""
    W = [w.copy() for w in W]
    H = [h.copy() for h in H]
    active = 0
    for _ in range(steps):
        _, gW, gH = grads(X, Ys, W, H)
        cos, u_pc, _ = pcgrad_trunk(gW)
        active += cos < 0
        for l in range(len(W)):
            W[l] = W[l] - eta_layers[l] * u_pc[l]
        for i in range(2):
            H[i] = H[i] - eta_head * (gH[0][i] + gH[1][i])
    losses, _, _ = grads(X, Ys, W, H)
    return losses, int(active)


def gauge_sweep(row, depth=4, width=64, d_in=16, n=256, eta=0.05,
                cs=np.logspace(-1, 1, 81), total_steps=400, n_ctrl=6, n_ctrl_dense=20,
                ctrl_dense_at=(0.1, 0.126, 0.42, 1.88, 10.0)):
    """Sweep c along (c, 1/c, 1, 1). For every c the conjugate run is retrained from
    c_l W_0 with step sizes c_l^2 eta up to the checkpoint step T, and everything is
    evaluated on that run's own iterate (no rescaling of the reference checkpoint).
    Also per c: one PCGrad step vs one summed step (at eta, eta/10, eta/100, with the
    first-order prediction), one step of eta-metric PCGrad, and the continuation of
    PCGrad vs summed GD from T to total_steps - 1 (final per-task losses)."""
    T = row["step"]
    K = total_steps - 1 - T
    rng = np.random.default_rng(1000 + row["seed"])
    X, Ys = make_tasks(rng, n, d_in, row["rho"])
    W0, H0 = init_net(rng, d_in, width, depth)
    ref = gd_run(X, Ys, W0, H0, [eta] * depth, eta, T + 1, record_every=1)
    Wk, Hk = ref["W"][T], ref["H"][T]
    ips, hc, gWk, _ = layer_inner_products(X, Ys, Wk, Hk)
    out = {"T": T, "K_continuation": K, "c": [], "loss1": [], "loss2": [], "cos": [], "pcgrad_ratio": [],
           "eta_weighted_ip": [], "layer_terms": [], "conj_resid": [],
           "dL1_pc_minus_sum": [], "dL2_pc_minus_sum": [],
           "dL_pc_minus_sum_first_order": [], "dL_pc_minus_sum_eta_x0.1": [], "dL_pc_minus_sum_eta_x0.01": [],
           "cos_eta": [], "loss_after_pcgrad_eta_metric": [], "loss_after_sum_step": [],
           "final_pc": [], "final_sum": [], "pc_active_steps": [], "ips": ips,
           # control: random perturbation of the same norm as PCGrad's correction on the same step
           "ctrl_final": [], "ctrl_norm": [], "ctrl_dense": {},
           # directed control: PCGrad's correction applied with the opposite sign on the same step
           # (same direction, same norm), then summed GD; also kicks of PCGrad's norm along +/- u_sum
           "anti_final": [], "gsum_plus_final": [], "gsum_minus_final": []}
    # sharpness of the reference trajectory (c = 1): eta*lambda_max at T and at the last step
    ref_full = gd_run(X, Ys, W0, H0, [eta] * depth, eta, total_steps, record_every=1)
    out["eta_lambda_max_at_T"] = float(eta * hessian_top_eig(X, Ys, Wk, Hk))
    out["eta_lambda_max_at_end"] = float(eta * hessian_top_eig(X, Ys, ref_full["final_W"], ref_full["final_H"]))
    tot = np.array(ref_full["loss"]).sum(1)
    out["summed_loss_nonmonotone_steps_after_T"] = int((np.diff(tot[T:]) > 0).sum())
    dense_idx = {int(np.argmin(np.abs(cs - v))) for v in ctrl_dense_at}
    for ci, cval in enumerate(cs):
        c = np.ones(depth); c[0], c[1] = cval, 1.0 / cval
        etas = [eta * cl ** 2 for cl in c]
        run = gd_run(X, Ys, [cl * w for cl, w in zip(c, W0)], H0, etas, eta, T + 1, record_every=1)
        Wc, Hc = run["W"][T], run["H"][T]
        resid = max(np.max(np.abs(wc - cl * wk)) for cl, wc, wk in zip(c, Wc, Wk))
        losses, gW, gH = grads(X, Ys, Wc, Hc)             # independent gradients in gauge coords
        cos, u_pc, u_sum = pcgrad_trunk(gW)
        ratio = float(np.sqrt(sum(np.sum((a - b) ** 2) for a, b in zip(u_pc, u_sum)))
                      / np.sqrt(sum(np.sum(b ** 2) for b in u_sum)))
        # one step of each method in gauge coordinates with the compensated step sizes
        gHsum = [gH[0][i] + gH[1][i] for i in range(2)]

        def one_step(u, scale=1.0):
            Hn = [h - scale * eta * g for h, g in zip(Hc, gHsum)]
            L, _, _ = grads(X, Ys, [w - scale * e * uu for w, e, uu in zip(Wc, etas, u)], Hn)
            return L
        L_pc, L_sum = one_step(u_pc), one_step(u_sum)
        first = [-float(sum(e * np.sum(gW[i][l] * (u_pc[l] - u_sum[l])) for l, e in enumerate(etas))) for i in range(2)]
        scaled = {}
        for sc in (0.1, 0.01):
            Lp, Ls = one_step(u_pc, sc), one_step(u_sum, sc)
            scaled[sc] = [float(Lp[i] - Ls[i]) for i in range(2)]
        cos_eta, u_P = pcgrad_trunk_metric(gW, etas)
        L_P = one_step(u_P)
        fin_pc, n_act = pcgrad_run(X, Ys, Wc, Hc, etas, eta, K)
        fin_sum = gd_run(X, Ys, Wc, Hc, etas, eta, K)["loss"][-1]
        corr = float(np.sqrt(sum(np.sum((a - b) ** 2) for a, b in zip(u_pc, u_sum))))
        if corr > 0 and K > 0:
            ctrl = perturbed_continuation(X, Ys, Wc, Hc, etas, eta, u_sum, gHsum, corr, K, n_ctrl, seed=ci)
            if ci in dense_idx:
                out["ctrl_dense"][f"{cval:.4g}"] = {"pcgrad_over_sum": [float(fin_pc[i] / fin_sum[i]) for i in range(2)],
                                                    "ctrl_over_sum": [[f[i] / fin_sum[i] for i in range(2)] for f in
                                                                      perturbed_continuation(X, Ys, Wc, Hc, etas, eta, u_sum, gHsum, corr, K, n_ctrl_dense, seed=1000 + ci)],
                                                    "relative_perturbation": corr / float(np.sqrt(sum(np.sum(b ** 2) for b in u_sum))),
                                                    "active_steps": n_act}
            delta = [a - b for a, b in zip(u_pc, u_sum)]
            nsum = float(np.sqrt(sum(np.sum(b ** 2) for b in u_sum)))

            def cont(u):
                Wn = [w - e * uu for w, e, uu in zip(Wc, etas, u)]
                Hn = [h - eta * g for h, g in zip(Hc, gHsum)]
                return [float(v) for v in gd_run(X, Ys, Wn, Hn, etas, eta, K)["loss"][-1]]
            anti = cont([b - d for b, d in zip(u_sum, delta)])
            gplus = cont([b * (1 + corr / nsum) for b in u_sum]); gminus = cont([b * (1 - corr / nsum) for b in u_sum])
        else:
            ctrl = [[float(v) for v in fin_sum]] * n_ctrl
            anti = gplus = gminus = [float(v) for v in fin_sum]
        out["ctrl_final"].append(ctrl); out["ctrl_norm"].append(corr)
        out["anti_final"].append(anti); out["gsum_plus_final"].append(gplus); out["gsum_minus_final"].append(gminus)
        out["c"].append(float(cval)); out["loss1"].append(losses[0]); out["loss2"].append(losses[1])
        out["cos"].append(cos); out["pcgrad_ratio"].append(ratio); out["conj_resid"].append(float(resid))
        out["eta_weighted_ip"].append(float(sum(e * np.sum(gW[0][l] * gW[1][l]) for l, e in enumerate(etas))))
        out["layer_terms"].append([float(np.sum(gW[0][l] * gW[1][l])) for l in range(depth)])
        out["dL1_pc_minus_sum"].append(float(L_pc[0] - L_sum[0]))
        out["dL2_pc_minus_sum"].append(float(L_pc[1] - L_sum[1]))
        out["dL_pc_minus_sum_first_order"].append(first)
        out["dL_pc_minus_sum_eta_x0.1"].append(scaled[0.1]); out["dL_pc_minus_sum_eta_x0.01"].append(scaled[0.01])
        out["cos_eta"].append(cos_eta)
        out["loss_after_pcgrad_eta_metric"].append([float(v) for v in L_P])
        out["loss_after_sum_step"].append([float(v) for v in L_sum])
        out["final_pc"].append([float(v) for v in fin_pc]); out["final_sum"].append([float(v) for v in fin_sum])
        out["pc_active_steps"].append(n_act)
    return out


def zero_crossings(c, y):
    c, y = np.asarray(c), np.asarray(y)
    s = np.sign(y); out = []
    for k in np.where(s[1:] != s[:-1])[0]:
        x0, x1 = np.log(c[k]), np.log(c[k + 1]); y0, y1 = y[k], y[k + 1]
        out.append(float(np.exp(x0 - y0 * (x1 - x0) / (y1 - y0))))
    return out


# ----------------------------------------------------------------------------- reachability
def init_scale_profiles(d_in=16, width=64, depth=4, seeds=range(5)):
    """Per-layer Frobenius norms of He / LeCun / orthogonal initialisation and the
    per-layer factor each induces relative to He (global factor removed)."""
    out = {}
    for scheme in ("he", "lecun", "orthogonal"):
        fro = np.array([[np.linalg.norm(w) for w in init_net(np.random.default_rng(1000 + s), d_in, width, depth, scheme=scheme)[0]]
                        for s in seeds])
        out[scheme] = fro.mean(0).tolist()
    he = np.array(out["he"])
    for scheme in ("lecun", "orthogonal"):
        r = np.array(out[scheme]) / he
        out[scheme + "_over_he"] = r.tolist()
        out[scheme + "_over_he_prod1"] = (r / np.prod(r) ** (1 / depth)).tolist()
    return out


def mgda_alpha(A, G1, G2, c):
    """Two-task MGDA weight on task 1 in gauge c, closed form from the persisted layerwise
    inner products and squared gradient norms: alpha = clip((G2-N)/(G1+G2-2N), 0, 1)."""
    w = np.asarray(c, float) ** -2.0
    N, g1, g2 = A @ w, G1 @ w, G2 @ w
    return np.clip((g2 - N) / (g1 + g2 - 2 * N), 0.0, 1.0)


def reachability(rows, seed=0, n_draws=2000, named=True, measured_profile=None):
    A = np.array([r["ips"] for r in rows])                # (cells, L)
    L = A.shape[1]
    S = A.sum(1); sign0 = np.sign(S)
    mixed = np.array([r["mixed"] for r in rows])

    def sign_at(c):
        return np.sign(A @ (np.asarray(c, float) ** -2.0))

    # closed-form cosine in gauge c (needs per-layer gradient norms): the PCGrad correction
    # in that gauge is max(0, -cos(c)), so |cos(c)| on flipped cells bounds what PCGrad does there.
    G1 = np.array([r["g1n2"] for r in rows]); G2 = np.array([r["g2n2"] for r in rows])

    def cos_at(c):
        w = np.asarray(c, float) ** -2.0
        return (A @ w) / np.sqrt((G1 @ w) * (G2 @ w))
    cos1 = cos_at(np.ones(L)); alpha1 = mgda_alpha(A, G1, G2, np.ones(L))

    # Reading of a gauge c (Sec. III): N(c) is the Euclidean numerator of the conjugate run
    # (c_l W_0, c_l^2 eta), i.e. of a run with per-layer step sizes eta_l = c_l^2 eta. A
    # practitioner with step sizes eta_l therefore sits at c_l = sqrt(eta_l/eta): layerwise
    # decay eta_l = eta d^(L-1-l) is c_l = d^((L-1-l)/2).
    named_out = {}
    if named:
        named_g = {
            "orthogonal_vs_he_(1/2,1,1,1)": [0.5] + [1] * (L - 1),
            "input_scale_(1/4,1,1,1)": [0.25] + [1] * (L - 1),
            "input_scale_(4,1,1,1)": [4] + [1] * (L - 1),
            "input_scale_(1/2,1,1,1)": [0.5] + [1] * (L - 1),
            "input_scale_(2,1,1,1)": [2] + [1] * (L - 1),
            "lecun_vs_he_global_(1,1,1,1)": [1] * L,
        }
        if measured_profile is not None:
            named_g["orthogonal_vs_he_measured_profile"] = list(measured_profile)
        for d in (0.9, 0.8, 0.65, 0.5):
            named_g[f"step_decay_toward_input_d={d}_(eta_l=eta*d^(L-1-l))"] = [d ** ((L - 1 - l) / 2) for l in range(L)]
            named_g[f"step_growth_toward_input_d={d}_(eta_l=eta*d^-(L-1-l))"] = [d ** (-(L - 1 - l) / 2) for l in range(L)]
        for k, c in named_g.items():
            sc = sign_at(c)
            named_out[k] = {"c": [float(v) for v in c], "flips": int((sc != sign0).sum()),
                            "conflict_rate": float((sc < 0).mean()),
                            "mgda_alpha_abs_change_max": float(np.max(np.abs(mgda_alpha(A, G1, G2, c) - alpha1))),
                            "mgda_alpha_abs_change_median": float(np.median(np.abs(mgda_alpha(A, G1, G2, c) - alpha1)))}
    rng = np.random.default_rng(seed)
    rand_out = {}
    for lo, hi, key in ((1 / np.sqrt(2), np.sqrt(2), "logU[1/sqrt2,sqrt2]"), (0.5, 2.0, "logU[1/2,2]"), (0.25, 4.0, "logU[1/4,4]")):
        flips = []
        crates = []
        flipped_abs_cos, flipped_abs_cos_either, dalpha = [], [], []
        for _ in range(n_draws):
            c = np.exp(rng.uniform(np.log(lo), np.log(hi), size=L))
            sc = sign_at(c)
            fl = sc != sign0
            flips.append(int(fl.sum())); crates.append(float((sc < 0).mean()))
            dalpha.append(np.abs(mgda_alpha(A, G1, G2, c) - alpha1))
            if fl.any():
                cc = np.abs(cos_at(c)[fl])
                flipped_abs_cos.extend(cc.tolist())
                # PCGrad's correction on whichever side of the flip it is active: max of |cos| at c and at 1
                flipped_abs_cos_either.extend(np.maximum(cc, np.abs(cos1[fl])).tolist())
        fac = np.array(flipped_abs_cos); fae = np.array(flipped_abs_cos_either); da = np.concatenate(dalpha)
        rand_out[key] = {"mean_flips": float(np.mean(flips)), "max_flips": int(np.max(flips)),
                         "mean_frac": float(np.mean(flips) / len(rows)), "max_frac": float(np.max(flips) / len(rows)),
                         "conflict_rate_min": float(min(crates)), "conflict_rate_max": float(max(crates)),
                         "flipped_cells_abs_cos_max": float(fac.max()) if fac.size else 0.0,
                         "flipped_cells_abs_cos_median": float(np.median(fac)) if fac.size else 0.0,
                         "flipped_cells_abs_cos_q90": float(np.percentile(fac, 90)) if fac.size else 0.0,
                         "flipped_cells_abs_cos_either_side_max": float(fae.max()) if fae.size else 0.0,
                         "flipped_cells_abs_cos_either_side_median": float(np.median(fae)) if fae.size else 0.0,
                         "n_flipped_observations": int(fac.size),
                         "mgda_alpha_abs_change_median": float(np.median(da)), "mgda_alpha_abs_change_q90": float(np.percentile(da, 90)),
                         "mgda_alpha_abs_change_max": float(da.max()), "mgda_alpha_frac_change_above_0.1": float((da > 0.1).mean())}

    # nearest sign-flipping gauge per mixed cell on any product-one two-layer path
    # (c on layer i, 1/c on layer j): a_j u^2 + R u + a_i = 0, u = c^2.
    pair_logc, single_logc = [], []
    for a, m in zip(A, mixed):
        if not m:
            continue
        best = np.inf
        for i in range(L):
            for j in range(L):
                if i == j:
                    continue
                R = a.sum() - a[i] - a[j]
                roots = np.roots([a[j], R, a[i]])
                for u in roots:
                    if abs(u.imag) < 1e-12 and u.real > 0:
                        best = min(best, abs(np.log(u.real)) / 2)
        pair_logc.append(best)
        bs = np.inf
        Ssum = a.sum()
        for l in range(L):
            q = 1 - Ssum / a[l]
            if q > 0:
                bs = min(bs, abs(np.log(1 / q)) / 2)
        single_logc.append(bs)
    pair_c = np.exp(np.array(pair_logc)); single_c = np.exp(np.array(single_logc))

    # Nearest flip over the WHOLE per-layer group, measured by the largest per-layer factor
    # max_l |log c_l|. Without the product constraint the optimum is closed-form: every layer whose a_l
    # has the sign of the sum (the 'majority' of N) goes to c = e^t, every layer of the minority sign to
    # e^-t, and the sign flips at t* = (1/4) log(sum_maj |a_l| / sum_min |a_l|). With prod c_l = 1 (the
    # page's convention, sum_l log c_l = 0) the optimum is found by bisection on t over a grid of the
    # (L-1)-dimensional slice of the box [-t,t]^L (a feasible-point, hence upper, bound; the two-layer
    # (c,1/c) value is a feasible point too, so the bisection is bracketed by [t*, log pair_c]).
    def all_layer_closed_form(a):
        s = np.sign(a.sum()); maj = np.abs(a[np.sign(a) == s]).sum(); mino = np.abs(a[np.sign(a) == -s]).sum()
        return 0.25 * np.log(maj / mino) if mino > 0 else np.inf

    n_g = max(9, int(round(2.3e5 ** (1.0 / (L - 1)))))
    u = np.linspace(-1.0, 1.0, n_g)
    grids = np.stack(np.meshgrid(*([u] * (L - 1)), indexing="ij"), -1).reshape(-1, L - 1)
    last = -grids.sum(1)
    feas = np.abs(last) <= 1 + 1e-12
    Xg = np.concatenate([grids[feas], last[feas, None]], 1)          # points of the slice, in units of t

    def flips_at(a, t):
        s = np.sign(a.sum())
        return bool(np.min((np.exp(-2.0 * t * Xg) * (s * a)).sum(1)) <= 0)

    all_logc, all_cf_logc = [], []
    for a, plc in zip(A[mixed], pair_logc):                     # pair_logc is over the mixed cells only
        lo = all_layer_closed_form(a); hi = plc if np.isfinite(plc) else np.log(1e3)
        all_cf_logc.append(lo)
        if flips_at(a, lo):
            all_logc.append(lo); continue
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if flips_at(a, mid):
                hi = mid
            else:
                lo = mid
        all_logc.append(hi)
    all_c = np.exp(np.array(all_logc)); all_cf_c = np.exp(np.array(all_cf_logc))

    # Supremum (over a grid of the whole box) of |cos| on either side of a flip, for factors
    # within sqrt2 and within 2: the random draws above give a sample maximum, this gives a bound
    # on PCGrad's per-step correction on flipped cells.
    box_out = {}
    n_b = max(7, int(round(4e5 ** (1.0 / L))))
    Am, G1m, G2m, cos1m, sign0m = A[mixed], G1[mixed], G2[mixed], cos1[mixed], sign0[mixed]
    for hi_f, key in ((np.sqrt(2), "box[1/sqrt2,sqrt2]"), (2.0, "box[1/2,2]")):
        ub = np.linspace(-np.log(hi_f), np.log(hi_f), n_b)
        Xb = np.stack(np.meshgrid(*([ub] * L), indexing="ij"), -1).reshape(-1, L)
        Wb = np.exp(-2.0 * Xb).T                                       # (L, P)
        Nb = Am @ Wb; cb = Nb / np.sqrt((G1m @ Wb) * (G2m @ Wb))
        fl = np.sign(Nb) != sign0m[:, None]
        either = np.maximum(np.abs(cb), np.abs(cos1m)[:, None])
        cell_max = np.where(fl.any(1), np.where(fl, either, 0.0).max(1), np.nan)
        box_out[key] = {"grid_points_per_axis": int(n_b), "n_cells_flipping_in_box": int(fl.any(1).sum()),
                        "abs_cos_either_side_sup": float(np.nanmax(cell_max)) if fl.any() else 0.0,
                        "abs_cos_either_side_per_cell_max_median": float(np.nanmedian(cell_max)) if fl.any() else 0.0,
                        "n_cells_with_max_above_0.05": int(np.nansum(cell_max > 0.05))}
    # brute-force check of the closed form on the fixed (c,1/c,1,..,1) grid for all mixed cells
    grid = np.logspace(-1, 1, 2001)
    onpath = []
    for a, m in zip(A, mixed):
        if not m:
            continue
        sg = np.sign(a[0] * grid ** -2 + a[1] * grid ** 2 + a[2:].sum())
        fl = grid[sg != np.sign(a.sum())]
        onpath.append(float(np.min(np.abs(np.log(fl)))) if fl.size else np.inf)
    onpath_c = np.exp(np.array(onpath))
    n_mixed = int(mixed.sum()); n_all = len(rows)

    def summ(cvals):
        finite = cvals[np.isfinite(cvals)]          # statistics over the cells that flip at all
        return {"median_of_finite": float(np.median(finite)), "q25_of_finite": float(np.percentile(finite, 25)),
                "q75_of_finite": float(np.percentile(finite, 75)),
                "n_within_sqrt2": int((cvals <= np.sqrt(2) + 1e-12).sum()), "n_within_2": int((cvals <= 2 + 1e-12).sum()),
                "n_within_3": int((cvals <= 3 + 1e-12).sum()), "n_within_10": int((cvals <= 10 + 1e-12).sum()),
                "n_mixed": n_mixed, "n_cells": n_all, "n_finite": int(finite.size),
                "n_infinite": int(cvals.size - finite.size)}

    fro = np.array([r["layer_fro"] for r in rows])
    steps = np.array([r["step"] for r in rows])
    ratio = fro.max(1) / fro.min(1)
    conflict_by_step = {int(s): float((sign0[steps == s] < 0).mean()) for s in np.unique(steps)}
    return {"conflict_rate_c1": float((sign0 < 0).mean()), "conflict_rate_c1_by_step": conflict_by_step,
            "mgda_alpha_c1_median": float(np.median(alpha1)), "mgda_alpha_c1_frac_at_boundary": float(((alpha1 == 0) | (alpha1 == 1)).mean()),
            "mixed_per_seed": {int(s): int(sum(1 for r in rows if r["seed"] == s and r["mixed"])) for s in sorted({r["seed"] for r in rows})},
            "mixed_post_init": int(sum(1 for r in rows if r["mixed"] and r["step"] > 0)),
            "layer_fro_ratio_max_over_min_step0_max": float(ratio[steps == 0].max()),
            "layer_fro_ratio_max_over_min_after_step0_max": float(ratio[steps > 0].max()),
            "named_gauges": named_out, "random_gauges": rand_out,
            "nearest_flip_best_pair": summ(pair_c), "nearest_flip_single_layer": summ(single_c),
            "nearest_flip_all_layers_prod1": summ(all_c), "nearest_flip_all_layers_closed_form_unconstrained": summ(all_cf_c),
            "nearest_flip_all_layers_prod1_values": [float(v) if np.isfinite(v) else None for v in all_c],
            "nearest_flip_all_layers_slice_grid_points_per_axis": int(n_g),
            "abs_cos_either_side_box_supremum": box_out,
            "nearest_flip_on_path_c_1c_1_1": summ(onpath_c),
            "nearest_flip_best_pair_values": [float(v) if np.isfinite(v) else None for v in pair_c],
            "abs_cos_trunk_median_mixed": float(np.median(np.abs([r["cos_trunk"] for r in rows if r["mixed"]]))),
            "abs_cos_trunk_median_single_sign": float(np.median(np.abs([r["cos_trunk"] for r in rows if not r["mixed"]]))),
            "layer_fro_ratio_max_over_min_median": float(np.median(fro.max(1) / fro.min(1))),
            "layer_fro_ratio_max_over_min_max": float(np.max(fro.max(1) / fro.min(1)))}


def flipped_cells_continuation(rows, depth=4, width=64, d_in=16, n=256, eta=0.05, total_steps=400, n_ctrl=4):
    """For every mixed census cell whose sign flips somewhere on (c,1/c,1,..,1), c in [0.1,10],
    with steps left to run: retrain the conjugate at c* (geometric centre of the flipped region
    nearest c = 1), continue PCGrad vs summed GD to the last step, and compare with random
    perturbations of PCGrad's norm on the same step. Answers whether a gauge that switches
    PCGrad on helps systematically, or only moves an edge-of-stability trajectory."""
    grid = np.logspace(-1, 1, 2001)
    out = []
    for r in rows:
        if not r["mixed"] or r["step"] >= total_steps - 1:
            continue
        a = np.array(r["ips"])
        sg = np.sign(a[0] * grid ** -2 + a[1] * grid ** 2 + a[2:].sum())
        fl = sg != np.sign(a.sum())
        if not fl.any():
            continue
        # contiguous flipped segments; pick the one nearest c = 1
        idx = np.where(fl)[0]
        segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        seg = min(segs, key=lambda s: np.min(np.abs(np.log(grid[s]))))
        cstar = float(np.exp(np.mean(np.log(grid[seg]))))
        T = r["step"]; K = total_steps - 1 - T
        rng = np.random.default_rng(1000 + r["seed"])
        X, Ys = make_tasks(rng, n, d_in, r["rho"])
        W0, H0 = init_net(rng, d_in, width, depth)
        rec = {"seed": r["seed"], "rho": r["rho"], "step": T, "c_star": cstar, "seg_c_range": [float(grid[seg[0]]), float(grid[seg[-1]])]}
        for label, cval in (("c1", 1.0), ("cstar", cstar)):
            c = np.ones(depth); c[0], c[1] = cval, 1.0 / cval
            etas = [eta * cl ** 2 for cl in c]
            run = gd_run(X, Ys, [cl * w for cl, w in zip(c, W0)], H0, etas, eta, T + 1, record_every=1)
            Wc, Hc = run["W"][T], run["H"][T]
            _, gW, gH = grads(X, Ys, Wc, Hc)
            cos, u_pc, u_sum = pcgrad_trunk(gW)
            fin_pc, n_act = pcgrad_run(X, Ys, Wc, Hc, etas, eta, K)
            fin_sum = gd_run(X, Ys, Wc, Hc, etas, eta, K)["loss"][-1]
            rec[label] = {"cos": cos, "active_steps": n_act, "pc_over_sum": [float(fin_pc[i] / fin_sum[i]) for i in range(2)]}
            corr = float(np.sqrt(sum(np.sum((p - q) ** 2) for p, q in zip(u_pc, u_sum))))
            if corr > 0:
                gHsum = [gH[0][i] + gH[1][i] for i in range(2)]
                ctrl = perturbed_continuation(X, Ys, Wc, Hc, etas, eta, u_sum, gHsum, corr, K, n_ctrl, seed=T + r["seed"])
                rec[label]["ctrl_over_sum_median"] = [float(np.median([f[i] / fin_sum[i] for f in ctrl])) for i in range(2)]
                rec[label]["relative_perturbation"] = corr / float(np.sqrt(sum(np.sum(q ** 2) for q in u_sum)))
        out.append(rec)
    return out


# ----------------------------------------------------------------------------- figure
def plot_sweep(sw, path, nearest=None, n_cells=125, neuron_nearest=None, nearest_label="any layers"):
    """Left: trunk cosine along (c,1/c,1,1) for the sweep cell (conjugates retrained per c).
    Right: over the census, fraction of all cells whose sign some rescaling of factor <= c
    flips: `nearest` = nearest flipping factor per mixed cell (over the whole product-one
    per-layer group, largest per-layer factor), and, if given, c on one hidden neuron's fan-in and
    1/c on its fan-out (neuron-wise symmetry). Curves are labelled in place (no legend box)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = np.array(sw["c"]); cos = np.array(sw["cos"])
    sign0 = np.sign(cos[np.argmin(np.abs(c - 1))])
    dev = max(np.ptp(sw["loss1"]), np.ptp(sw["loss2"]))
    fig, ax = plt.subplots(1, 2, figsize=(3.4, 0.92), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.15]})
    flip = np.sign(cos) != sign0
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].plot(c, cos, color="C3", lw=1.2); ax[0].set_ylabel("trunk cosine", fontsize=6.5, labelpad=1)
    if flip.any():
        ax[0].fill_between(c, cos.min() - 0.02, cos.max() + 0.02, where=flip, color="C3", alpha=0.15, lw=0)
    ax[0].set_xscale("log"); ax[0].set_xlabel(r"gauge $c$ on $(c,1/c,1,1)$", fontsize=6.5, labelpad=1)
    ax[0].set_ylim(cos.min() - 0.02, cos.max() + 0.02)
    xs = np.logspace(0, np.log10(30), 400)
    if nearest is not None:
        v = np.array([x if x is not None else np.inf for x in nearest], float)
        frac = np.array([(v <= x).mean() * len(v) / n_cells for x in xs])
        ax[1].plot(xs, frac, color="C0", lw=1.2)
        ax[1].text(9.5, float(frac[-1]) - 0.06, nearest_label, color="C0", fontsize=6, ha="left", va="top")
    if neuron_nearest is not None:
        v2 = np.array([x if x is not None else np.inf for x in neuron_nearest], float)
        frac2 = np.array([(v2 <= x).mean() * len(v2) / n_cells for x in xs])
        ax[1].plot(xs, frac2, color="C2", lw=1.2, ls="--")
        ax[1].text(2.35, 0.86, "one neuron", color="C2", fontsize=6, ha="left", va="bottom")
    for xv, lab in ((np.sqrt(2), r"$\sqrt{2}$"), (2, "2")):
        ax[1].axvline(xv, color="0.5", lw=0.6, ls=":")
        ax[1].text(xv, 0.97, lab, fontsize=6, ha="center", va="top", color="0.3")
    ax[1].set_xscale("log"); ax[1].set_xlim(1, 30); ax[1].set_ylim(-0.03, 1.05)
    ax[1].set_xticks([1, 2, 5, 10, 30]); ax[1].set_xticklabels(["1", "2", "5", "10", "30"])
    ax[1].set_ylabel("cells flipped", fontsize=6.5, labelpad=1)
    ax[1].set_xlabel(r"nearest flipping factor $c$", fontsize=6.5, labelpad=1)
    for a in ax: a.tick_params(labelsize=6, pad=1.5)
    fig.savefig(path, dpi=200)
    fig.savefig(os.path.splitext(path)[0] + ".pdf")
    return dev


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    print("== 1. exact conjugation ==")
    conj = check_conjugation()
    print(json.dumps(conj, indent=1))
    conj_single = check_conjugation(c=[0.5, 1, 1, 1])      # prod != 1, heads absorb
    print("head-absorbed, c=(1/2,1,1,1):", json.dumps(conj_single))

    print("\n== 3. sign census ==")
    cen = sign_census(sharpness=True)
    print(f"mixed-sign cells: {cen['mixed']}/{cen['total']} ({100*cen['frac']:.0f}%), diverged {cen['diverged']}")
    print(f"max |head cross term| over all cells: {cen['max_abs_head_cross']:.3e}")
    print("mixed by step:", cen["mixed_by_step"], " by rho:", cen["mixed_by_rho"])
    print("conflict rate:", cen["conflict_rate"], "by step:", cen["conflict_by_step"])
    print("eta*lambda_max by step:", json.dumps(cen["eta_lambda_max_by_step"]), "frac > 2 post-init:", cen["eta_lambda_max_frac_above_2_post_init"])

    print("\n== 5. reachability ==")
    prof = init_scale_profiles()
    print("init profiles (over He, product-one):", {k: np.round(v, 3).tolist() for k, v in prof.items() if k.endswith("prod1")})
    reach = reachability(cen["rows"], measured_profile=prof["orthogonal_over_he_prod1"])
    print(json.dumps({k: v for k, v in reach.items() if not k.endswith("_values")}, indent=1))
    print("all-layer nearest flip (prod 1):", reach["nearest_flip_all_layers_prod1"], " closed form (unconstrained):", reach["nearest_flip_all_layers_closed_form_unconstrained"])
    print("box supremum of |cos| either side:", json.dumps(reach["abs_cos_either_side_box_supremum"]))
    print("conflict by rho:", cen["conflict_by_rho"], "post-init:", cen["conflict_by_rho_post_init"], "cos median by rho post-init:", cen["cos_trunk_median_by_rho_post_init"])

    print("\n== 6. PCGrad continuation on every flipping cell (c* vs c=1, with random-kick control) ==")
    flipc = flipped_cells_continuation(cen["rows"])
    rs = np.array([r["cstar"]["pc_over_sum"] for r in flipc]); r1 = np.array([r["c1"]["pc_over_sum"] for r in flipc])
    cs_ = np.array([r["cstar"].get("ctrl_over_sum_median", [np.nan, np.nan]) for r in flipc])
    print(f"cells: {len(flipc)}; at c*: PCGrad/summed median {np.median(rs,0)}, min {rs.min(0)}, max {rs.max(0)}; "
          f"random-kick control median {np.nanmedian(cs_,0)}; at c=1: median {np.median(r1,0)}, active steps at c=1 {[r['c1']['active_steps'] for r in flipc]}")
    print("c* values:", np.round([r["c_star"] for r in flipc], 3).tolist())

    print("\n== 4. gauge sweep ==")
    mixed_rows = [r for r in cen["rows"] if r["mixed"]]
    if not mixed_rows:
        print("no mixed-sign cell found; sweep skipped"); sys.exit(0)
    # SELECTION (disclosed in the abstract): among mixed cells, the one whose sign-change
    # window on the fixed (c,1/c,1,1) path is widest within c in [0.1, 10], by closed form.
    grid = np.logspace(-1, 1, 2001)
    def window(r):
        a = r["ips"]
        sg = np.sign(a[0] * grid ** -2 + a[1] * grid ** 2 + a[2] + a[3])
        return int((sg != np.sign(sum(a))).sum())
    row = max(mixed_rows, key=window)
    print("checkpoint:", {k: row[k] for k in ("seed", "rho", "step", "signs")}, "flipped grid points:", window(row))
    sw = gauge_sweep(row)
    print("layerwise <g1,g2>:", np.array(sw["ips"]))
    cosarr = np.array(sw["cos"]); c = np.array(sw["c"])
    i1 = int(np.argmin(np.abs(c - 1)))
    print(f"trunk cosine at c=1: {cosarr[i1]:+.4f}; range: [{cosarr.min():+.4f}, {cosarr.max():+.4f}]")
    print("sign changes at c =", zero_crossings(c, cosarr))
    print(f"max conjugation residual over the retrained sweep: {max(sw['conj_resid']):.2e}")
    ew = np.array(sw["eta_weighted_ip"]); print(f"eta-weighted invariant: {ew[i1]:+.6e}, ptp {np.ptp(ew):.2e}")
    print(f"PCGrad ratio range: [{min(sw['pcgrad_ratio']):.3f}, {max(sw['pcgrad_ratio']):.3f}]; max|ratio-max(0,-cos)| = {np.max(np.abs(np.array(sw['pcgrad_ratio'])-np.maximum(0,-cosarr))):.1e}")
    within = (c >= 0.5) & (c <= 2)
    print(f"PCGrad ratio within c in [1/2,2]: max {max(np.array(sw['pcgrad_ratio'])[within]):.4f}")
    d1, d2 = np.array(sw["dL1_pc_minus_sum"]), np.array(sw["dL2_pc_minus_sum"])
    print(f"one-step loss effect of PCGrad minus summed step: task1 [{d1.min():+.2e},{d1.max():+.2e}], task2 [{d2.min():+.2e},{d2.max():+.2e}]")
    print(f"  at c=0.1: {d1[0]:+.2e} {d2[0]:+.2e}; at c=10: {d1[-1]:+.2e} {d2[-1]:+.2e}; loss at c=1: {sw['loss1'][i1]:.5f} {sw['loss2'][i1]:.5f}")
    fo = np.array(sw["dL_pc_minus_sum_first_order"]); s01 = np.array(sw["dL_pc_minus_sum_eta_x0.1"]); s001 = np.array(sw["dL_pc_minus_sum_eta_x0.01"])
    print(f"  eta scaling at c=0.1: first-order prediction {fo[0]}, measured at eta/10 {s01[0]} (pred {fo[0]*0.1}), at eta/100 {s001[0]} (pred {fo[0]*0.01})")
    print(f"  eta scaling at c=10 : first-order prediction {fo[-1]}, measured at eta/10 {s01[-1]}, at eta/100 {s001[-1]}")
    ce = np.array(sw["cos_eta"]); LP = np.array(sw["loss_after_pcgrad_eta_metric"])
    print(f"eta-metric PCGrad: cos_eta {ce[i1]:+.5f}, ptp {np.ptp(ce):.1e}; one-step losses ptp {np.ptp(LP[:,0]):.1e} {np.ptp(LP[:,1]):.1e}")
    fp = np.array(sw["final_pc"]); fs = np.array(sw["final_sum"]); act = np.array(sw["pc_active_steps"])
    print(f"continuation to step {sw['T']+sw['K_continuation']}: summed-GD final losses ptp {np.ptp(fs[:,0]):.1e} {np.ptp(fs[:,1]):.1e} (value {fs[i1,0]:.6f} {fs[i1,1]:.6f})")
    rat = fp / fs
    print(f"  PCGrad/summed final-loss ratio: at c=1 {rat[i1]}, c=0.1 {rat[0]}, c=10 {rat[-1]}, min {rat.min(0)}, max {rat.max(0)}; PCGrad active steps at c=0.1/1/10: {act[0]}/{act[i1]}/{act[-1]}")
    inwin = (c > 0.44) & (c < 1.86)
    print(f"  inside the positive-cosine interval: ratio ptp {np.ptp(rat[inwin],0)}, active steps max {act[inwin].max()}")
    print(f"  sharpness: eta*lambda_max at T = {sw['eta_lambda_max_at_T']:.3f}, at step {sw['T']+sw['K_continuation']} = {sw['eta_lambda_max_at_end']:.3f}; "
          f"summed-loss increases on {sw['summed_loss_nonmonotone_steps_after_T']} of {sw['K_continuation']} continuation steps")
    ct = np.array(sw["ctrl_final"]) / fs[:, None, :]
    outside = ~inwin
    print(f"  random-kick control (same norm as PCGrad's correction, same step): outside the window PCGrad ratio min {rat[outside].min(0)}, "
          f"control min {ct[outside].min(axis=(0,1))}, control median {np.median(ct[outside],axis=(0,1))}; PCGrad median {np.median(rat[outside],0)}")
    an = np.array(sw["anti_final"]) / fs; gp = np.array(sw["gsum_plus_final"]) / fs; gm = np.array(sw["gsum_minus_final"]) / fs
    print(f"  directed controls outside the window: anti-PCGrad median {np.median(an[outside],0)} min {an[outside].min(0)}; "
          f"+u_sum kick median {np.median(gp[outside],0)} min {gp[outside].min(0)}; -u_sum kick median {np.median(gm[outside],0)} min {gm[outside].min(0)}; "
          f"PCGrad below anti in {(rat[outside] < an[outside]).mean(0)} of gauges; at c=0.126: PCGrad {rat[np.argmin(np.abs(c-0.126))]} anti {an[np.argmin(np.abs(c-0.126))]}")
    for k, v in sw["ctrl_dense"].items():
        co = np.array(v["ctrl_over_sum"])
        print(f"  c={k}: PCGrad {np.round(v['pcgrad_over_sum'],3)} (rel. perturbation {v['relative_perturbation']:.4f}, active {v['active_steps']}); "
              f"20 random kicks: median {np.round(np.median(co,0),3)}, min {np.round(co.min(0),3)}, max {np.round(co.max(0),3)}")
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump({"conjugation": conj, "conjugation_head_absorbed": conj_single,
                   "census": {k: cen[k] for k in ("mixed", "total", "frac", "diverged", "max_abs_head_cross", "mixed_by_step", "mixed_by_rho",
                                                  "conflict_rate", "conflict_rate_post_init", "conflict_by_step", "conflict_by_rho", "conflict_by_rho_post_init",
                                                  "cos_trunk_median_by_rho_post_init", "cos_trunk_median_by_rho_all",
                                                  "eta_lambda_max_by_step", "eta_lambda_max_frac_above_2_post_init", "rows")},
                   "init_scale_profiles": prof, "reachability": reach,
                   "flipped_cells_pcgrad_continuation": flipc,
                   "sweep_checkpoint": {k: row[k] for k in ("seed", "rho", "step", "signs")},
                   "sweep_selection": "mixed cell with the widest sign-flipped region on (c,1/c,1,1), c in [0.1,10]",
                   "sweep_sign_changes_at_c": zero_crossings(c, cosarr),
                   "sweep": sw}, f, indent=1, allow_nan=False)
    os.makedirs(FIGDIR, exist_ok=True)
    neuron_vals = None
    if os.path.exists(os.path.join(HERE, "eos_neuron.json")):      # optional single-neuron nearest flips
        neuron_vals = json.load(open(os.path.join(HERE, "eos_neuron.json")))["base_eta0.05_400"]["neuron"]["nearest_single_neuron_values"]
    dev = plot_sweep(sw, os.path.join(FIGDIR, "gauge_sweep.png"), nearest=reach["nearest_flip_all_layers_prod1_values"], n_cells=cen["total"], neuron_nearest=neuron_vals)
    print(f"loss deviation along path: {dev:.2e}; figure -> figures/gauge_sweep.png")
