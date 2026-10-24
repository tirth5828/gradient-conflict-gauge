"""census_scale.py -- the sign census at 1000 seeds, with seed-cluster bootstrap CIs.

Three studies (all float64, all reusing make_tasks / init_net / grads from conjugation_check.py,
same rng convention: rng = default_rng(1000 + s), inputs, teachers and W_0 shared across rho and
across eta within a seed; seeds s = 10..1009, disjoint from the development seeds 0-9):

  main   : 1000 seeds x rho in {-0.5,-0.2,0,0.3,0.7}, width 64, depth 4, eta 0.05, 400 steps.
  eta    : eta in {0.01, 0.02, 0.1} x 400 steps and eta 0.01 x 2000 steps, seeds 10..209
           (eta 0.05 x 400 is the first 200 seeds of `main`).
  near1  : rho in {0.9, 0.99, 1.0} at (eta 0.05, 400), (eta 0.01, 400), (eta 0.01, 2000), seeds 10..209
           (rho 0.7 at the same regimes comes from `main` / `eta`).

Per run we record at EVERY step: the four layerwise inner products a_l = <g_1^l, g_2^l> (as a sign
code), the trunk cosine, both losses; at the five checkpoints (linspace(0, steps-1, 5)) the full a_l,
per-layer squared gradient norms of both tasks and both losses. The GD loop is our own (so that the
per-task gradients that grads() already computes are recorded every step) and is checked against
gd_run to 0.0 on one run (stage `verify`), which also reproduces the seeds 0-4 census.

Usage:  python census_scale.py verify | run | analyze      (run: ~2 h with 3 workers)
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from conjugation_check import make_tasks, init_net, grads, gd_run, layer_inner_products, trunk_cosine  # noqa: E402

SCRATCH = os.path.join(HERE, "census_scale_parts")
KEY = "census_scale"
N_WORKERS = 3
D_IN, WIDTH, DEPTH, N = 16, 64, 4, 256
RHOS_MAIN = (-0.5, -0.2, 0.0, 0.3, 0.7)
RHOS_NEAR1 = (0.9, 0.99, 1.0)
SEEDS_MAIN = list(range(10, 1010))
SEEDS_SUB = list(range(10, 210))
SEEDS_SUB100 = list(range(10, 110))   # the 2000-step configs, trimmed to fit the compute budget

CONFIGS = {
    # name: (seeds, rhos, eta, steps)
    "main_eta0.05_T400": (SEEDS_MAIN, RHOS_MAIN, 0.05, 400),
    "eta0.01_T400": (SEEDS_SUB, RHOS_MAIN, 0.01, 400),
    "eta0.02_T400": (SEEDS_SUB, RHOS_MAIN, 0.02, 400),
    "eta0.1_T400": (SEEDS_SUB, RHOS_MAIN, 0.1, 400),
    "eta0.01_T2000": (SEEDS_SUB100, RHOS_MAIN, 0.01, 2000),
    "near1_eta0.05_T400": (SEEDS_SUB, RHOS_NEAR1, 0.05, 400),
    "near1_eta0.01_T400": (SEEDS_SUB, RHOS_NEAR1, 0.01, 400),
    "near1_eta0.01_T2000": (SEEDS_SUB100, RHOS_NEAR1, 0.01, 2000),
}


# ----------------------------------------------------------------------------- one run
def run_one(s, rho, eta, steps, n_ckpt=5):
    """GD on the summed loss from the standard rng stream; records per-step and checkpoint statistics."""
    rng = np.random.default_rng(1000 + s)
    X, Ys = make_tasks(rng, N, D_IN, rho)
    W, H = init_net(rng, D_IN, WIDTH, DEPTH)
    W = [w.copy() for w in W]; H = [h.copy() for h in H]
    ckpts = np.linspace(0, steps - 1, n_ckpt).astype(int)
    ck_pos = {int(k): i for i, k in enumerate(ckpts)}
    a_step = np.empty((steps, DEPTH))            # kept in float64 in memory; sign code persisted
    cos_step = np.empty(steps)
    loss_step = np.empty((steps, 2))
    ck_a = np.full((n_ckpt, DEPTH), np.nan); ck_g1 = np.full((n_ckpt, DEPTH), np.nan)
    ck_g2 = np.full((n_ckpt, DEPTH), np.nan); ck_loss = np.full((n_ckpt, 2), np.nan)
    ck_cos = np.full(n_ckpt, np.nan); ck_fro = np.full((n_ckpt, DEPTH), np.nan)
    diverged_at = -1
    for t in range(steps):
        losses, gW, gH = grads(X, Ys, W, H)
        a = np.array([np.sum(gW[0][l] * gW[1][l]) for l in range(DEPTH)])
        g1n2 = np.array([np.sum(g ** 2) for g in gW[0]])
        g2n2 = np.array([np.sum(g ** 2) for g in gW[1]])
        a_step[t] = a
        cos_step[t] = a.sum() / np.sqrt(g1n2.sum() * g2n2.sum())
        loss_step[t] = losses
        if t in ck_pos:
            i = ck_pos[t]
            ck_a[i], ck_g1[i], ck_g2[i], ck_loss[i], ck_cos[i] = a, g1n2, g2n2, losses, cos_step[t]
            ck_fro[i] = [np.linalg.norm(w) for w in W]
        if not np.all(np.isfinite(losses)):
            diverged_at = t
            break
        for l in range(DEPTH):
            W[l] = W[l] - eta * (gW[0][l] + gW[1][l])
        for i in range(2):
            H[i] = H[i] - eta * (gH[0][i] + gH[1][i])
    # sign code over the 4 layers: bit l set iff a_l < 0 (0 = all positive, 15 = all negative)
    neg = a_step < 0
    code = (neg[:, 0] * 1 + neg[:, 1] * 2 + neg[:, 2] * 4 + neg[:, 3] * 8).astype(np.int8)
    if diverged_at >= 0:
        code[diverged_at:] = -1; cos_step[diverged_at:] = np.nan; loss_step[diverged_at:] = np.nan
    return dict(code=code, cos=cos_step, loss=loss_step, ck_a=ck_a, ck_g1=ck_g1, ck_g2=ck_g2,
                ck_loss=ck_loss, ck_cos=ck_cos, ck_fro=ck_fro, diverged_at=diverged_at,
                final_W=W, final_H=H)


def run_seed(s, rhos, eta, steps):
    outs = [run_one(s, rho, eta, steps) for rho in rhos]
    for o in outs:
        o.pop("final_W"); o.pop("final_H")
    return outs


def stack(results, seeds, rhos, steps):
    S, R = len(seeds), len(rhos)
    out = {
        "seeds": np.array(seeds), "rhos": np.array(rhos), "steps": steps,
        "code": np.stack([[o["code"] for o in row] for row in results]),          # (S,R,T) int8
        "cos": np.stack([[o["cos"] for o in row] for row in results]),            # (S,R,T)
        "loss": np.stack([[o["loss"] for o in row] for row in results]),          # (S,R,T,2)
        "ck_a": np.stack([[o["ck_a"] for o in row] for row in results]),          # (S,R,5,4)
        "ck_g1n2": np.stack([[o["ck_g1"] for o in row] for row in results]),
        "ck_g2n2": np.stack([[o["ck_g2"] for o in row] for row in results]),
        "ck_loss": np.stack([[o["ck_loss"] for o in row] for row in results]),    # (S,R,5,2)
        "ck_cos": np.stack([[o["ck_cos"] for o in row] for row in results]),      # (S,R,5)
        "ck_fro": np.stack([[o["ck_fro"] for o in row] for row in results]),
        "diverged_at": np.array([[o["diverged_at"] for o in row] for row in results]),
        "ckpts": np.linspace(0, steps - 1, 5).astype(int),
    }
    return out


# ----------------------------------------------------------------------------- stages
def verify():
    """(a) our loop == gd_run to 0.0; (b) seeds 0-4 reproduce the 52/125 mixed cells and 18.4% conflict."""
    o = run_one(0, 0.3, 0.05, 400)
    rng = np.random.default_rng(1000)
    X, Ys = make_tasks(rng, N, D_IN, 0.3); W, H = init_net(rng, D_IN, WIDTH, DEPTH)
    hist = gd_run(X, Ys, W, H, [0.05] * DEPTH, 0.05, 400, record_every=1)
    dev = max(np.max(np.abs(a - b)) for a, b in zip(o["final_W"], hist["final_W"]))
    # checkpoint a_l against layer_inner_products on the recorded iterates
    devs = []
    for i, k in enumerate([0, 99, 199, 299, 399]):
        ips, _, gW, _ = layer_inner_products(X, Ys, hist["W"][k], hist["H"][k])
        devs.append(max(np.max(np.abs(np.array(ips) - o["ck_a"][i])), abs(trunk_cosine(gW) - o["ck_cos"][i])))
    print(f"loop vs gd_run: max |dW| = {dev:.3e}; checkpoint a_l / cos deviation max = {max(devs):.3e}")
    res = [run_seed(s, RHOS_MAIN, 0.05, 400) for s in range(5)]
    d = stack(res, list(range(5)), RHOS_MAIN, 400)
    mixed = (d["ck_a"] < 0).any(-1) & (d["ck_a"] > 0).any(-1)
    conf = d["ck_a"].sum(-1) < 0
    print(f"seeds 0-4: mixed {mixed.sum()}/125 (post-init {mixed[:, :, 1:].sum()}/100), conflict {conf.mean():.3f}, "
          f"by step {conf.reshape(-1, 5).mean(0)}, by rho {conf.transpose(1, 0, 2).reshape(5, -1).mean(1)}")
    print(f"seeds 0-4 whole-run conflict rate {(d['cos'] < 0).mean():.3f}; even steps {(d['cos'][:, :, 0::2] < 0).mean():.3f}, odd {(d['cos'][:, :, 1::2] < 0).mean():.3f}")
    # timing
    t0 = time.time(); run_one(10, 0.0, 0.05, 400); print(f"one 400-step run: {time.time() - t0:.2f} s")


def run(names=None):
    from joblib import Parallel, delayed
    os.makedirs(SCRATCH, exist_ok=True)
    names = names or list(CONFIGS)
    for name in names:
        path = os.path.join(SCRATCH, f"{name}.npz")
        if os.path.exists(path):
            print(f"[skip] {name} exists"); continue
        seeds, rhos, eta, steps = CONFIGS[name]
        t0 = time.time()
        print(f"[run] {name}: {len(seeds)} seeds x {len(rhos)} rho x {steps} steps, eta {eta}", flush=True)
        res = Parallel(n_jobs=N_WORKERS, batch_size=4, verbose=0)(delayed(run_seed)(s, rhos, eta, steps) for s in seeds)
        d = stack(res, seeds, rhos, steps)
        np.savez(path, **d)
        print(f"[done] {name} in {(time.time() - t0) / 60:.1f} min; diverged runs {(d['diverged_at'] >= 0).sum()}", flush=True)


# ----------------------------------------------------------------------------- analysis helpers
N_BOOT = 2000
BOOT_RNG = np.random.default_rng(20260901)


def boot_frac(num, den, n_boot=N_BOOT):
    """Cluster bootstrap over seeds (axis 0) of a ratio sum(num)/sum(den). num, den: (S, ...)."""
    num = np.asarray(num, float).reshape(len(num), -1).sum(1)
    den = np.asarray(den, float).reshape(len(den), -1).sum(1)
    if den.sum() == 0:
        return None, None, None
    S = len(num)
    idx = BOOT_RNG.integers(0, S, size=(n_boot, S))
    with np.errstate(invalid="ignore"):
        b = num[idx].sum(1) / den[idx].sum(1)
    b = b[np.isfinite(b)]
    return float(num.sum() / den.sum()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def frac(mask, valid=None):
    """mask: (S, ...) boolean; returns dict with point estimate, CI, counts."""
    mask = np.asarray(mask)
    valid = np.ones_like(mask, bool) if valid is None else np.asarray(valid)
    p, lo, hi = boot_frac(mask & valid, valid)
    if p is None:
        return {"frac": None, "ci95": [None, None], "n": int(valid.sum()), "k": int((mask & valid).sum())}
    return {"frac": p, "ci95": [lo, hi], "n": int(valid.sum()), "k": int((mask & valid).sum())}


def boot_median(values_by_seed, n_boot=500):
    """values_by_seed: list of 1-D arrays (one per seed); cluster bootstrap of the pooled median."""
    vals = [np.asarray(v, float) for v in values_by_seed]
    pooled = np.concatenate(vals)
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return {"median": None, "ci95": [None, None], "q25": None, "q75": None, "n": 0}
    S = len(vals)
    meds = []
    for _ in range(n_boot):
        idx = BOOT_RNG.integers(0, S, size=S)
        bp = np.concatenate([vals[i] for i in idx]); bp = bp[np.isfinite(bp)]
        if bp.size:
            meds.append(np.median(bp))
    return {"median": float(np.median(pooled)), "ci95": [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))],
            "q25": float(np.percentile(pooled, 25)), "q75": float(np.percentile(pooled, 75)), "n": int(pooled.size)}


def pct(d):
    if d["frac"] is None:
        return f"n/a ({d['k']}/{d['n']})"
    return f"{100 * d['frac']:.1f}% [{100 * d['ci95'][0]:.1f}, {100 * d['ci95'][1]:.1f}] ({d['k']}/{d['n']})"


def load(name):
    d = dict(np.load(os.path.join(SCRATCH, f"{name}.npz")))
    d["steps"] = int(d["steps"])
    return d


def cell_stats(d, rho_sel=None):
    """Checkpoint-cell statistics with CIs for one config (optionally a subset of rho indices)."""
    a = d["ck_a"]; cos = d["ck_cos"]; ck = d["ckpts"]; rhos = d["rhos"]
    if rho_sel is not None:
        a = a[:, rho_sel]; cos = cos[:, rho_sel]; rhos = rhos[rho_sel]
    S, R, K, L = a.shape
    valid = np.isfinite(a).all(-1)
    mixed = (a < 0).any(-1) & (a > 0).any(-1) & valid
    conf = (a.sum(-1) < 0) & valid
    post = np.zeros_like(valid); post[:, :, 1:] = True
    out = {
        "n_seeds": int(S), "rhos": rhos.tolist(), "checkpoints": ck.tolist(),
        "cells_valid": int(valid.sum()), "cells_total": int(valid.size), "diverged_runs": int((d["diverged_at"] >= 0).sum()),
        "mixed_all": frac(mixed, valid), "mixed_post_init": frac(mixed, valid & post),
        "mixed_by_step": {int(ck[k]): frac(mixed[:, :, k], valid[:, :, k]) for k in range(K)},
        "mixed_by_rho": {str(rhos[r]): frac(mixed[:, r], valid[:, r]) for r in range(R)},
        "mixed_by_rho_post_init": {str(rhos[r]): frac(mixed[:, r, 1:], valid[:, r, 1:]) for r in range(R)},
        "conflict_cells_all": frac(conf, valid), "conflict_cells_post_init": frac(conf, valid & post),
        "conflict_by_step": {int(ck[k]): frac(conf[:, :, k], valid[:, :, k]) for k in range(K)},
        "conflict_by_rho": {str(rhos[r]): frac(conf[:, r], valid[:, r]) for r in range(R)},
        "conflict_by_rho_post_init": {str(rhos[r]): frac(conf[:, r, 1:], valid[:, r, 1:]) for r in range(R)},
        "conflict_by_rho_step": {str(rhos[r]): {int(ck[k]): frac(conf[:, r, k], valid[:, r, k]) for k in range(K)} for r in range(R)},
        "cos_median_by_rho_post_init": {str(rhos[r]): boot_median([cos[s, r, 1:][valid[s, r, 1:]] for s in range(S)]) for r in range(R)},
        "cos_median_by_rho_all": {str(rhos[r]): boot_median([cos[s, r][valid[s, r]] for s in range(S)]) for r in range(R)},
        "abs_cos_mixed": boot_median([np.abs(cos[s][mixed[s]]) for s in range(S)]),
        "abs_cos_single_sign": boot_median([np.abs(cos[s][valid[s] & ~mixed[s]]) for s in range(S)]),
        "abs_cos_mixed_post_init": boot_median([np.abs(cos[s, :, 1:][mixed[s, :, 1:]]) for s in range(S)]),
        "abs_cos_single_sign_post_init": boot_median([np.abs(cos[s, :, 1:][valid[s, :, 1:] & ~mixed[s, :, 1:]]) for s in range(S)]),
        "abs_cos_mixed_by_rho_post_init": {str(rhos[r]): boot_median([np.abs(cos[s, r, 1:][mixed[s, r, 1:]]) for s in range(S)], 200) for r in range(R)},
        "frac_abs_cos_below_0.1_mixed": frac(np.abs(np.nan_to_num(cos, nan=1.0)) < 0.1, mixed),
        "frac_abs_cos_below_0.1_single_sign": frac(np.abs(np.nan_to_num(cos, nan=1.0)) < 0.1, valid & ~mixed),
        "final_loss_summed_median": float(np.nanmedian(d["ck_loss"][:, :, -1].sum(-1))) if rho_sel is None else float(np.nanmedian(d["ck_loss"][:, rho_sel, -1].sum(-1))),
        "final_loss_median_by_rho": {str(rhos[r]): float(np.nanmedian((d["ck_loss"][:, rho_sel] if rho_sel is not None else d["ck_loss"])[:, r, -1].sum(-1))) for r in range(R)},
    }
    # sign-pattern frequencies over the 4 layers (bit l set iff a_l < 0)
    neg = (a < 0)
    code = neg[..., 0] * 1 + neg[..., 1] * 2 + neg[..., 2] * 4 + neg[..., 3] * 8
    pat = {}
    for c in range(16):
        lab = "".join("-" if (c >> l) & 1 else "+" for l in range(4))
        pat[lab] = {"all": frac(code == c, valid), "post_init": frac(code == c, valid & post),
                    "by_step": [frac(code[:, :, k] == c, valid[:, :, k])["frac"] for k in range(K)]}
    out["sign_patterns"] = pat
    # per-layer negative fractions
    out["frac_a_l_negative_post_init"] = [frac(neg[..., l], valid & post) for l in range(L)]
    out["frac_a_l_negative_step0"] = [frac(neg[:, :, 0, l], valid[:, :, 0]) for l in range(L)]
    # per-layer share of |a_l| in mixed cells: which layer carries the minority sign
    return out


def run_stats(d, rho_sel=None):
    """Whole-run (every step) conflict and mixed statistics."""
    cos = d["cos"]; code = d["code"]; rhos = d["rhos"]
    if rho_sel is not None:
        cos = cos[:, rho_sel]; code = code[:, rho_sel]; rhos = rhos[rho_sel]
    S, R, T = cos.shape
    valid = np.isfinite(cos)
    conf = (cos < 0) & valid
    mixed = (code > 0) & (code < 15) & valid
    dec = np.array_split(np.arange(T), 10)
    out = {
        "conflict_run_all": frac(conf, valid),
        "conflict_run_post_init": frac(conf[:, :, 1:], valid[:, :, 1:]),
        "conflict_run_even_steps": frac(conf[:, :, 0::2], valid[:, :, 0::2]),
        "conflict_run_odd_steps": frac(conf[:, :, 1::2], valid[:, :, 1::2]),
        "conflict_run_by_rho": {str(rhos[r]): frac(conf[:, r], valid[:, r]) for r in range(R)},
        "conflict_run_by_decile": [frac(conf[:, :, ix], valid[:, :, ix]) for ix in dec],
        "conflict_run_by_decile_by_rho": {str(rhos[r]): [frac(conf[:, r, ix], valid[:, r, ix])["frac"] for ix in dec] for r in range(R)},
        "mixed_run_all": frac(mixed, valid),
        "mixed_run_post_init": frac(mixed[:, :, 1:], valid[:, :, 1:]),
        "mixed_run_by_decile": [frac(mixed[:, :, ix], valid[:, :, ix]) for ix in dec],
        "conflict_per_step_curve": (conf.sum((0, 1)) / np.maximum(valid.sum((0, 1)), 1)).tolist(),
        "mixed_per_step_curve": (mixed.sum((0, 1)) / np.maximum(valid.sum((0, 1)), 1)).tolist(),
        "median_summed_loss_per_step": np.nanmedian(d["loss"][:, rho_sel].sum(-1) if rho_sel is not None else d["loss"].sum(-1), axis=(0, 1)).tolist(),
        # fraction of runs whose sign alternates step to step (period-2) at least half the time over the last 100 steps
        "frac_runs_sign_alternating_last100": frac((np.sign(cos[:, :, -100:-1]) != np.sign(cos[:, :, -99:])).mean(-1) > 0.5, valid.all(-1)),
        "sign_change_rate_last100": frac((np.sign(cos[:, :, -100:-1]) != np.sign(cos[:, :, -99:])), valid[:, :, -99:]),
    }
    return out


def matched_loss(d_ref, d, S_sub):
    """For each (seed, rho) among the first S_sub seeds: target = summed loss of the eta=0.05 run at its last step
    (step 399); in run d find the first step whose summed loss <= target, record whether the cosine is negative
    there (raw) and the conflict fraction over the 20 steps after it. Also the loss-matched per-step curve."""
    Lref = d_ref["loss"][:S_sub].sum(-1)[:, :, -1]                # (S,R)
    L = d["loss"][:S_sub].sum(-1)                                  # (S,R,T)
    cos = d["cos"][:S_sub]
    S, R, T = L.shape
    reached = np.zeros((S, R), bool); t_hit = np.full((S, R), -1)
    neg_at = np.zeros((S, R), bool); neg_win = np.full((S, R), np.nan)
    for s in range(S):
        for r in range(R):
            hit = np.where(L[s, r] <= Lref[s, r])[0]
            if hit.size:
                t = int(hit[0]); reached[s, r] = True; t_hit[s, r] = t
                neg_at[s, r] = cos[s, r, t] < 0
                neg_win[s, r] = np.mean(cos[s, r, t:t + 20] < 0)
    out = {"frac_runs_reaching_target": frac(reached),
           "median_step_at_match": float(np.median(t_hit[reached])) if reached.any() else None,
           "conflict_at_matched_loss_raw": frac(neg_at, reached),
           "conflict_at_matched_loss_20step_window": boot_frac(np.nan_to_num(neg_win) * reached, reached) if reached.any() else None,
           "target_loss_median": float(np.median(Lref))}
    return out


def analyze():
    out = {"design": {
        "model": "bias-free ReLU trunk width 64 depth 4 d_in 16, two linear heads, n=256, random ReLU teachers mixed to rho, full-batch GD on summed loss, float64",
        "rng": "numpy default_rng(1000 + s); inputs, teachers, W_0 shared across rho and eta within a seed (paired design)",
        "seeds_main": "s = 10..1009 (1000 seeds)", "seeds_sub": "s = 10..209 (200 seeds; a subset of main, so every comparison is paired)",
        "checkpoints": "linspace(0, steps-1, 5): 0,99,199,299,399 (T=400) or 0,499,999,1499,1999 (T=2000)",
        "conflict_cells": "fraction of checkpoint cells with sum_l a_l < 0 (= negative trunk cosine); the census conflict rate",
        "conflict_run": "fraction of (run, step) pairs over ALL steps with negative trunk cosine",
        "mixed": "cell whose a_l = <g_1^l, g_2^l> have strictly opposite signs across the 4 trunk layers",
        "ci": "cluster bootstrap over seeds, 2000 resamples (500 for medians), percentile 95% CI",
    }}
    main = load("main_eta0.05_T400")
    print("== main census: 1000 seeds ==")
    cs = cell_stats(main); rs = run_stats(main)
    out["main"] = {"cells": cs, "run": rs}
    # dependence on |rho|: sort by |rho|
    out["main"]["mixed_vs_abs_rho"] = sorted([(abs(float(r)), r, v["frac"], v["ci95"]) for r, v in cs["mixed_by_rho"].items()])
    out["main"]["mixed_post_init_vs_abs_rho"] = sorted([(abs(float(r)), r, v["frac"], v["ci95"]) for r, v in cs["mixed_by_rho_post_init"].items()])
    # replicate-size variability: 200 disjoint blocks of 5 seeds -> distribution of the 125-cell mixed count and conflict rate
    a = main["ck_a"]; valid = np.isfinite(a).all(-1)
    mixed = (a < 0).any(-1) & (a > 0).any(-1) & valid; conf = (a.sum(-1) < 0) & valid
    blocks_m = mixed.reshape(200, 5, 5, 5).sum((1, 2, 3)); blocks_c = conf.reshape(200, 5, 5, 5).mean((1, 2, 3))
    blocks_mpi = mixed[:, :, 1:].reshape(200, 5, 5, 4).sum((1, 2, 3)); blocks_cpi = conf[:, :, 1:].reshape(200, 5, 5, 4).mean((1, 2, 3))
    out["main"]["five_seed_blocks"] = {
        "n_blocks": 200,
        "mixed_of_125": {"mean": float(blocks_m.mean()), "sd": float(blocks_m.std()), "min": int(blocks_m.min()), "max": int(blocks_m.max()),
                          "q05": float(np.percentile(blocks_m, 5)), "q95": float(np.percentile(blocks_m, 95)),
                          "frac_blocks_ge_52": float((blocks_m >= 52).mean())},
        "mixed_post_init_of_100": {"mean": float(blocks_mpi.mean()), "sd": float(blocks_mpi.std()), "min": int(blocks_mpi.min()), "max": int(blocks_mpi.max()),
                                    "frac_blocks_ge_38": float((blocks_mpi >= 38).mean())},
        "conflict_cells": {"mean": float(blocks_c.mean()), "sd": float(blocks_c.std()), "min": float(blocks_c.min()), "max": float(blocks_c.max()),
                            "q05": float(np.percentile(blocks_c, 5)), "q95": float(np.percentile(blocks_c, 95))},
        "conflict_cells_post_init": {"mean": float(blocks_cpi.mean()), "sd": float(blocks_cpi.std()), "min": float(blocks_cpi.min()), "max": float(blocks_cpi.max())},
    }
    print(f"mixed all {pct(cs['mixed_all'])}, post-init {pct(cs['mixed_post_init'])}")
    print(f"conflict cells {pct(cs['conflict_cells_all'])}, post-init {pct(cs['conflict_cells_post_init'])}")
    print("conflict by step", {k: round(100 * v['frac'], 1) for k, v in cs['conflict_by_step'].items()})
    print("conflict by rho", {k: round(100 * v['frac'], 1) for k, v in cs['conflict_by_rho'].items()})
    print("mixed by rho", {k: round(100 * v['frac'], 1) for k, v in cs['mixed_by_rho'].items()})
    print(f"conflict over run {pct(rs['conflict_run_all'])}; even {pct(rs['conflict_run_even_steps'])} odd {pct(rs['conflict_run_odd_steps'])}")
    print("conflict by decile", [round(100 * v['frac'], 1) for v in rs['conflict_run_by_decile']])
    print(f"|cos| mixed {cs['abs_cos_mixed']['median']:.3f} vs single {cs['abs_cos_single_sign']['median']:.3f}")
    print("5-seed blocks mixed/125:", out["main"]["five_seed_blocks"]["mixed_of_125"])

    # ---- eta regime
    print("== eta regime (200 seeds) ==")
    S_sub = len(SEEDS_SUB)
    main_sub = {k: (v[:S_sub] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == len(SEEDS_MAIN) else v) for k, v in main.items()}
    eta_cfgs = [("eta0.01_T400", "eta0.01_T400"), ("eta0.02_T400", "eta0.02_T400"), ("eta0.05_T400", None), ("eta0.1_T400", "eta0.1_T400"), ("eta0.01_T2000", "eta0.01_T2000")]
    out["eta"] = {}
    for label, name in eta_cfgs:
        d = main_sub if name is None else load(name)
        cs_e = cell_stats(d); rs_e = run_stats(d)
        ml = matched_loss(main_sub, d, S_sub) if name is not None else None
        # final loss quantiles
        Lf = np.nan_to_num(d["ck_loss"][:, :, -1].sum(-1), nan=np.inf)
        out["eta"][label] = {"cells": cs_e, "run": rs_e, "matched_loss_vs_eta0.05_T400": ml,
                             "final_summed_loss": {"median": float(np.median(Lf)), "q25": float(np.percentile(Lf, 25)), "q75": float(np.percentile(Lf, 75))},
                             "final_task_losses_median": np.nanmedian(d["ck_loss"][:, :, -1], axis=(0, 1)).tolist()}
        print(f"{label}: mixed {pct(cs_e['mixed_all'])} post-init {pct(cs_e['mixed_post_init'])}; conflict cells {pct(cs_e['conflict_cells_all'])} "
              f"post-init {pct(cs_e['conflict_cells_post_init'])}; run {pct(rs_e['conflict_run_all'])}; by step {[round(100 * v['frac']) for v in cs_e['conflict_by_step'].values()]}; "
              f"by rho {[round(100 * v['frac']) for v in cs_e['conflict_by_rho'].values()]}; final loss {np.median(Lf):.4f}; diverged {cs_e['diverged_runs']}")
        if ml:
            print(f"   matched loss: reached {pct(ml['frac_runs_reaching_target'])} at median step {ml['median_step_at_match']}, conflict there {pct(ml['conflict_at_matched_loss_raw'])}, window {ml['conflict_at_matched_loss_20step_window']}")

    # ---- rho near 1
    print("== rho near 1 (200 seeds) ==")
    out["near1"] = {}
    for label, name, src07 in [("eta0.05_T400", "near1_eta0.05_T400", main_sub), ("eta0.01_T400", "near1_eta0.01_T400", load("eta0.01_T400")),
                               ("eta0.01_T2000", "near1_eta0.01_T2000", load("eta0.01_T2000"))]:
        d = load(name)
        cs_n = cell_stats(d); rs_n = run_stats(d)
        # rho = 0.7 from the matching main/eta config (same seeds)
        r07 = int(np.where(np.isclose(src07["rhos"], 0.7))[0][0])
        cs07 = cell_stats(src07, rho_sel=[r07]); rs07 = run_stats(src07, rho_sel=[r07])
        out["near1"][label] = {"cells": cs_n, "run": rs_n, "rho0.7_cells": cs07, "rho0.7_run": rs07}
        allr = {"0.7": (cs07, rs07)}
        for r in d["rhos"]:
            allr[str(r)] = (cs_n, rs_n)
        line = []
        for r, (c, rr) in allr.items():
            line.append(f"rho {r}: conflict all {100 * c['conflict_by_rho'][r]['frac']:.0f} post-init {100 * c['conflict_by_rho_post_init'][r]['frac']:.0f} "
                        f"[{100 * c['conflict_by_rho_post_init'][r]['ci95'][0]:.0f},{100 * c['conflict_by_rho_post_init'][r]['ci95'][1]:.0f}] "
                        f"run {100 * rr['conflict_run_by_rho'][r]['frac']:.0f} mixed {100 * c['mixed_by_rho'][r]['frac']:.0f} cos_med {c['cos_median_by_rho_post_init'][r]['median']:+.2f} "
                        f"by step {[round(100 * v['frac']) for v in c['conflict_by_rho_step'][r].values()]}")
        print(label + "\n   " + "\n   ".join(line))

    with open(os.path.join(HERE, f"{KEY}.json"), "w") as f:
        json.dump(out, f, indent=1, allow_nan=False, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else (None if (isinstance(o, float) and not np.isfinite(o)) else str(o)))
    # combined per-cell npz (checkpoint-level arrays for every config; per-step cos/loss/code for every config)
    comb = {}
    for name in CONFIGS:
        d = load(name)
        for k in ("seeds", "rhos", "ckpts", "ck_a", "ck_g1n2", "ck_g2n2", "ck_loss", "ck_cos", "ck_fro", "diverged_at", "code", "cos"):
            comb[f"{name}/{k}"] = d[k]
        comb[f"{name}/loss_summed_step"] = d["loss"].sum(-1)
    np.savez_compressed(os.path.join(HERE, f"{KEY}_cells.npz"), **comb)
    print("wrote", os.path.join(HERE, f"{KEY}.json"), "and", os.path.join(HERE, f"{KEY}_cells.npz"))


def plot():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = os.path.join(HERE, "..", "figures"); os.makedirs(figdir, exist_ok=True)
    res = json.load(open(os.path.join(HERE, f"{KEY}.json")))
    # (1) per-step conflict rate for each eta (200 seeds) with the checkpoint cells marked
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for label, col in zip(["eta0.01_T400", "eta0.02_T400", "eta0.05_T400", "eta0.1_T400"], ["C0", "C1", "C2", "C3"]):
        e = res["eta"][label]
        curve = np.array(e["run"]["conflict_per_step_curve"])
        axes[0].plot(curve, color=col, lw=0.8, alpha=0.9, label=label.replace("_T400", "").replace("eta", "eta="))
        ck = e["cells"]["checkpoints"]
        axes[0].plot(ck, [e["cells"]["conflict_by_step"][str(k)]["frac"] for k in ck], "o", color=col, ms=4)
    axes[0].set_xlabel("step"); axes[0].set_ylabel("fraction of runs with cos < 0"); axes[0].set_title("400 steps, 200 seeds x 5 rho; dots = census checkpoints")
    axes[0].legend(fontsize=8)
    e = res["eta"]["eta0.01_T2000"]
    curve = np.array(e["run"]["conflict_per_step_curve"]); ck = e["cells"]["checkpoints"]
    axes[1].plot(curve, color="C0", lw=0.8, label="eta=0.01, 2000 steps")
    axes[1].plot(ck, [e["cells"]["conflict_by_step"][str(k)]["frac"] for k in ck], "o", color="C0", ms=4)
    ax2 = axes[1].twinx(); ax2.semilogy(e["run"]["median_summed_loss_per_step"], color="gray", lw=0.8, ls="--", label="median summed loss")
    ax2.set_ylabel("median summed loss")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("fraction of runs with cos < 0"); axes[1].legend(loc="center right", fontsize=8)
    axes[1].set_title("slow regime, long run")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, f"{KEY}_conflict_vs_step.png"), dpi=150); plt.close(fig)
    # (2) mixed fraction and conflict by rho (main, 1000 seeds) and near-1 post-init conflict by rho per regime
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    cs = res["main"]["cells"]
    rhos = [float(r) for r in cs["mixed_by_rho"]]
    for key, lab, col in [("mixed_by_rho", "mixed, all cells", "C0"), ("mixed_by_rho_post_init", "mixed, post-init", "C1"),
                          ("conflict_by_rho", "conflict (cells), all", "C2"), ("conflict_by_rho_post_init", "conflict (cells), post-init", "C3")]:
        v = [cs[key][str(r)] for r in cs[key]]
        m = np.array([x["frac"] for x in v]); lo = np.array([x["ci95"][0] for x in v]); hi = np.array([x["ci95"][1] for x in v])
        axes[0].errorbar(rhos, m, yerr=[m - lo, hi - m], fmt="o-", color=col, ms=4, capsize=2, label=lab)
    axes[0].set_xlabel("rho"); axes[0].set_ylabel("fraction"); axes[0].set_title("main census, 1000 seeds (95% CI)"); axes[0].legend(fontsize=7)
    # sign patterns
    pats = cs["sign_patterns"]; labs = list(pats); vals = [pats[k]["post_init"]["frac"] for k in labs]
    order = np.argsort(vals)[::-1]
    axes[1].bar(range(16), [vals[i] for i in order]); axes[1].set_xticks(range(16)); axes[1].set_xticklabels([labs[i] for i in order], rotation=90, fontsize=7)
    axes[1].set_ylabel("fraction of post-init cells"); axes[1].set_title("sign pattern of (a_1..a_4), post-init")
    for regime, col in [("eta0.05_T400", "C2"), ("eta0.01_T400", "C0"), ("eta0.01_T2000", "C4")]:
        n = res["near1"][regime]
        rr = ["0.7", "0.9", "0.99", "1.0"]
        v = [n["rho0.7_cells"]["conflict_by_rho_post_init"]["0.7"]] + [n["cells"]["conflict_by_rho_post_init"][r] for r in rr[1:]]
        m = np.array([x["frac"] for x in v]); lo = np.array([x["ci95"][0] for x in v]); hi = np.array([x["ci95"][1] for x in v])
        axes[2].errorbar(range(4), m, yerr=[m - lo, hi - m], fmt="o-", color=col, ms=4, capsize=2, label=regime)
    axes[2].set_xticks(range(4)); axes[2].set_xticklabels(["0.7", "0.9", "0.99", "1.0"]); axes[2].set_xlabel("rho"); axes[2].set_ylabel("post-init conflict rate (cells)")
    axes[2].set_title("near-identical tasks, 200 seeds"); axes[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, f"{KEY}_by_rho.png"), dpi=150); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if stage == "verify":
        verify()
    elif stage == "run":
        run(sys.argv[2:] or None)
    elif stage == "analyze":
        analyze()
    elif stage == "plot":
        plot()
