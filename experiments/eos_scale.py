"""eos_scale: edge-of-stability mechanism replication at scale (200 seeds), with an exact HVP.

Measures the edge-of-stability mechanism on 200 seeds, with an exact Hessian:
  * Hessian-vector products by torch double backward in float64 (exact for the piecewise-quadratic
    loss inside the iterate's ReLU activation pattern; a central finite difference with eps = 1e-6
    straddles a kink in a few cells and inflates lambda_max there).
  * lambda_max by Lanczos (scipy eigsh, k = 2, tol 1e-10) with the eigenpair residual recorded, and,
    for comparison, the original 40-iteration power iteration on the same exact HVP.
Per (seed, rho, checkpoint) cell it records (see CELL_FIELDS): eta*lambda_max, lambda_2, the fraction
of ||g_1+g_2||^2 along v (full-parameter and trunk-block versions), the fraction of each task
gradient's squared norm along v_t, the trunk cosine raw / with v_t projected out / two-step averaged,
the period-2 indicator (sign flips of <g_sum(s), v> over 6 consecutive steps around the checkpoint),
the losses, per-layer a_l and gradient norms, and the norm ratios that make the "anti-aligned by
construction" argument checkable (||r_1 + r_2|| / min(||r_1||, ||r_2||) off v).

Design (seed offsets disjoint from the 0-9 used by the project scripts: rng = default_rng(1000 + s)):
  eta = 0.05, 400 steps: seeds 100-299, rho in {-0.5,-0.2,0,0.3,0.7}, checkpoints 0,99,199,299,399;
     seeds 100-149 additionally every 25 steps (24, 49, ..., 399).
  eta = 0.01, 400 steps: same.
  eta = 0.01, 2000 steps: seeds 100-199, checkpoints 0,99,199,299,399,499,999,1499,1999;
     seeds 100-149 additionally every 100 steps.
Writes eos_scale_cells.npz (all per-cell arrays) and eos_scale.json (summaries with seed-cluster
bootstrap 95% CIs). Run:  python eos_scale.py [--workers 3] [--quick]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, time, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from conjugation_check import make_tasks, init_net, gd_run, grads  # noqa: E402

RHOS = (-0.5, -0.2, 0.0, 0.3, 0.7)
CELL_FIELDS = ["config", "seed", "rho", "step", "eta", "eta_lambda_max", "lambda_max", "lambda_2", "eig_residual",
               "lambda_max_power40", "n_hvp", "loss1", "loss2", "loss_sum",
               "cos_trunk", "cos_trunk_minus_top", "cos_two_step_avg", "cos_next_step", "cos_full",
               "gsum_frac_along_v_full", "gsum_frac_along_vt_trunk", "f1_along_vt", "f2_along_vt", "same_side",
               "norm_g1", "norm_g2", "norm_gsum", "norm_r1", "norm_r2", "norm_rsum",
               "n_sign_flips_of_5", "window_start", "loss_nonmonotone_of_5",
               "a_l", "norm_g1_l", "norm_g2_l", "vt_frac_by_layer", "eig_all_same_pattern"]


def flat_trunk(gW_i):
    return np.concatenate([g.ravel() for g in gW_i])


def make_hvp(X, Ys, W, H):
    """Exact Hessian-vector product of the summed loss at (W, H) via torch double backward (float64).
    Returns (hvp(v) -> ndarray, flat gradient ndarray, counter list)."""
    import torch
    torch.set_num_threads(1)
    Xt = torch.tensor(X, dtype=torch.float64)
    Yt = [torch.tensor(y, dtype=torch.float64) for y in Ys]
    params = [torch.tensor(w, dtype=torch.float64, requires_grad=True) for w in W] + \
             [torch.tensor(h, dtype=torch.float64, requires_grad=True) for h in H]
    Wt, Ht = params[:len(W)], params[len(W):]
    h = Xt
    for w in Wt:
        h = torch.relu(h @ w)
    loss = sum(0.5 * torch.mean((h @ Ht[i] - Yt[i]) ** 2) for i in range(2))
    g = torch.autograd.grad(loss, params, create_graph=True)
    gflat = torch.cat([x.reshape(-1) for x in g])
    count = [0]

    def hvp(v):
        vt = torch.as_tensor(np.asarray(v, dtype=np.float64))
        hv = torch.autograd.grad(gflat @ vt, params, retain_graph=True)
        count[0] += 1
        return torch.cat([x.reshape(-1) for x in hv]).detach().numpy()
    return hvp, gflat.detach().numpy(), count


def top_eigs(hvp, P, tol=1e-10, seed=0):
    from scipy.sparse.linalg import LinearOperator, eigsh
    op = LinearOperator((P, P), matvec=hvp, dtype=np.float64)
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(P)
    vals, vecs = eigsh(op, k=2, which="LA", tol=tol, v0=v0, maxiter=2000)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    v = vecs[:, 0]
    res = float(np.linalg.norm(hvp(v) - vals[0] * v) / max(abs(vals[0]), 1e-300))
    return float(vals[0]), float(vals[1]), v, res


def power40(hvp, P, iters=40, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(P); v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        hv = hvp(v); lam = float(v @ hv); nrm = np.linalg.norm(hv)
        if nrm == 0:
            break
        v = hv / nrm
    return lam


def run_one(config, eta, steps, seed, ckpts, depth=4, width=64, d_in=16, n=256):
    """One seed: five rho values (paired inputs/teachers/W_0, as the project censuses), all checkpoints."""
    ntr = d_in * width + (depth - 1) * width * width
    out = []
    for rho in RHOS:
        rng = np.random.default_rng(1000 + seed)
        X, Ys = make_tasks(rng, n, d_in, rho)
        W, H = init_net(rng, d_in, width, depth, scheme="he")
        hist = gd_run(X, Ys, W, H, [eta] * depth, eta, steps, record_every=1)
        loss_hist = np.array(hist["loss"])                       # (steps+1, 2)

        def iterate(k):
            return (hist["W"][k], hist["H"][k]) if k < steps else (hist["final_W"], hist["final_H"])

        gcache = {}

        def task_grads(k):
            if k not in gcache:
                _, gW, gH = grads(X, Ys, *iterate(k))
                g1 = flat_trunk(gW[0]); g2 = flat_trunk(gW[1])
                gfull = np.concatenate([g1 + g2, (gH[0][0] + gH[1][0]).ravel(), (gH[0][1] + gH[1][1]).ravel()])
                gcache[k] = (gW, g1, g2, gfull)
            return gcache[k]

        for k in ckpts:
            Wk, Hk = iterate(k)
            gW, g1, g2, gfull = task_grads(k)
            hvp, gflat_torch, count = make_hvp(X, Ys, Wk, Hk)
            assert np.allclose(gflat_torch, gfull, rtol=1e-10, atol=1e-14), "torch gradient != numpy gradient"
            P = gfull.size
            lam1, lam2, v, res = top_eigs(hvp, P, seed=seed)
            n_hvp_eig = count[0]
            lam_p40 = power40(hvp, P, seed=0)
            vt = v[:ntr]; vt_norm = np.linalg.norm(vt); vt = vt / vt_norm
            gsum = g1 + g2
            f1 = float((g1 @ vt) ** 2 / (g1 @ g1)); f2 = float((g2 @ vt) ** 2 / (g2 @ g2))
            r1 = g1 - (g1 @ vt) * vt; r2 = g2 - (g2 @ vt) * vt; rsum = r1 + r2
            cos0 = float(g1 @ g2 / (np.linalg.norm(g1) * np.linalg.norm(g2)))
            cos1 = float(r1 @ r2 / (np.linalg.norm(r1) * np.linalg.norm(r2)))
            # full-parameter task gradients (trunk + own head) cosine, for reference (head cross term is 0)
            _, _, gH_k = grads(X, Ys, Wk, Hk)
            nh1 = np.linalg.norm(gH_k[0][0]); nh2 = np.linalg.norm(gH_k[1][1])
            cos_full = float(g1 @ g2 / (np.sqrt(g1 @ g1 + nh1 ** 2) * np.sqrt(g2 @ g2 + nh2 ** 2)))
            # two-step average with the next iterate
            _, g1n, g2n, _ = task_grads(k + 1)
            b1, b2 = 0.5 * (g1 + g1n), 0.5 * (g2 + g2n)
            cos_avg = float(b1 @ b2 / (np.linalg.norm(b1) * np.linalg.norm(b2)))
            cos_next = float(g1n @ g2n / (np.linalg.norm(g1n) * np.linalg.norm(g2n)))
            # period-2 indicator: signs of <g_sum(s), v(t)> for 6 consecutive steps containing t
            w0 = min(max(k - 4, 0), steps - 5)
            projs = np.array([task_grads(s)[3] @ v for s in range(w0, w0 + 6)])
            flips = int((np.sign(projs[1:]) != np.sign(projs[:-1])).sum())
            ls = loss_hist[w0:w0 + 6].sum(1)
            nonmono = int((np.diff(ls) > 0).sum())
            # layer quantities
            a_l = np.array([float(np.sum(gW[0][l] * gW[1][l])) for l in range(depth)])
            n1_l = np.array([float(np.linalg.norm(gW[0][l])) for l in range(depth)])
            n2_l = np.array([float(np.linalg.norm(gW[1][l])) for l in range(depth)])
            sizes = [d_in * width] + [width * width] * (depth - 1)
            vt_by_layer = []; off = 0
            for m in sizes:
                vt_by_layer.append(float(np.sum(vt[off:off + m] ** 2))); off += m
            out.append(dict(config=config, seed=seed, rho=rho, step=k, eta=eta, eta_lambda_max=eta * lam1, lambda_max=lam1,
                            lambda_2=lam2, eig_residual=res, lambda_max_power40=lam_p40, n_hvp=n_hvp_eig,
                            loss1=float(loss_hist[k, 0]), loss2=float(loss_hist[k, 1]), loss_sum=float(loss_hist[k].sum()),
                            cos_trunk=cos0, cos_trunk_minus_top=cos1, cos_two_step_avg=cos_avg, cos_next_step=cos_next,
                            cos_full=cos_full,
                            gsum_frac_along_v_full=float((gfull @ v) ** 2 / (gfull @ gfull)),
                            gsum_frac_along_vt_trunk=float((gsum @ vt) ** 2 / (gsum @ gsum)),
                            f1_along_vt=f1, f2_along_vt=f2, same_side=int(np.sign(g1 @ vt) == np.sign(g2 @ vt)),
                            norm_g1=float(np.linalg.norm(g1)), norm_g2=float(np.linalg.norm(g2)), norm_gsum=float(np.linalg.norm(gsum)),
                            norm_r1=float(np.linalg.norm(r1)), norm_r2=float(np.linalg.norm(r2)), norm_rsum=float(np.linalg.norm(rsum)),
                            n_sign_flips_of_5=flips, window_start=w0, loss_nonmonotone_of_5=nonmono,
                            a_l=a_l, norm_g1_l=n1_l, norm_g2_l=n2_l, vt_frac_by_layer=np.array(vt_by_layer),
                            eig_all_same_pattern=float(vt_norm ** 2)))
            del hvp
    return out


# ----------------------------------------------------------------------------- summaries
def boot_ci(vals, seeds, stat, n_boot=2000, rng_seed=0):
    """Seed-cluster bootstrap 95% CI of stat(vals) (vals: 1-D array over cells; seeds: seed id per cell)."""
    vals = np.asarray(vals, float); seeds = np.asarray(seeds)
    us = np.unique(seeds)
    idx = {u: np.where(seeds == u)[0] for u in us}
    rng = np.random.default_rng(rng_seed)
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(us, size=us.size, replace=True)
        sel = np.concatenate([idx[u] for u in pick])
        bs.append(stat(vals[sel]))
    return [float(stat(vals)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def frac_neg(x):
    return float(np.mean(x < 0))


def summarize(cells):
    """cells: dict of arrays (from the npz). Returns the JSON summary."""
    cfg = cells["config"]; st = cells["step"]; sd = cells["seed"]
    med = np.median
    out = {}
    for c in np.unique(cfg):
        m = cfg == c
        steps = np.unique(st[m])
        main_steps = [s for s in (99, 199, 299, 399) if s in steps]
        late_steps = [s for s in (199, 299, 399) if s in steps]
        res = {"n_cells": int(m.sum()), "n_seeds": int(np.unique(sd[m]).size), "steps": [int(s) for s in steps],
               "eta": float(cells["eta"][m][0])}

        def blk(mask):
            b = {"n": int(mask.sum())}
            if mask.sum() == 0:
                return b
            s_ = sd[mask]
            b["conflict_raw"] = boot_ci(cells["cos_trunk"][mask], s_, frac_neg)
            b["conflict_minus_top"] = boot_ci(cells["cos_trunk_minus_top"][mask], s_, frac_neg)
            b["conflict_two_step_avg"] = boot_ci(cells["cos_two_step_avg"][mask], s_, frac_neg)
            b["conflict_next_step"] = boot_ci(cells["cos_next_step"][mask], s_, frac_neg)
            b["sign_changes_after_removal_frac"] = boot_ci((np.sign(cells["cos_trunk"][mask]) != np.sign(cells["cos_trunk_minus_top"][mask])).astype(float), s_, np.mean)
            b["sign_agreement_two_step_vs_minus_top"] = boot_ci((np.sign(cells["cos_two_step_avg"][mask]) == np.sign(cells["cos_trunk_minus_top"][mask])).astype(float), s_, np.mean)
            b["median_cos_raw"] = boot_ci(cells["cos_trunk"][mask], s_, med)
            b["median_cos_minus_top"] = boot_ci(cells["cos_trunk_minus_top"][mask], s_, med)
            b["median_cos_two_step_avg"] = boot_ci(cells["cos_two_step_avg"][mask], s_, med)
            b["eta_lambda_max_median"] = boot_ci(cells["eta_lambda_max"][mask], s_, med)
            b["eta_lambda_max_q10"] = boot_ci(cells["eta_lambda_max"][mask], s_, lambda x: np.percentile(x, 10))
            b["eta_lambda_max_q90"] = boot_ci(cells["eta_lambda_max"][mask], s_, lambda x: np.percentile(x, 90))
            b["frac_eta_lambda_in_1.8_2.1"] = boot_ci(((cells["eta_lambda_max"][mask] >= 1.8) & (cells["eta_lambda_max"][mask] <= 2.1)).astype(float), s_, np.mean)
            b["frac_eta_lambda_above_2.1"] = boot_ci((cells["eta_lambda_max"][mask] > 2.1).astype(float), s_, np.mean)
            b["gsum_frac_along_v_full_median"] = boot_ci(cells["gsum_frac_along_v_full"][mask], s_, med)
            b["gsum_frac_along_v_full_q10"] = boot_ci(cells["gsum_frac_along_v_full"][mask], s_, lambda x: np.percentile(x, 10))
            b["gsum_frac_along_vt_trunk_median"] = boot_ci(cells["gsum_frac_along_vt_trunk"][mask], s_, med)
            b["gsum_frac_along_vt_trunk_q10"] = boot_ci(cells["gsum_frac_along_vt_trunk"][mask], s_, lambda x: np.percentile(x, 10))
            b["f1_along_vt_median"] = boot_ci(cells["f1_along_vt"][mask], s_, med)
            b["f2_along_vt_median"] = boot_ci(cells["f2_along_vt"][mask], s_, med)
            b["f1_along_vt_q90"] = boot_ci(cells["f1_along_vt"][mask], s_, lambda x: np.percentile(x, 90))
            b["same_side_frac"] = boot_ci(cells["same_side"][mask].astype(float), s_, np.mean)
            b["gsum_over_g1_norm_median"] = boot_ci(cells["norm_gsum"][mask] / cells["norm_g1"][mask], s_, med)
            b["stationarity_ratio_median"] = boot_ci(cells["norm_gsum"][mask] / (cells["norm_g1"][mask] + cells["norm_g2"][mask]), s_, med)
            b["off_v_cancellation_ratio_median"] = boot_ci(cells["norm_rsum"][mask] / np.minimum(cells["norm_r1"][mask], cells["norm_r2"][mask]), s_, med)
            b["off_v_cancellation_ratio_q90"] = boot_ci(cells["norm_rsum"][mask] / np.minimum(cells["norm_r1"][mask], cells["norm_r2"][mask]), s_, lambda x: np.percentile(x, 90))
            b["period2_frac_5of5_flips"] = boot_ci((cells["n_sign_flips_of_5"][mask] == 5).astype(float), s_, np.mean)
            b["period2_frac_ge4_flips"] = boot_ci((cells["n_sign_flips_of_5"][mask] >= 4).astype(float), s_, np.mean)
            b["loss_nonmonotone_frac_ge2_of_5"] = boot_ci((cells["loss_nonmonotone_of_5"][mask] >= 2).astype(float), s_, np.mean)
            b["loss_sum_median"] = boot_ci(cells["loss_sum"][mask], s_, med)
            b["eig_residual_max"] = float(cells["eig_residual"][mask].max())
            b["power40_rel_err_median"] = float(med(np.abs(cells["lambda_max_power40"][mask] - cells["lambda_max"][mask]) / cells["lambda_max"][mask]))
            b["power40_rel_err_q90"] = float(np.percentile(np.abs(cells["lambda_max_power40"][mask] - cells["lambda_max"][mask]) / cells["lambda_max"][mask], 90))
            b["power40_rel_err_max"] = float((np.abs(cells["lambda_max_power40"][mask] - cells["lambda_max"][mask]) / cells["lambda_max"][mask]).max())
            b["gap_ratio_lambda2_over_lambda1_median"] = float(med(cells["lambda_2"][mask] / cells["lambda_max"][mask]))
            b["gap_ratio_q90"] = float(np.percentile(cells["lambda_2"][mask] / cells["lambda_max"][mask], 90))
            b["n_hvp_eigsh_median"] = float(med(cells["n_hvp"][mask]))
            return b

        res["all"] = blk(m)
        res["post_init_main"] = blk(m & np.isin(st, main_steps))
        res["late_199_399"] = blk(m & np.isin(st, late_steps))
        res["step0"] = blk(m & (st == 0))
        res["by_step"] = {int(s): blk(m & (st == s)) for s in steps}
        res["by_rho_post_init_main"] = {str(r): blk(m & np.isin(st, main_steps) & (cells["rho"] == r)) for r in RHOS}
        # (e) conflict vs loss reached and vs stationarity, post-init cells
        pm = m & np.isin(st, main_steps)
        if pm.sum() > 0:
            L = cells["loss_sum"][pm]; C = cells["cos_trunk"][pm]; R = cells["norm_gsum"][pm] / (cells["norm_g1"][pm] + cells["norm_g2"][pm])
            qs = np.percentile(L, [0, 20, 40, 60, 80, 100])
            bins = []
            for i in range(5):
                mm = (L >= qs[i]) & (L <= qs[i + 1]) if i == 4 else (L >= qs[i]) & (L < qs[i + 1])
                bins.append({"loss_range": [float(qs[i]), float(qs[i + 1])], "n": int(mm.sum()),
                             "conflict": float(np.mean(C[mm] < 0)), "median_cos": float(med(C[mm])),
                             "median_stationarity_ratio": float(med(R[mm])), "median_eta_lambda": float(med(cells["eta_lambda_max"][pm][mm]))})
            from scipy.stats import spearmanr
            res["conflict_vs_loss_quintiles_post_init"] = bins
            res["spearman_cos_vs_loss_post_init"] = [float(x) for x in spearmanr(C, L)]
            res["spearman_cos_vs_stationarity_post_init"] = [float(x) for x in spearmanr(C, R)]
            res["spearman_cos_vs_eta_lambda_post_init"] = [float(x) for x in spearmanr(C, cells["eta_lambda_max"][pm])]
            # conflict vs stationarity quintiles
            qs = np.percentile(R, [0, 20, 40, 60, 80, 100]); bins = []
            for i in range(5):
                mm = (R >= qs[i]) & (R <= qs[i + 1]) if i == 4 else (R >= qs[i]) & (R < qs[i + 1])
                bins.append({"ratio_range": [float(qs[i]), float(qs[i + 1])], "n": int(mm.sum()),
                             "conflict": float(np.mean(C[mm] < 0)), "median_cos": float(med(C[mm])), "median_loss": float(med(L[mm]))})
            res["conflict_vs_stationarity_quintiles_post_init"] = bins
        out[str(c)] = res
    return out


def to_npz_dict(rows):
    d = {}
    for f in CELL_FIELDS:
        vals = [r[f] for r in rows]
        if isinstance(vals[0], np.ndarray):
            d[f] = np.stack(vals)
        elif isinstance(vals[0], str):
            d[f] = np.array(vals)
        else:
            d[f] = np.array(vals, dtype=float if isinstance(vals[0], float) else (int if isinstance(vals[0], (int, np.integer)) else float))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="timing run: 2 seeds per config")
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--seeds_long", type=int, default=100)
    ap.add_argument("--seeds_dense", type=int, default=50)
    ap.add_argument("--phase2", action="store_true", help="run only the phase-2 top-k decomposition")
    args = ap.parse_args()
    if args.phase2:
        return phase2_main()
    from joblib import Parallel, delayed

    S0 = 100
    main_ck = [0, 99, 199, 299, 399]
    dense_ck = sorted(set(main_ck) | set(range(24, 400, 25)))
    long_ck = [0, 99, 199, 299, 399, 499, 999, 1499, 1999]
    long_dense = sorted(set(long_ck) | set(range(99, 2000, 100)))
    nS, nL, nD = (2, 2, 1) if args.quick else (args.seeds, args.seeds_long, args.seeds_dense)
    jobs = []
    for eta in (0.05, 0.01):
        for s in range(S0, S0 + nS):
            ck = dense_ck if s < S0 + nD else main_ck
            jobs.append((f"eta{eta}_400", eta, 400, s, ck))
    for s in range(S0, S0 + nL):
        ck = long_dense if s < S0 + nD else long_ck
        jobs.append(("eta0.01_2000", 0.01, 2000, s, ck))
    # longest jobs first for load balance
    jobs.sort(key=lambda j: -(j[2] + 40 * len(j[4])))
    print(f"{len(jobs)} jobs, {args.workers} workers", flush=True)
    t0 = time.time()
    results = Parallel(n_jobs=args.workers, verbose=5, batch_size=1)(delayed(run_one)(*j) for j in jobs)
    rows = [r for rr in results for r in rr]
    runtime_min = (time.time() - t0) / 60
    print(f"{len(rows)} cells in {runtime_min:.1f} min", flush=True)
    cells = to_npz_dict(rows)
    tag = "_quick" if args.quick else ""
    np.savez_compressed(os.path.join(HERE, f"eos_scale{tag}_cells.npz"), **cells)
    summ = summarize(cells)
    summ["_meta"] = {"runtime_min": runtime_min, "n_cells": len(rows), "seeds": f"{S0}-{S0 + nS - 1} (400-step configs), {S0}-{S0 + nL - 1} (2000-step), dense checkpoints on {S0}-{S0 + nD - 1}",
                     "rng": "numpy default_rng(1000 + seed); inputs, teachers, W_0 shared across rho within a seed (paired design, as the project censuses)",
                     "hvp": "torch double backward, float64, exact inside the ReLU activation pattern", "eig": "scipy eigsh k=2 which=LA tol=1e-10; residual ||Hv - lambda v||/lambda recorded",
                     "ci": "seed-cluster percentile bootstrap, 2000 resamples"}
    json.dump(summ, open(os.path.join(HERE, f"eos_scale{tag}.json"), "w"), indent=1, allow_nan=False)
    print("wrote eos_scale json/npz", flush=True)




# ----------------------------------------------------------------------------- phase 2 (top-k)
# Ran 2026-09-01 as a separate script (verbatim below, imports resolved at call time); results are
# merged into eos_scale.json under key "phase2_topk" and into eos_scale_cells.npz as phase2_* arrays.
# Invoke with:  python eos_scale.py --phase2
def run_cell(config, eta, steps, seed, rho, k=20, depth=4, width=64, d_in=16, n=256):
    from scipy.sparse.linalg import LinearOperator, eigsh
    rng = np.random.default_rng(1000 + seed)
    X, Ys = make_tasks(rng, n, d_in, rho)
    W, H = init_net(rng, d_in, width, depth, scheme="he")
    hist = gd_run(X, Ys, W, H, [eta] * depth, eta, steps, record_every=steps)
    Wk, Hk = hist["final_W"], hist["final_H"]
    _, gW, gH = grads(X, Ys, Wk, Hk)
    g1, g2 = flat_trunk(gW[0]), flat_trunk(gW[1])
    gsum = g1 + g2
    g1f = np.concatenate([g1, gH[0][0].ravel(), np.zeros(gH[0][1].size)])
    g2f = np.concatenate([g2, np.zeros(gH[1][0].size), gH[1][1].ravel()])
    ntr = g1.size
    hvp, gflat, _ = make_hvp(X, Ys, Wk, Hk)
    P = gflat.size
    op = LinearOperator((P, P), matvec=hvp, dtype=np.float64)
    v0 = np.random.default_rng(seed).standard_normal(P)
    vals, vecs = eigsh(op, k=k, which="LA", tol=1e-8, v0=v0, maxiter=4000)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    gsumf = gflat
    cos_after, frac_sum, frac_1, frac_2 = [], [], [], []
    for j in range(k + 1):
        if j == 0:
            r1, r2 = g1, g2
            frac_sum.append(0.0); frac_1.append(0.0); frac_2.append(0.0)
        else:
            Vt = vecs[:ntr, :j]
            Q, _ = np.linalg.qr(Vt)  # orthonormal basis of trunk blocks
            r1 = g1 - Q @ (Q.T @ g1); r2 = g2 - Q @ (Q.T @ g2)
            Vf = vecs[:, :j]
            frac_sum.append(float(np.sum((Vf.T @ gsumf) ** 2) / (gsumf @ gsumf)))
            frac_1.append(float(np.sum((Vf.T @ g1f) ** 2) / (g1f @ g1f)))
            frac_2.append(float(np.sum((Vf.T @ g2f) ** 2) / (g2f @ g2f)))
        cos_after.append(float(r1 @ r2 / (np.linalg.norm(r1) * np.linalg.norm(r2))))
    return dict(config=config, seed=seed, rho=rho, step=steps - 1,
                eta_lambda=(eta * vals).tolist(), cos_after_topj=cos_after,
                gsum_frac_topj=frac_sum, g1_frac_topj=frac_1, g2_frac_topj=frac_2,
                loss_sum=float(np.array(hist["loss"])[-1].sum()))



def phase2_main():
    from joblib import Parallel, delayed
    cfgs = [("eta0.05_400", 0.05, 400), ("eta0.01_400", 0.01, 400), ("eta0.01_2000", 0.01, 2000)]
    jobs = [(c, e, s, seed, rho) for (c, e, s) in cfgs for seed in range(100, 115) for rho in RHOS]
    t0 = time.time()
    rows = Parallel(n_jobs=3, verbose=5, batch_size=1)(delayed(run_cell)(*j) for j in jobs)
    print(f"{len(rows)} cells in {(time.time()-t0)/60:.1f} min", flush=True)
    # summarize: median over cells of cos_after and fracs at each j, conflict fraction at each j
    summ = {}
    for c, _, _ in cfgs:
        rr = [r for r in rows if r["config"] == c]
        A = np.array([r["cos_after_topj"] for r in rr])       # (75, 21)
        F = np.array([r["gsum_frac_topj"] for r in rr])
        F1 = np.array([r["g1_frac_topj"] for r in rr])
        EL = np.array([r["eta_lambda"] for r in rr])
        summ[c] = {"n_cells": len(rr), "checkpoint": rr[0]["step"],
                   "median_cos_after_topj_0_to_20": np.median(A, 0).round(4).tolist(),
                   "conflict_after_topj_0_to_20": (A < 0).mean(0).round(4).tolist(),
                   "median_gsum_frac_topj": np.median(F, 0).round(5).tolist(),
                   "median_g1_frac_topj": np.median(F1, 0).round(5).tolist(),
                   "median_eta_lambda_1_to_20": np.median(EL, 0).round(4).tolist()}
    summ["_meta"] = {"seeds": "100-114", "k": 20, "eig": "eigsh k=20 LA tol=1e-8 on exact torch HVP",
                     "projection": "trunk blocks of top-j full-parameter eigenvectors, orthonormalized (QR), projected out of both trunk task gradients",
                     "runtime_min": (time.time() - t0) / 60}
    json.dump(summ, open(os.path.join(HERE, "eos_scale_phase2_tmp.json"), "w"), indent=1, allow_nan=False)
    np.savez_compressed(os.path.join(HERE, "eos_scale_phase2_tmp.npz"),
                        **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    print("wrote phase2 tmp files", flush=True)

if __name__ == "__main__":
    main()
