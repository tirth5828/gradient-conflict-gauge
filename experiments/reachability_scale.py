"""reachability_scale.py -- induced-gauge reachability at scale (Sec. III "Coordinates" of the
abstract replicated on 500 fresh seeds with cluster-bootstrap 95% CIs).

Design (identical to conjugation_check.sign_census, only the seeds change):
  * seeds 10..509 (rng stream default_rng(1000 + s), disjoint from the development seeds 0-9),
    rho in {-0.5, -0.2, 0, 0.3, 0.7} (paired: inputs, teachers, W_0 shared across rho inside a
    seed), He init, eta = 0.05, 400 full-batch GD steps, checkpoints 0, 99, 199, 299, 399;
    12 500 cells from 500 independent (inputs, teachers, W_0) draws.
  * per cell (base run): a_l = <g_1^l, g_2^l> per trunk layer, per-layer squared gradient norms,
    layer Frobenius norms, both losses, the trunk cosine, the head cross term, and the neuron-wise
    incoming column sums b_{l,j} = sum_i P_l[i,j] (all 4 layers) and outgoing row sums
    r_{l,j} = sum_k P_{l+1}[j,k] (layers 0..2) of P_l = g_1^l * g_2^l (elementwise), so the nearest
    nearest single-neuron flip can be recomputed from the npz.
  * three further trajectories per (seed, rho), TRAINED directly with layerwise step decay
    eta_l = eta d^(L-1-l), d in {0.8, 0.65, 0.5}, from the same W_0 (same a_l etc. at the same
    checkpoints); "flip" there = own-coordinate Euclidean sign(sum_l a_l) differs from the run's
    own invariant sign(sum_l eta_l a_l).
Analyses (all closed-form from the persisted a_l and norms, exactly as conjugation_check.reachability):
  * induced gauges: input-layer step-size factors k in {1/16, 1/4, 1/2, 2, 4, 16} (c_0 = sqrt(k)),
    orthogonal-vs-He (rounded (1/2,1,1,1) and measured profile over the 500 seeds), LeCun-vs-He
    (global, (1,1,1,1)), layerwise decay/growth read as c_l = d^(+-(L-1-l)/2) (the re-weighting proxy,
    compared with the directly trained decay runs);
  * random per-layer factors logU within sqrt2, 2, 4 (2000 draws, each applied to every cell);
    |cos| on either side of a flip (max of |cos(c)| and |cos(1)|) over flipped (draw, cell) pairs, and
    the grid supremum over the box [1/sqrt2, sqrt2]^4 and [1/2, 2]^4 (25 points per axis);
  * nearest sign-flipping gauge per mixed cell: best two-layer (c, 1/c) path (closed form), best single
    layer (closed form), all layers WITHOUT the product constraint (closed form t* = (1/4) log(sum_maj
    |a_l| / sum_min |a_l|)) and WITH prod c_l = 1 (bisection on t over the 61^3 grid of the slice
    sum_l log c_l = 0 of the box, bracketed by t* and the two-layer value; an upper bound within ~1%);
  * nearest single-neuron flip (roots of r u^2 + (S - b - r) u + b = 0, u = c^2, over the hidden
    neurons of layers 0..2), vectorized and checked against an np.roots version;
  * flip curves (fraction of all cells flipped by some rescaling of largest factor <= c) with a
    seed-cluster bootstrap 95% band, for the figure's right panel.
CIs: cluster bootstrap over seeds (2000 resamples, rng 12345), since cells inside a seed are paired.

Usage:  python reachability_scale.py run  [n_seeds]     -> reachability_scale_cells.npz  (~70 min, 3 workers)
        python reachability_scale.py analyse             -> reachability_scale.json, figures, report numbers
"""
import os, sys, json, time, pickle   # pickle: only for this script's own chunk checkpoints
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from conjugation_check import make_tasks, init_net, gd_run, grads, layer_inner_products, trunk_cosine, FIGDIR  # noqa: E402

KEY = "reachability_scale"
ETA, L, WIDTH, DIN, N, STEPS = 0.05, 4, 64, 16, 256, 400
RHOS = (-0.5, -0.2, 0.0, 0.3, 0.7)
CKPTS = [0, 99, 199, 299, 399]
DECAYS = (0.8, 0.65, 0.5)
SEED0, N_SEEDS_DEFAULT = 10, 500
N_JOBS = 3
SCRATCH = os.path.join(os.environ.get("SCRATCH_DIR", HERE), "reachability_scale_chunks")
NPZ = os.path.join(HERE, f"{KEY}_cells.npz")
JSON_OUT = os.path.join(HERE, f"{KEY}.json")


def llrd(d):
    """Layerwise step-size decay toward the input: eta_l = eta d^(L-1-l) ."""
    return [ETA * d ** (L - 1 - l) for l in range(L)]


# ----------------------------------------------------------------------------- run stage
def cell_features(X, Ys, W, H):
    losses, gW, gH = grads(X, Ys, W, H)
    P = [gW[0][l] * gW[1][l] for l in range(L)]
    a = np.array([p.sum() for p in P])
    return {"a": a, "loss": np.array(losses),
            "g1n2": np.array([np.sum(g ** 2) for g in gW[0]]), "g2n2": np.array([np.sum(g ** 2) for g in gW[1]]),
            "fro": np.array([np.linalg.norm(w) for w in W]), "cos": trunk_cosine(gW),
            "head_cross": float(sum(np.sum(gH[0][i] * gH[1][i]) for i in range(2))),
            "b_in": np.stack([P[l].sum(0) for l in range(L)]),            # (L, width): incoming column sums
            "r_out": np.stack([P[l + 1].sum(1) for l in range(L - 1)])}   # (L-1, width): outgoing row sums of hidden neurons


def trajectory_cells(X, Ys, W, H, etas):
    """GD in segments between checkpoints (bit-identical to one gd_run with record_every=1, without
    storing 400 weight copies); features at every checkpoint."""
    out, Wc, Hc, t = [], [w.copy() for w in W], [h.copy() for h in H], 0
    for k in CKPTS:
        if k > t:
            hist = gd_run(X, Ys, Wc, Hc, etas, ETA, k - t)
            Wc, Hc, t = hist["final_W"], hist["final_H"], k
        f = cell_features(X, Ys, Wc, Hc); f["step"] = k
        out.append(f)
    return out


def run_seed(s):
    rows = []
    for rho in RHOS:
        rng = np.random.default_rng(1000 + s)                     # same stream convention as sign_census
        X, Ys = make_tasks(rng, N, DIN, rho)
        W, H = init_net(rng, DIN, WIDTH, L)
        base = trajectory_cells(X, Ys, W, H, [ETA] * L)
        dec = {d: trajectory_cells(X, Ys, W, H, llrd(d)) for d in DECAYS}
        for i, k in enumerate(CKPTS):
            r = {"seed": s, "rho": rho, "step": k, "base": base[i], "decay": {d: dec[d][i] for d in DECAYS}}
            rows.append(r)
    return rows


def init_profile(seeds):
    """Mean per-layer Frobenius norm of orthogonal (gain 1) / LeCun init relative to He, global factor removed."""
    out = {}
    for scheme in ("he", "lecun", "orthogonal"):
        fro = np.array([[np.linalg.norm(w) for w in init_net(np.random.default_rng(1000 + s), DIN, WIDTH, L, scheme=scheme)[0]] for s in seeds])
        out[scheme] = fro.mean(0)
    prof = {}
    for scheme in ("lecun", "orthogonal"):
        r = out[scheme] / out["he"]
        prof[scheme + "_over_he"] = r.tolist(); prof[scheme + "_over_he_prod1"] = (r / np.prod(r) ** (1 / L)).tolist()
    return prof


def run_stage(n_seeds):
    from joblib import Parallel, delayed
    os.makedirs(SCRATCH, exist_ok=True)
    seeds = list(range(SEED0, SEED0 + n_seeds))
    chunk = 15
    t0 = time.time()
    all_rows = []
    for ci in range(0, len(seeds), chunk):
        sub = seeds[ci:ci + chunk]
        path = os.path.join(SCRATCH, f"chunk_{sub[0]}_{sub[-1]}.pkl")
        if os.path.exists(path):
            rows = pickle.load(open(path, "rb"))
        else:
            res = Parallel(n_jobs=N_JOBS)(delayed(run_seed)(s) for s in sub)
            rows = [r for rr in res for r in rr]
            pickle.dump(rows, open(path, "wb"))
        all_rows.extend(rows)
        done = ci + len(sub)
        el = time.time() - t0
        print(f"seeds {sub[0]}-{sub[-1]} done: {done}/{len(seeds)} [{el/60:.1f} min, eta {el/done*(len(seeds)-done)/60:.1f} min]", flush=True)
    # assemble arrays
    n = len(all_rows)
    arr = {"seed": np.array([r["seed"] for r in all_rows]), "rho": np.array([r["rho"] for r in all_rows]),
           "step": np.array([r["step"] for r in all_rows])}
    for key in ("a", "loss", "g1n2", "g2n2", "fro", "b_in", "r_out"):
        arr[key] = np.stack([r["base"][key] for r in all_rows])
    arr["cos"] = np.array([r["base"]["cos"] for r in all_rows]); arr["head_cross"] = np.array([r["base"]["head_cross"] for r in all_rows])
    for d in DECAYS:
        tag = f"decay{d}"
        for key in ("a", "loss", "g1n2", "g2n2", "fro"):
            arr[f"{tag}_{key}"] = np.stack([r["decay"][d][key] for r in all_rows])
        arr[f"{tag}_cos"] = np.array([r["decay"][d]["cos"] for r in all_rows])
        arr[f"{tag}_etas"] = np.array(llrd(d))
    arr["ckpts"] = np.array(CKPTS); arr["rhos"] = np.array(RHOS); arr["decays"] = np.array(DECAYS); arr["eta"] = np.array(ETA)
    prof = init_profile(seeds)
    arr["orthogonal_over_he_prod1"] = np.array(prof["orthogonal_over_he_prod1"]); arr["lecun_over_he_prod1"] = np.array(prof["lecun_over_he_prod1"])
    arr["orthogonal_over_he"] = np.array(prof["orthogonal_over_he"]); arr["lecun_over_he"] = np.array(prof["lecun_over_he"])
    np.savez_compressed(NPZ, **arr)
    print(f"wrote {NPZ}: {n} cells, {len(seeds)} seeds, {time.time()-t0:.0f} s total", flush=True)


# ----------------------------------------------------------------------------- analysis stage
BOOT_B, BOOT_SEED = 2000, 12345


class Boot:
    """Seed-cluster bootstrap: the same resampled seed sets for every statistic."""

    def __init__(self, seeds):
        self.seeds = seeds
        self.uniq, self.inv = np.unique(seeds, return_inverse=True)
        rng = np.random.default_rng(BOOT_SEED)
        self.samples = rng.integers(0, len(self.uniq), size=(BOOT_B, len(self.uniq)))
        # per-seed index lists for median statistics
        order = np.argsort(self.inv, kind="stable"); self.idx_sorted = order
        self.bounds = np.searchsorted(self.inv[order], np.arange(len(self.uniq) + 1))

    def mean(self, values, mask=None):
        """Mean of `values` over cells (optionally restricted to mask); returns (point, lo, hi)."""
        v = np.asarray(values, float)
        w = np.ones_like(v) if mask is None else np.asarray(mask, float)
        num = np.bincount(self.inv, weights=v * w, minlength=len(self.uniq))
        den = np.bincount(self.inv, weights=w, minlength=len(self.uniq))
        point = num.sum() / den.sum()
        bs = num[self.samples].sum(1) / np.maximum(den[self.samples].sum(1), 1e-300)
        return float(point), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    def quantile(self, values, q=50, mask=None):
        v = np.asarray(values, float)
        m = np.ones(len(v), bool) if mask is None else np.asarray(mask, bool)
        point = float(np.percentile(v[m], q))
        out = np.empty(BOOT_B)
        for b in range(BOOT_B):
            idx = np.concatenate([self.idx_sorted[self.bounds[s]:self.bounds[s + 1]] for s in self.samples[b]])
            vv = v[idx][m[idx]]
            out[b] = np.percentile(vv, q) if vv.size else np.nan
        return point, float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def fmt(t, pct=True, nd=1):
    if pct:
        return f"{100*t[0]:.{nd}f}% [{100*t[1]:.{nd}f}, {100*t[2]:.{nd}f}]"
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def nearest_pair_single(A):
    """Closed-form nearest flips on two-layer (c,1/c) paths and on a single layer, per cell (inf if none)."""
    n = len(A); pair = np.full(n, np.inf); single = np.full(n, np.inf)
    for idx, a in enumerate(A):
        s = np.sign(a);
        if s.max() == s.min():
            continue
        best = np.inf
        for i in range(L):
            for j in range(L):
                if i == j:
                    continue
                R = a.sum() - a[i] - a[j]
                for u in np.roots([a[j], R, a[i]]):
                    if abs(u.imag) < 1e-12 and u.real > 0:
                        best = min(best, abs(np.log(u.real)) / 2)
        pair[idx] = best
        bs, S = np.inf, a.sum()
        for l in range(L):
            q = 1 - S / a[l]
            if q > 0:
                bs = min(bs, abs(np.log(1 / q)) / 2)
        single[idx] = bs
    return np.exp(pair), np.exp(single)


def all_layer_unconstrained(A):
    S = A.sum(1); s = np.sign(S)
    maj = np.where(np.sign(A) == s[:, None], np.abs(A), 0).sum(1)
    mino = np.where(np.sign(A) == -s[:, None], np.abs(A), 0).sum(1)
    with np.errstate(divide="ignore"):
        return np.where(mino > 0, 0.25 * np.log(maj / np.maximum(mino, 1e-300)), np.inf)


def slice_grid(n_g=61):
    u = np.linspace(-1.0, 1.0, n_g)
    grids = np.stack(np.meshgrid(*([u] * (L - 1)), indexing="ij"), -1).reshape(-1, L - 1)
    last = -grids.sum(1); feas = np.abs(last) <= 1 + 1e-12
    return np.concatenate([grids[feas], last[feas, None]], 1)


def all_layer_constrained_chunk(A, lo_arr, hi_arr, n_iter=30):
    """Bisection on t (largest |log c_l|) over the product-one slice of the box, as conjugation_check."""
    Xg = slice_grid()
    out = np.empty(len(A))
    for i, (a, lo, hi) in enumerate(zip(A, lo_arr, hi_arr)):
        s = np.sign(a.sum()); sa = s * a
        def flips(t):
            return bool(np.min(np.exp(-2.0 * t * Xg) @ sa) <= 0)
        if not np.isfinite(lo):
            out[i] = np.inf; continue
        if flips(lo):
            out[i] = lo; continue
        hi = hi if np.isfinite(hi) else np.log(1e3)
        if not flips(hi):          # should not happen (two-layer point is feasible); widen once
            hi = np.log(1e6)
            if not flips(hi):
                out[i] = np.inf; continue
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            if flips(mid):
                hi = mid
            else:
                lo = mid
        out[i] = hi
    return out


def neuron_nearest_vec(b_in, r_out, S):
    """Vectorized nearest single-neuron flip: for hidden neuron (l, j), l = 0..L-2, u = c^2 solves
    r u^2 + (S - b - r) u + b = 0 with b = incoming column sum, r = outgoing row sum. Returns factor per cell."""
    b = b_in[:, :L - 1, :]; r = r_out                                   # (n, L-1, width)
    S3 = S[:, None, None]
    B = S3 - b - r
    best = np.full(b.shape, np.inf)
    quad = np.abs(r) > 1e-300
    disc = B ** 2 - 4 * r * b
    with np.errstate(divide="ignore", invalid="ignore"):
        sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
        for sgn in (+1, -1):
            u = (-B + sgn * sq) / (2 * r)
            ok = quad & np.isfinite(u) & (u > 0)
            best = np.where(ok, np.minimum(best, np.abs(np.log(np.where(ok, u, 1.0))) / 2), best)
        u = -b / B                                                       # linear case r == 0
        ok = (~quad) & np.isfinite(u) & (u > 0)
        best = np.where(ok, np.minimum(best, np.abs(np.log(np.where(ok, u, 1.0))) / 2), best)
    return np.exp(best.reshape(len(S), -1).min(1))


def neuron_nearest_roots(b_in, r_out, S):
    """np.roots version, for validation on a few cells."""
    best = np.inf
    for l in range(L - 1):
        b = b_in[l]; r = r_out[l]
        for j in range(WIDTH):
            rest = S - b[j] - r[j]
            for u in np.roots([r[j], rest, b[j]]):
                if abs(u.imag) < 1e-12 and u.real > 0:
                    best = min(best, abs(np.log(u.real)) / 2)
    return float(np.exp(best))


def analyse_stage():
    from joblib import Parallel, delayed
    t0 = time.time()
    D = dict(np.load(NPZ))
    seeds, rho, step = D["seed"], D["rho"], D["step"]
    A, G1, G2, cos1 = D["a"], D["g1n2"], D["g2n2"], D["cos"]
    n_cells, n_seeds = len(A), len(np.unique(seeds))
    S = A.sum(1); sign0 = np.sign(S)
    mixed = np.sign(A).max(1) != np.sign(A).min(1)
    post = step > 0
    bt = Boot(seeds)
    out = {"key": KEY, "n_seeds": int(n_seeds), "seeds": f"{int(seeds.min())}-{int(seeds.max())} (rng streams 1000+s)", "n_cells": int(n_cells),
           "design": "He init, eta=0.05, 400 steps, rho in " + str(list(RHOS)) + ", checkpoints " + str(CKPTS) + ", paired within seed",
           "ci": f"seed-cluster bootstrap, {BOOT_B} resamples, 95%", "diverged_cells": int((~np.isfinite(A).all(1)).sum()),
           "max_abs_head_cross": float(np.abs(D["head_cross"]).max())}
    assert out["diverged_cells"] == 0

    # ---- census basics
    out["census"] = {
        "mixed_frac": bt.mean(mixed), "mixed_frac_post_init": bt.mean(mixed, post),
        "mixed_by_step": {int(k): bt.mean(mixed, step == k) for k in CKPTS},
        "mixed_by_rho": {str(r): bt.mean(mixed, rho == r) for r in RHOS},
        "conflict_rate": bt.mean(sign0 < 0), "conflict_rate_post_init": bt.mean(sign0 < 0, post),
        "conflict_by_step": {int(k): bt.mean(sign0 < 0, step == k) for k in CKPTS},
        "conflict_by_rho": {str(r): bt.mean(sign0 < 0, rho == r) for r in RHOS},
        "conflict_by_rho_post_init": {str(r): bt.mean(sign0 < 0, (rho == r) & post) for r in RHOS},
        "abs_cos_median_mixed": bt.quantile(np.abs(cos1), 50, mixed), "abs_cos_median_single_sign": bt.quantile(np.abs(cos1), 50, ~mixed),
        "layer_fro_ratio_median": float(np.median(D["fro"].max(1) / D["fro"].min(1))), "layer_fro_ratio_max": float((D["fro"].max(1) / D["fro"].min(1)).max()),
        "initial_loss_median": [float(np.median(D["loss"][step == 0][:, i])) for i in range(2)],
        "final_loss_median": [float(np.median(D["loss"][step == 399][:, i])) for i in range(2)],
    }
    print(f"census: mixed {fmt(out['census']['mixed_frac'])} (post-init {fmt(out['census']['mixed_frac_post_init'])}); conflict {fmt(out['census']['conflict_rate'])} "
          f"(post-init {fmt(out['census']['conflict_rate_post_init'])}); by step {[round(100*v[0],1) for v in out['census']['conflict_by_step'].values()]}; "
          f"by rho {[round(100*v[0],1) for v in out['census']['conflict_by_rho'].values()]}", flush=True)

    def sign_at(c):
        return np.sign(A @ (np.asarray(c, float) ** -2.0))

    def cos_at(c):
        w = np.asarray(c, float) ** -2.0
        return (A @ w) / np.sqrt((G1 @ w) * (G2 @ w))

    # ---- named (induced) gauges
    named = {}
    prof = D["orthogonal_over_he_prod1"]
    gauges = {}
    for k in (1 / 16, 1 / 4, 1 / 2, 2, 4, 16):
        gauges[f"input_layer_step_x{k:g}_(c0=sqrt(k))"] = [np.sqrt(k)] + [1] * (L - 1)
    gauges["input_scale_(1/4,1,1,1)_[step_x1/16]"] = [0.25, 1, 1, 1]; gauges["input_scale_(4,1,1,1)_[step_x16]"] = [4, 1, 1, 1]
    gauges["orthogonal_vs_he_rounded_(1/2,1,1,1)"] = [0.5, 1, 1, 1]
    gauges["orthogonal_vs_he_measured_profile"] = list(map(float, prof))
    gauges["lecun_vs_he_global_(1,1,1,1)"] = [1] * L
    for d in (0.9, 0.8, 0.65, 0.5):
        gauges[f"step_decay_toward_input_d={d}_proxy"] = [d ** ((L - 1 - l) / 2) for l in range(L)]
        gauges[f"step_growth_toward_input_d={d}_proxy"] = [d ** (-(L - 1 - l) / 2) for l in range(L)]
    for k, c in gauges.items():
        sc = sign_at(c); fl = sc != sign0
        named[k] = {"c": [float(v) for v in c], "flip_frac": bt.mean(fl), "n_flips": int(fl.sum()), "flips_per_125": float(fl.mean() * 125),
                    "conflict_rate": bt.mean(sc < 0), "flip_frac_post_init": bt.mean(fl, post),
                    "flips_switch_pcgrad_on": int((fl & (sc < 0)).sum()), "flips_switch_pcgrad_off": int((fl & (sc > 0)).sum())}
    out["named_gauges"] = named
    inp_keys = [k for k in named if k.startswith("input_layer_step")]
    out["input_layer_step_max_flip_frac_over_6_factors"] = max(named[k]["flip_frac"][0] for k in inp_keys)
    out["init_profiles"] = {"orthogonal_over_he": D["orthogonal_over_he"].tolist(), "orthogonal_over_he_prod1": prof.tolist(),
                            "lecun_over_he": D["lecun_over_he"].tolist(), "lecun_over_he_prod1": D["lecun_over_he_prod1"].tolist()}
    for k in named:
        print(f"  {k:48s} flips {fmt(named[k]['flip_frac'],nd=2)} ({named[k]['flips_per_125']:.1f}/125) conflict {fmt(named[k]['conflict_rate'])}", flush=True)

    # ---- random per-layer gauges (each draw applied to every cell)
    rng = np.random.default_rng(0)
    rand = {}
    n_draws = 2000
    for lo, hi, key in ((1 / np.sqrt(2), np.sqrt(2), "logU[1/sqrt2,sqrt2]"), (0.5, 2.0, "logU[1/2,2]"), (0.25, 4.0, "logU[1/4,4]")):
        C = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(n_draws, L)))
        Wt = (C ** -2.0).T                                               # (L, draws)
        Nd = A @ Wt                                                      # (cells, draws)
        sd = np.sign(Nd); fl = sd != sign0[:, None]
        p_cell = fl.mean(1)                                              # per-cell flip probability over draws
        cosd = Nd / np.sqrt((G1 @ Wt) * (G2 @ Wt))
        either = np.maximum(np.abs(cosd), np.abs(cos1)[:, None])
        fac = np.abs(cosd)[fl]; fae = either[fl]
        crates = (sd < 0).mean(0)
        # bootstrap the medians of the flipped-pair |cos| by seed cluster: per-cell median over its flipped draws is a
        # decent proxy; the pooled median is reported as the point estimate
        rand[key] = {"n_draws": n_draws, "mean_flip_frac": bt.mean(p_cell), "mean_flip_frac_post_init": bt.mean(p_cell, post),
                     "flips_per_125_mean": float(p_cell.mean() * 125),
                     "max_flip_frac_over_draws": float(fl.mean(0).max()), "min_flip_frac_over_draws": float(fl.mean(0).min()),
                     "frac_cells_ever_flipped": bt.mean(p_cell > 0),
                     "conflict_rate_min": float(crates.min()), "conflict_rate_max": float(crates.max()),
                     "flipped_pairs_abs_cos_median": float(np.median(fac)), "flipped_pairs_abs_cos_max": float(fac.max()), "flipped_pairs_abs_cos_q90": float(np.percentile(fac, 90)),
                     "flipped_pairs_abs_cos_either_side_median": float(np.median(fae)), "flipped_pairs_abs_cos_either_side_q90": float(np.percentile(fae, 90)),
                     "flipped_pairs_abs_cos_either_side_max": float(fae.max()), "n_flipped_pairs": int(fac.size),
                     "flipped_pairs_switch_pcgrad_on_frac": float(((sd < 0) & fl).sum() / fl.sum())}
        print(f"  random {key}: flip frac {fmt(rand[key]['mean_flip_frac'],nd=2)} ({rand[key]['flips_per_125_mean']:.2f}/125), max over draws {100*rand[key]['max_flip_frac_over_draws']:.1f}%, "
              f"|cos| either side median {rand[key]['flipped_pairs_abs_cos_either_side_median']:.4f} max {rand[key]['flipped_pairs_abs_cos_either_side_max']:.4f}", flush=True)
    out["random_gauges"] = rand

    # ---- box supremum of |cos| on either side over mixed cells
    box = {}
    n_b = 25
    Am, G1m, G2m, c1m, s0m = A[mixed], G1[mixed], G2[mixed], cos1[mixed], sign0[mixed]
    for hi_f, key in ((np.sqrt(2), "box[1/sqrt2,sqrt2]"), (2.0, "box[1/2,2]")):
        ub = np.linspace(-np.log(hi_f), np.log(hi_f), n_b)
        Xb = np.stack(np.meshgrid(*([ub] * L), indexing="ij"), -1).reshape(-1, L)
        Wb = np.exp(-2.0 * Xb).T
        cell_max = np.full(len(Am), np.nan); flips_any = np.zeros(len(Am), bool)
        for i0 in range(0, len(Am), 40):
            sl = slice(i0, i0 + 40)
            Nb = Am[sl] @ Wb; cb = Nb / np.sqrt((G1m[sl] @ Wb) * (G2m[sl] @ Wb))
            fl = np.sign(Nb) != s0m[sl, None]
            either = np.maximum(np.abs(cb), np.abs(c1m[sl])[:, None])
            flips_any[sl] = fl.any(1)
            cell_max[sl] = np.where(fl.any(1), np.where(fl, either, 0.0).max(1), np.nan)
        full = np.full(n_cells, np.nan); full[np.where(mixed)[0]] = cell_max
        fa = np.zeros(n_cells, bool); fa[np.where(mixed)[0]] = flips_any
        box[key] = {"grid_points_per_axis": n_b, "frac_cells_flipping_in_box": bt.mean(fa), "n_cells_flipping_in_box": int(fa.sum()),
                    "abs_cos_either_side_sup": float(np.nanmax(cell_max)), "abs_cos_either_side_per_cell_max_median": float(np.nanmedian(cell_max)),
                    "abs_cos_either_side_per_cell_max_q90": float(np.nanpercentile(cell_max, 90)),
                    "frac_flipping_cells_with_max_above_0.05": float(np.nanmean(cell_max > 0.05)),
                    "frac_flipping_cells_with_max_above_0.071": float(np.nanmean(cell_max > 0.0706)),
                    "frac_flipping_cells_with_max_above_0.1": float(np.nanmean(cell_max > 0.1))}
        print(f"  {key}: cells flipping {fmt(box[key]['frac_cells_flipping_in_box'])}, |cos| either side sup {box[key]['abs_cos_either_side_sup']:.4f}, per-cell max median {box[key]['abs_cos_either_side_per_cell_max_median']:.4f}", flush=True)
    out["abs_cos_either_side_box_supremum"] = box

    # ---- nearest flipping gauges
    pair_c, single_c = nearest_pair_single(A)
    uncon_logc = all_layer_unconstrained(A)
    mi = np.where(mixed)[0]
    chunks = np.array_split(mi, 3 * N_JOBS)
    res = Parallel(n_jobs=N_JOBS)(delayed(all_layer_constrained_chunk)(A[ch], uncon_logc[ch], np.log(pair_c[ch])) for ch in chunks)
    con_logc = np.full(n_cells, np.inf); con_logc[np.concatenate(chunks)] = np.concatenate(res)
    all_c, uncon_c = np.exp(con_logc), np.exp(uncon_logc)
    neuron_c = neuron_nearest_vec(D["b_in"], D["r_out"], S)
    # validation of the vectorized neuron flip against the np.roots form on 40 cells
    chk = np.random.default_rng(1).choice(n_cells, 40, replace=False)
    ref = np.array([neuron_nearest_roots(D["b_in"][i], D["r_out"][i], S[i]) for i in chk])
    neuron_check_maxrel = float(np.max(np.abs(ref - neuron_c[chk]) / ref))
    assert neuron_check_maxrel < 1e-8, neuron_check_maxrel

    def summ(cv, name):
        fin = np.isfinite(cv)
        d = {"n_finite": int(fin.sum()), "n_mixed": int(mixed.sum()), "n_cells": n_cells,
             "median_of_finite": bt.quantile(cv, 50, fin), "q25_of_finite": bt.quantile(cv, 25, fin), "q75_of_finite": bt.quantile(cv, 75, fin),
             "median_of_finite_post_init": bt.quantile(cv, 50, fin & post)}
        for thr, lab in ((np.sqrt(2), "sqrt2"), (2, "2"), (3, "3"), (10, "10")):
            w = cv <= thr + 1e-12
            d[f"frac_all_cells_within_{lab}"] = bt.mean(w); d[f"per_125_within_{lab}"] = float(w.mean() * 125)
            d[f"frac_post_init_cells_within_{lab}"] = bt.mean(w, post)
            d[f"frac_mixed_cells_within_{lab}"] = bt.mean(w, mixed) if name != "neuron" else bt.mean(w)
        print(f"  nearest flip [{name}]: median {fmt(d['median_of_finite'],pct=False)}; within sqrt2 {fmt(d['frac_all_cells_within_sqrt2'])} ({d['per_125_within_sqrt2']:.1f}/125); "
              f"within 2 {fmt(d['frac_all_cells_within_2'])} ({d['per_125_within_2']:.1f}/125); within 3 {d['per_125_within_3']:.1f}/125; within 10 {d['per_125_within_10']:.1f}/125", flush=True)
        return d
    out["nearest_flip"] = {"all_layers_prod1": summ(all_c, "all layers, prod 1"), "all_layers_unconstrained_closed_form": summ(uncon_c, "all layers, unconstrained"),
                           "best_pair": summ(pair_c, "two-layer pair"), "single_layer": summ(single_c, "single layer"), "single_neuron": summ(neuron_c, "neuron"),
                           "constrained_equals_unconstrained_frac_of_mixed": float(np.mean(np.abs(con_logc[mixed] - uncon_logc[mixed]) < 1e-6)),
                           "neuron_vectorized_vs_roots_max_rel_err_40_cells": neuron_check_maxrel,
                           "slice_grid_points_per_axis": 61, "bisection_iterations": 30}
    # |cos| at c = 1 on cells whose nearest per-layer flip is within sqrt2 / 2 (what PCGrad sees on the active side at c=1;
    # at the flip itself cos = 0). Also the direction: sign at c = 1 positive => the flip switches PCGrad ON.
    for thr, lab in ((np.sqrt(2), "sqrt2"), (2, "2")):
        w = all_c <= thr + 1e-12
        out["nearest_flip"][f"abs_cos_at_c1_cells_flipping_within_{lab}"] = {"median": float(np.median(np.abs(cos1[w]))), "q90": float(np.percentile(np.abs(cos1[w]), 90)), "max": float(np.abs(cos1[w]).max()),
                                                                             "frac_flip_switches_pcgrad_on": float((sign0[w] > 0).mean()), "n": int(w.sum())}
    # ---- directly trained layerwise step decay
    dec = {}
    for d in DECAYS:
        Ad = D[f"decay{d}_a"]; etas = D[f"decay{d}_etas"]
        se = np.sign(Ad.sum(1)); si = np.sign(Ad @ (etas / ETA)); fl = se != si
        mixed_d = np.sign(Ad).max(1) != np.sign(Ad).min(1)
        proxy_fl = named[f"step_decay_toward_input_d={d}_proxy"]["flip_frac"]
        # the proxy re-weighting on the base census vs. the direct run, cell by cell
        c_proxy = np.array([d ** ((L - 1 - l) / 2) for l in range(L)])
        proxy_cells = sign_at(c_proxy) != sign0
        dec[str(d)] = {"etas": etas.tolist(), "flip_frac_own_vs_invariant": bt.mean(fl), "flips_per_125": float(fl.mean() * 125), "n_flips": int(fl.sum()),
                       "flip_frac_post_init": bt.mean(fl, post), "flip_frac_step0": bt.mean(fl, step == 0),
                       "flips_by_step_per_125": {int(k): float(fl[step == k].mean() * 25) for k in CKPTS},
                       "conflict_rate_own": bt.mean(se < 0), "conflict_rate_invariant": bt.mean(si < 0), "mixed_frac": bt.mean(mixed_d),
                       "conflict_own_by_step": {int(k): float((se < 0)[step == k].mean()) for k in CKPTS},
                       "reweighting_proxy_on_base_census_flip_frac": proxy_fl, "proxy_and_direct_agree_on_flip_frac": float((proxy_cells == fl).mean()),
                       "abs_cos_own_median_on_flipped": float(np.median(np.abs(D[f"decay{d}_cos"][fl]))) if fl.any() else None,
                       "final_loss_median": [float(np.median(D[f"decay{d}_loss"][step == 399][:, i])) for i in range(2)],
                       "frac_seeds_with_any_flip": float(len(np.unique(seeds[fl])) / n_seeds),
                       "flips_per_seed_max": int(np.bincount(np.unique(seeds[fl], return_inverse=True)[1]).max()) if fl.any() else 0}
        print(f"  decay d={d}: flips own-vs-invariant {fmt(dec[str(d)]['flip_frac_own_vs_invariant'],nd=2)} ({dec[str(d)]['flips_per_125']:.1f}/125; by step {dec[str(d)]['flips_by_step_per_125']}), "
              f"conflict own {fmt(dec[str(d)]['conflict_rate_own'])} invariant {fmt(dec[str(d)]['conflict_rate_invariant'])}; proxy {100*proxy_fl[0]:.2f}%", flush=True)
    out["step_decay_direct"] = dec

    # ---- flip curves with CI band
    xs = np.logspace(0, np.log10(30), 300)
    curves = {}
    for name, cv in (("per_layer", all_c), ("neuron", neuron_c), ("two_layer", pair_c), ("single_layer", single_c)):
        W_ = (cv[:, None] <= xs[None, :] + 1e-12).astype(float)                        # (cells, xs)
        per_seed = np.zeros((len(bt.uniq), len(xs))); np.add.at(per_seed, bt.inv, W_)
        cnt = np.bincount(bt.inv, minlength=len(bt.uniq)).astype(float)
        point = per_seed.sum(0) / cnt.sum()
        bs = per_seed[bt.samples].sum(1) / cnt[bt.samples].sum(1)[:, None]
        curves[name] = {"point": point, "lo": np.percentile(bs, 2.5, axis=0), "hi": np.percentile(bs, 97.5, axis=0)}
    out["flip_curve_xs_note"] = "arrays in the npz: flipcurve_xs, flipcurve_<name>_{point,lo,hi}"
    plot_curves(xs, curves, n_seeds)

    # ---- persist per-cell derived arrays alongside the raw ones
    D.update({"mixed": mixed, "sign0": sign0, "nearest_all_layers_prod1": all_c, "nearest_all_layers_unconstrained": uncon_c,
              "nearest_pair": pair_c, "nearest_single_layer": single_c, "nearest_neuron": neuron_c, "flipcurve_xs": xs})
    for name, cv in curves.items():
        for k, v in cv.items():
            D[f"flipcurve_{name}_{k}"] = v
    np.savez_compressed(NPZ, **D)
    out["runtime_analysis_s"] = time.time() - t0
    json.dump(out, open(JSON_OUT, "w"), indent=1, allow_nan=False)
    print(f"wrote {JSON_OUT} and updated {NPZ} [{time.time()-t0:.0f}s]", flush=True)
    return out


def plot_curves(xs, curves, n_seeds=500):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIGDIR, exist_ok=True)
    for tag, size, fs in (("", (1.85, 0.92), 6), ("_large", (4.2, 2.6), 8)):
        fig, ax = plt.subplots(figsize=size, constrained_layout=True)
        for name, col, ls, lab, (tx, ty, va) in (("per_layer", "C0", "-", "any layers", (9.5, None, "top")), ("neuron", "C2", "--", "one neuron", (2.35, 0.86, "bottom"))):
            c = curves[name]
            ax.fill_between(xs, c["lo"], c["hi"], color=col, alpha=0.2, lw=0)
            ax.plot(xs, c["point"], color=col, lw=1.2, ls=ls)
            ax.text(tx, float(c["point"][-1]) - 0.06 if ty is None else ty, lab, color=col, fontsize=fs, ha="left", va=va)
        for xv, lab in ((np.sqrt(2), r"$\sqrt{2}$"), (2, "2")):
            ax.axvline(xv, color="0.5", lw=0.6, ls=":"); ax.text(xv, 0.97, lab, fontsize=fs, ha="center", va="top", color="0.3")
        ax.set_xscale("log"); ax.set_xlim(1, 30); ax.set_ylim(-0.03, 1.05)
        ax.set_xticks([1, 2, 5, 10, 30]); ax.set_xticklabels(["1", "2", "5", "10", "30"])
        ax.set_ylabel("cells flipped", fontsize=fs + 0.5, labelpad=1); ax.set_xlabel(r"nearest flipping factor $c$", fontsize=fs + 0.5, labelpad=1)
        ax.tick_params(labelsize=fs, pad=1.5)
        if tag:
            ax.set_title(f"{n_seeds} seeds x 5 rho x 5 checkpoints; band = seed-cluster bootstrap 95% CI", fontsize=fs)
        base = os.path.join(FIGDIR, f"{KEY}_flipcurve{tag}")
        fig.savefig(base + ".png", dpi=250); fig.savefig(base + ".pdf"); plt.close(fig)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        run_stage(int(sys.argv[2]) if len(sys.argv) > 2 else N_SEEDS_DEFAULT)
    elif mode == "analyse":
        analyse_stage()
    elif mode == "time":
        t0 = time.time(); run_seed(SEED0); print(f"one seed (5 rho x 4 trajectories = 20 runs): {time.time()-t0:.1f} s")
    else:
        raise SystemExit("usage: reachability_scale.py run [n_seeds] | analyse | time")
