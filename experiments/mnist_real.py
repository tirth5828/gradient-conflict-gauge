"""mnist_real.py -- real-data, minibatch two-task run (MNIST) for the gauge question.

Checks the synthetic results (width-64 regression, full-batch GD) on real data. Two tasks on the same MNIST image: (A) 10-way digit classification, softmax
cross-entropy; (B) parity (odd = +1), logistic loss. Bias-free ReLU MLP trunk 784-256-256-256
(depth 3; depth 4 = 784-256^4 as a second configuration), two linear heads, minibatch SGD on the
summed loss, batch 128, a FIXED batch order per seed (so the per-layer conjugation is exact),
float64 numpy throughout (the trunk gradients, a_l, cosines, PCGrad); torch float64 autograd only
for the top Hessian eigenpair (power iteration, 20 iterations, on a fixed 512-example batch).

Stages (python mnist_real.py [run|analyze|all], default all; `run` skips if the npz exists):
  1. Exact conjugation check: 300 steps at c = (3, 1/3, 1) with eta_l = c_l^2 eta vs uniform eta:
     per-step losses of both tasks, max relative difference; max |W^B - c W^A|.
  2. Census: seeds 100..119 x eta in {0.01, 0.05, 0.1} x 3 epochs (1404 steps). Every step: the
     per-layer a_l = <g_1^l, g_2^l>, per-layer gradient norms, both minibatch losses (hence the
     minibatch trunk cosine and the conflict rate). Every 100 steps (15 checkpoints): the same on a
     fixed 10k training subset ("full" gradient), per-neuron incoming inner products, layer
     Frobenius norms, test accuracy on both tasks, eta*lambda_max and the top eigenvector's trunk
     block (fraction of each task gradient along it, cosine with it removed).
  3. PCGrad (Euclidean, on the trunk; heads take their own gradients) vs summed SGD, same seeds,
     same batch order, same init: test accuracy on both tasks at the checkpoints and at the end.
  4. Post hoc from the persisted a_l and norms: mixed-cell fraction, conflict rate, induced-gauge
     flip fractions (input-layer factors, layerwise step decay/growth, measured orthogonal/He
     profile, random per-layer factors within sqrt2 / 2 / 4), nearest flipping per-layer gauge
     (two-layer closed form; all-layer constrained by bisection on a grid; unconstrained closed
     form), the self-limiting |cos| bound, all with cluster bootstrap CIs over seeds.

Seeds: init rng = default_rng(2000 + seed), batch order rng = default_rng(10000 + seed),
seed in 100..119 (depth 3) and 100..109 (depth 4); disjoint from the development seeds 0-9.
Writes mnist_real_cells.npz (all per-step and per-checkpoint arrays), mnist_real.json (summary
with CIs) and ../figures/mnist_real_*.png. Uses at most 3 worker processes.
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"

import gzip
import json
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from conjugation_check import pcgrad_trunk, trunk_cosine  # noqa: E402  (imported, not copied)

FIGDIR = os.path.join(HERE, "..", "figures")
MNIST_DIR = os.environ.get("MNIST_DIR", os.path.join(HERE, "..", "data", "mnist"))  # raw idx files
NPZ = os.path.join(HERE, "mnist_real_cells.npz")
RUNS_DIR = os.path.join(os.path.dirname(MNIST_DIR.rstrip("/")), "runs")
JSON_OUT = os.path.join(HERE, "mnist_real.json")

N_JOBS = 3
BATCH = 128
EPOCHS = 3
CKPT_EVERY = 100
ETAS = [0.01, 0.05, 0.1]
SEEDS_D3 = list(range(100, 120))
SEEDS_D3_PC = list(range(100, 110))  # PCGrad at depth 3 on 10 seeds
SEEDS_D4 = list(range(100, 110))
SEEDS_D4_PC = list(range(100, 105))   # PCGrad at depth 4 on 5 seeds
WIDTHS_D3 = (784, 256, 256, 256)
WIDTHS_D4 = (784, 256, 256, 256, 256)
SUB_N = 10000          # fixed training subset for the "full" gradient at checkpoints
HESS_N = 512           # fixed batch for the Hessian power iteration
HESS_ITERS = 20


# ----------------------------------------------------------------------------- data
def _idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = struct.unpack(">I", data[:4])[0]
    ndim = magic & 0xFF
    assert (magic >> 8) & 0xFF == 0x08, "expected ubyte IDX"
    dims = struct.unpack(">" + "I" * ndim, data[4:4 + 4 * ndim])
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


_DATA = None


def get_data():
    """(Xtr uint8 [60000,784], ytr int64, Xte uint8 [10000,784], yte). Pixels are scaled to [0,1]
    when a batch is formed. No centering (the trunk is bias-free; centering would only rescale)."""
    global _DATA
    if _DATA is None:
        Xtr = _idx(os.path.join(MNIST_DIR, "train-images-idx3-ubyte.gz")).reshape(-1, 784)
        ytr = _idx(os.path.join(MNIST_DIR, "train-labels-idx1-ubyte.gz")).astype(np.int64)
        Xte = _idx(os.path.join(MNIST_DIR, "t10k-images-idx3-ubyte.gz")).reshape(-1, 784)
        yte = _idx(os.path.join(MNIST_DIR, "t10k-labels-idx1-ubyte.gz")).astype(np.int64)
        _DATA = (Xtr, ytr, Xte, yte)
    return _DATA


# ----------------------------------------------------------------------------- model (float64)
def init_mlp(rng, widths, scheme="he"):
    W = []
    for fi, fo in zip(widths[:-1], widths[1:]):
        if scheme == "he":
            W.append(rng.standard_normal((fi, fo)) * np.sqrt(2.0 / fi))
        elif scheme == "orthogonal":
            A = rng.standard_normal((max(fi, fo), min(fi, fo)))
            Q, R = np.linalg.qr(A)
            Q = Q * np.sign(np.diag(R))
            W.append(Q if fi >= fo else Q.T)
        else:
            raise ValueError(scheme)
    w = widths[-1]
    H = [rng.standard_normal((w, 10)) * np.sqrt(1.0 / w), rng.standard_normal((w, 1)) * np.sqrt(1.0 / w)]
    return W, H


def forward(X, W):
    hs = [X]
    h = X
    for Wl in W:
        h = np.maximum(h @ Wl, 0.0)
        hs.append(h)
    return hs


def task_grads(X, y, W, H, want_grads=True):
    """Two-task losses and per-task gradients. Task A: softmax CE on digits; task B: logistic loss
    on parity (odd = +1). Returns (losses[2], accs[2], gW[task][layer], gH[task])."""
    hs = forward(X, W)
    h = hs[-1]
    n = X.shape[0]
    L = len(W)
    z = h @ H[0]
    z = z - z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1, keepdims=True))
    logp = z - lse
    loss_a = float(-logp[np.arange(n), y].mean())
    acc_a = float((z.argmax(axis=1) == y).mean())
    s = (h @ H[1])[:, 0]
    yb = np.where(y % 2 == 1, 1.0, -1.0)
    loss_b = float(np.mean(np.logaddexp(0.0, -yb * s)))
    acc_b = float(((s > 0) == (yb > 0)).mean())
    if not want_grads:
        return (loss_a, loss_b), (acc_a, acc_b), None, None
    p = np.exp(logp)
    p[np.arange(n), y] -= 1.0
    dz = p / n
    ds = (-yb * np.exp(-np.logaddexp(0.0, yb * s)) / n)[:, None]   # -y sigma(-y s) / n
    gH = [h.T @ dz, h.T @ ds]
    deltas = [dz @ H[0].T, ds @ H[1].T]
    gW = [[None] * L for _ in range(2)]
    for i in range(2):
        delta = deltas[i]
        for l in range(L - 1, -1, -1):
            delta = delta * (hs[l + 1] > 0)
            gW[i][l] = hs[l].T @ delta
            delta = delta @ W[l].T
    return (loss_a, loss_b), (acc_a, acc_b), gW, gH


def layer_stats(gW):
    a = np.array([np.sum(gW[0][l] * gW[1][l]) for l in range(len(gW[0]))])
    n1 = np.array([np.linalg.norm(gW[0][l]) for l in range(len(gW[0]))])
    n2 = np.array([np.linalg.norm(gW[1][l]) for l in range(len(gW[0]))])
    return a, n1, n2


def neuron_incoming(gW):
    """Per-neuron incoming inner products b_{l,j} = sum_i g1[l,i,j] g2[l,i,j] (width 256 per layer)."""
    return np.stack([(gW[0][l] * gW[1][l]).sum(axis=0) for l in range(len(gW[0]))])


def hessian_top(W, H, Xh, yh, iters=HESS_ITERS, seed=0):
    """Top eigenpair of the full-parameter Hessian of the summed loss on the fixed batch (Xh, yh),
    torch float64 double-backward power iteration. ReLU'' = 0 so this is the local Hessian, the same
    object as the finite-difference HVP inside one activation pattern used by the synthetic scripts.
    Returns (lambda_max, v_trunk (unit norm, trunk block only), v_trunk_energy)."""
    import torch
    torch.set_num_threads(1)
    L = len(W)
    params = [torch.tensor(w, dtype=torch.float64, requires_grad=True) for w in W] + \
             [torch.tensor(h, dtype=torch.float64, requires_grad=True) for h in H]
    Xt = torch.tensor(Xh, dtype=torch.float64)
    yt = torch.tensor(yh)
    h = Xt
    for w in params[:L]:
        h = torch.relu(h @ w)
    z = h @ params[L]
    s = (h @ params[L + 1])[:, 0]
    ybt = torch.where(yt % 2 == 1, 1.0, -1.0).to(torch.float64)
    loss = torch.nn.functional.cross_entropy(z, yt) + torch.nn.functional.softplus(-ybt * s).mean()
    g = torch.autograd.grad(loss, params, create_graph=True)
    gflat = torch.cat([x.reshape(-1) for x in g])
    rng = np.random.default_rng(seed)
    v = torch.tensor(rng.standard_normal(gflat.numel()), dtype=torch.float64)
    v = v / v.norm()
    lam = 0.0
    for _ in range(iters):
        hv = torch.autograd.grad(gflat @ v, params, retain_graph=True)
        hv = torch.cat([x.reshape(-1) for x in hv]).detach()
        lam = float(v @ hv)
        nrm = float(hv.norm())
        if nrm == 0.0:
            break
        v = hv / nrm
    n_trunk = sum(w.size for w in W)
    vt = v[:n_trunk].numpy().copy()
    e = float(vt @ vt)
    vt = vt / np.sqrt(e)
    return lam, vt, e


def flat(gWi):
    return np.concatenate([g.ravel() for g in gWi])


def eig_projection_stats(gW, vt):
    g1, g2 = flat(gW[0]), flat(gW[1])
    f1 = float((g1 @ vt) ** 2 / (g1 @ g1))
    f2 = float((g2 @ vt) ** 2 / (g2 @ g2))
    same = float(np.sign(g1 @ vt) == np.sign(g2 @ vt))
    gs = g1 + g2
    fs = float((gs @ vt) ** 2 / (gs @ gs))
    r1 = g1 - (g1 @ vt) * vt
    r2 = g2 - (g2 @ vt) * vt
    cos_rm = float(r1 @ r2 / (np.linalg.norm(r1) * np.linalg.norm(r2)))
    return f1, f2, same, fs, cos_rm


# ----------------------------------------------------------------------------- one training run
def train(seed, eta, widths, epochs=EPOCHS, mode="sum", c=None, ckpt_every=CKPT_EVERY, batch=BATCH,
          hess=True, steps=None, quiet=True):
    """One minibatch SGD run on the summed loss (mode 'sum') or Euclidean trunk PCGrad (mode 'pc').
    c: optional per-layer gauge (W_l -> c_l W_l, eta_l -> c_l^2 eta); heads keep eta.
    Returns a dict of per-step and per-checkpoint arrays."""
    Xtr, ytr, Xte, yte = get_data()
    L = len(widths) - 1
    W, H = init_mlp(np.random.default_rng(2000 + seed), widths)
    eta_layers = [eta] * L
    if c is not None:
        W = [cl * w for cl, w in zip(c, W)]
        eta_layers = [cl ** 2 * eta for cl in c]
    order_rng = np.random.default_rng(10000 + seed)
    spe = Xtr.shape[0] // batch
    T = epochs * spe if steps is None else steps
    sub_idx = np.random.default_rng(777).choice(Xtr.shape[0], SUB_N, replace=False)   # same for all seeds
    Xsub = Xtr[sub_idx] / 255.0
    ysub = ytr[sub_idx]
    Xh, yh = Xsub[:HESS_N], ysub[:HESS_N]
    Xte_f = Xte / 255.0

    A = np.zeros((T, L)); N1 = np.zeros((T, L)); N2 = np.zeros((T, L)); LOSS = np.zeros((T, 2))
    ACTIVE = np.zeros(T, dtype=bool)
    ck = {k: [] for k in ["step", "a_mb", "n1_mb", "n2_mb", "loss_mb", "a_full", "n1_full", "n2_full",
                          "loss_full", "acc_full", "test_acc", "test_loss", "layer_fro", "b_mb", "b_full",
                          "lam", "vt_energy", "f1_full", "f2_full", "same_full", "fsum_full", "cos_rm_full",
                          "f1_mb", "f2_mb", "same_mb", "fsum_mb", "cos_rm_mb", "cos_full", "cos_mb"]}
    t = 0
    t0 = time.time()
    done = False
    for ep in range(epochs):
        perm = order_rng.permutation(Xtr.shape[0])
        for b in range(spe):
            if t >= T:
                done = True
                break
            idx = perm[b * batch:(b + 1) * batch]
            Xb = Xtr[idx] / 255.0
            yb = ytr[idx]
            losses, _, gW, gH = task_grads(Xb, yb, W, H)
            a, n1, n2 = layer_stats(gW)
            A[t], N1[t], N2[t], LOSS[t] = a, n1, n2, losses
            if t % ckpt_every == 0:
                ck["step"].append(t)
                ck["a_mb"].append(a); ck["n1_mb"].append(n1); ck["n2_mb"].append(n2); ck["loss_mb"].append(losses)
                ck["cos_mb"].append(trunk_cosine(gW))
                lf, af, gWf, _ = task_grads(Xsub, ysub, W, H)
                af_, n1f, n2f = layer_stats(gWf)
                ck["a_full"].append(af_); ck["n1_full"].append(n1f); ck["n2_full"].append(n2f)
                ck["loss_full"].append(lf); ck["acc_full"].append(af); ck["cos_full"].append(trunk_cosine(gWf))
                lt, at, _, _ = task_grads(Xte_f, yte, W, H, want_grads=False)
                ck["test_acc"].append(at); ck["test_loss"].append(lt)
                ck["layer_fro"].append([np.linalg.norm(w) for w in W])
                ck["b_mb"].append(neuron_incoming(gW)); ck["b_full"].append(neuron_incoming(gWf))
                if hess:
                    lam, vt, e = hessian_top(W, H, Xh, yh, seed=seed)
                    ck["lam"].append(lam); ck["vt_energy"].append(e)
                    for tag, g in (("full", gWf), ("mb", gW)):
                        f1, f2, same, fs, cr = eig_projection_stats(g, vt)
                        ck["f1_" + tag].append(f1); ck["f2_" + tag].append(f2); ck["same_" + tag].append(same)
                        ck["fsum_" + tag].append(fs); ck["cos_rm_" + tag].append(cr)
                if not quiet:
                    print(f"  seed {seed} eta {eta} step {t} loss {losses} test {at} {time.time()-t0:.0f}s", flush=True)
            if mode == "sum":
                u = [gW[0][l] + gW[1][l] for l in range(L)]
            else:
                cos, u, _ = pcgrad_trunk(gW)
                ACTIVE[t] = cos < 0
            for l in range(L):
                W[l] = W[l] - eta_layers[l] * u[l]
            for i in range(2):
                H[i] = H[i] - eta * gH[i]
            t += 1
        if done:
            break
    lt, at, _, _ = task_grads(Xte_f, yte, W, H, want_grads=False)
    lf, af, _, _ = task_grads(Xsub, ysub, W, H, want_grads=False)
    out = {"A": A, "N1": N1, "N2": N2, "LOSS": LOSS, "ACTIVE": ACTIVE,
           "final_test_acc": np.array(at), "final_test_loss": np.array(lt),
           "final_sub_loss": np.array(lf), "final_sub_acc": np.array(af),
           "final_W": [w.copy() for w in W], "final_H": [h.copy() for h in H], "seconds": time.time() - t0}
    for k, v in ck.items():
        if len(v):
            out["ck_" + k] = np.array(v)
    return out


# ----------------------------------------------------------------------------- conjugation check
def conjugation_check(seed=100, eta=0.05, steps=300, widths=WIDTHS_D3, c=(3.0, 1 / 3.0, 1.0)):
    """Run A: (W_0, eta). Run B: (c_l W_0, c_l^2 eta), prod c_l = 1, same batch order. Exact identity."""
    c = np.array(c, dtype=np.float64)
    assert abs(np.prod(c) - 1.0) < 1e-12
    ra = train(seed, eta, widths, mode="sum", steps=steps, hess=False)
    rb = train(seed, eta, widths, mode="sum", c=c, steps=steps, hess=False)
    rel = np.abs(ra["LOSS"] - rb["LOSS"]) / np.abs(ra["LOSS"])
    dW = [np.max(np.abs(wb - cl * wa)) for wa, wb, cl in zip(ra["final_W"], rb["final_W"], c)]
    dW_rel = [np.max(np.abs(wb - cl * wa)) / np.max(np.abs(cl * wa)) for wa, wb, cl in zip(ra["final_W"], rb["final_W"], c)]
    # Euclidean cosine differs between the two runs; sum_l eta_l a_l does not.
    cos_a = A_to_cos(ra["A"], ra["N1"], ra["N2"])
    cos_b = A_to_cos(rb["A"], rb["N1"], rb["N2"])
    inv_a = ra["A"] @ np.full(len(c), eta)
    inv_b = rb["A"] @ (c ** 2 * eta)
    return {"c": c.tolist(), "eta": eta, "steps": steps, "seed": seed,
            "max_rel_loss_diff": float(rel.max()), "max_abs_W_residual": float(max(dW)),
            "max_rel_W_residual": float(max(dW_rel)),
            "final_losses_A": ra["LOSS"][-1].tolist(), "final_losses_B": rb["LOSS"][-1].tolist(),
            "max_abs_cos_diff_between_runs": float(np.max(np.abs(cos_a - cos_b))),
            "sign_disagreements_between_runs": int(np.sum(np.sign(cos_a) != np.sign(cos_b))),
            "max_abs_invariant_diff": float(np.max(np.abs(inv_a - inv_b))),
            "max_abs_invariant": float(np.max(np.abs(inv_a)))}


def A_to_cos(A, N1, N2):
    return A.sum(axis=1) / np.sqrt((N1 ** 2).sum(axis=1) * (N2 ** 2).sum(axis=1))


# ----------------------------------------------------------------------------- the sweep
def _job(seed, eta, widths, mode):
    depth = len(widths) - 1
    os.makedirs(RUNS_DIR, exist_ok=True)
    fn = os.path.join(RUNS_DIR, f"d{depth}_{mode}_e{eta}_s{seed}.npz")
    if os.path.exists(fn):
        with np.load(fn) as z:
            return (seed, eta, depth, mode), {k: z[k] for k in z.files}
    r = train(seed, eta, widths, mode=mode, hess=(mode == "sum"))
    r.pop("final_W"); r.pop("final_H")
    r = {k: np.asarray(v) for k, v in r.items()}
    np.savez_compressed(fn + ".tmp.npz", **r)
    os.replace(fn + ".tmp.npz", fn)
    return (seed, eta, depth, mode), r


def run_all(n_jobs=N_JOBS):
    from joblib import Parallel, delayed
    t0 = time.time()
    print("conjugation check (depth 3) ...", flush=True)
    conj3 = conjugation_check()
    print(json.dumps(conj3, indent=1), flush=True)
    conj4 = conjugation_check(widths=WIDTHS_D4, c=(3.0, 1 / 3.0, 2.0, 0.5), steps=200)
    print(json.dumps(conj4, indent=1), flush=True)
    conj_rand = conjugation_check(c=(1.7, 0.4, 1 / (1.7 * 0.4)), steps=200)
    print(json.dumps(conj_rand, indent=1), flush=True)

    jobs = []
    for mode in ("sum", "pc"):
        for eta in ETAS:
            for s in (SEEDS_D3 if mode == "sum" else SEEDS_D3_PC):
                jobs.append((s, eta, WIDTHS_D3, mode))
            for s in (SEEDS_D4 if mode == "sum" else SEEDS_D4_PC):
                jobs.append((s, eta, WIDTHS_D4, mode))
    print(f"{len(jobs)} runs on {n_jobs} workers", flush=True)
    results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_job)(*j) for j in jobs)
    res = dict(results)
    print(f"sweep done in {(time.time()-t0)/60:.1f} min", flush=True)

    out = {"etas": np.array(ETAS), "seeds_d3": np.array(SEEDS_D3), "seeds_d3_pc": np.array(SEEDS_D3_PC), "seeds_d4": np.array(SEEDS_D4), "seeds_d4_pc": np.array(SEEDS_D4_PC),
           "conj_json": np.array(json.dumps({"depth3": conj3, "depth4": conj4, "depth3_random_c": conj_rand}))}
    for depth, mode, seeds in ((3, "sum", SEEDS_D3), (3, "pc", SEEDS_D3_PC), (4, "sum", SEEDS_D4), (4, "pc", SEEDS_D4_PC)):
        if True:
            keys = [k for k in res[(seeds[0], ETAS[0], depth, mode)] if k != "seconds"]
            for k in keys:
                arr = np.stack([np.stack([np.asarray(res[(s, e, depth, mode)][k]) for e in ETAS]) for s in seeds])
                out[f"d{depth}_{mode}_{k}"] = arr
            out[f"d{depth}_{mode}_seconds"] = np.array([[res[(s, e, depth, mode)]["seconds"] for e in ETAS] for s in seeds])
    # measured orthogonal/He layer-norm profile (5 seeds), global factor removed
    for depth, widths in ((3, WIDTHS_D3), (4, WIDTHS_D4)):
        prof = []
        for s in range(5):
            Wh, _ = init_mlp(np.random.default_rng(5000 + s), widths, "he")
            Wo, _ = init_mlp(np.random.default_rng(5000 + s), widths, "orthogonal")
            r = np.array([np.linalg.norm(o) / np.linalg.norm(h) for o, h in zip(Wo, Wh)])
            prof.append(r / np.exp(np.mean(np.log(r))))
        out[f"d{depth}_orth_profile"] = np.mean(prof, axis=0)
    np.savez_compressed(NPZ, **out)
    print(f"saved {NPZ} ({os.path.getsize(NPZ)/1e6:.1f} MB) after {(time.time()-t0)/60:.1f} min", flush=True)


# ============================================================================= analysis
SQ2 = np.sqrt(2.0)
PAL = {"0.01": "#2a78d6", "0.05": "#eb6834", "0.1": "#1baf7a"}      # fixed categorical order (validated default palette)
TXT, TXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e2"


def boot_ci(per_seed, fn, B=2000, seed=0):
    """Cluster bootstrap over seeds: per_seed is a list (one entry per seed) of whatever fn consumes
    as a list; fn(list_of_entries) -> scalar. Returns [point, lo, hi] with a percentile 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(per_seed)
    point = fn(per_seed)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        vals.append(fn([per_seed[i] for i in idx]))
    vals = np.array(vals, dtype=float)
    return [float(point), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def frac(per_seed):
    x = np.concatenate([np.asarray(v, dtype=float).ravel() for v in per_seed])
    return float(np.mean(x)) if x.size else float("nan")


def cos_from(A, N1, N2):
    return A.sum(-1) / np.sqrt((N1 ** 2).sum(-1) * (N2 ** 2).sum(-1))


def mixed_mask(A):
    return (A.min(-1) < 0) & (A.max(-1) > 0)


def gauge_sign_flips(a, n1, n2, C):
    """a, n1, n2: (cells, L); C: (draws, L) per-layer factors. Returns flip (cells, draws) bool,
    |cos(c)| (cells, draws) and cos at c = 1 (cells, 1)."""
    Winv = (C ** -2.0).T
    N = a @ Winv
    N0 = a.sum(-1, keepdims=True)
    G1 = (n1 ** 2) @ Winv
    G2 = (n2 ** 2) @ Winv
    cosc = N / np.sqrt(G1 * G2)
    cos0 = N0 / np.sqrt((n1 ** 2).sum(-1, keepdims=True) * (n2 ** 2).sum(-1, keepdims=True))
    flip = np.sign(N) != np.sign(N0)
    return flip, cosc, cos0


def two_layer_nearest(a):
    """Nearest flip on a two-layer path (c on layer i, 1/c on layer j): roots of a_j u^2 + R u + a_i,
    u = c^2, over all ordered pairs; the factor is max(c, 1/c)."""
    L = len(a)
    best = np.inf
    for i in range(L):
        for j in range(L):
            if i == j or a[j] == 0:
                continue
            R = a.sum() - a[i] - a[j]
            for u in np.roots([a[j], R, a[i]]):
                if abs(u.imag) < 1e-12 and u.real > 0:
                    u = u.real
                    best = min(best, max(np.sqrt(u), 1 / np.sqrt(u)))
    return float(best)


def unconstrained_nearest(a):
    """Box max_l |log c_l| <= t without the product constraint: t* = (1/4) log(sum_maj|a| / sum_min|a|)."""
    s = np.sign(a.sum())
    maj = np.abs(a[np.sign(a) == s]).sum()
    mn = np.abs(a[np.sign(a) == -s]).sum()
    if mn == 0:
        return float("inf")
    return float(np.exp(0.25 * np.log(maj / mn)))


def _slice_grid(L, t, n):
    g = np.linspace(-t, t, n)
    mesh = np.meshgrid(*([g] * (L - 1)), indexing="ij")
    U = np.stack([m.ravel() for m in mesh], axis=1)
    last = -U.sum(1, keepdims=True)
    U = np.concatenate([U, last], axis=1)
    return U[np.abs(last[:, 0]) <= t]


def constrained_nearest(a, n=None):
    """Nearest sign flip over the box max_l |log c_l| <= t WITH prod c_l = 1: bisection on t over a
    grid of the slice sum log c = 0 (a feasible-point upper bound within ~1% in factor), bracketed
    below by the unconstrained closed form and above by the two-layer value."""
    L = len(a)
    n = n or (201 if L == 3 else 61)
    s = np.sign(a.sum())
    lo = np.log(unconstrained_nearest(a))
    hi = np.log(two_layer_nearest(a))
    if not np.isfinite(hi):
        return float("inf")

    def flips(t):
        U = _slice_grid(L, t, n)
        return bool(np.any(s * (np.exp(-2 * U) @ a) <= 0))
    k = 0
    while not flips(hi) and k < 20:
        hi *= 1.05
        k += 1
    if not flips(hi):
        return float(np.exp(hi))
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        if flips(mid):
            hi = mid
        else:
            lo = mid
    return float(np.exp(hi))


def named_gauges(L, orth_profile):
    g = {}
    for c1 in (0.25, 0.5, 2.0, 4.0):
        c = np.ones(L)
        c[0] = c1
        g[f"input_scale_{c1}"] = c
    for d in (0.9, 0.8, 0.65, 0.5):
        g[f"step_decay_to_input_d{d}"] = np.array([d ** ((L - 1 - l) / 2) for l in range(L)])
        g[f"step_growth_to_input_d{d}"] = np.array([d ** (l / 2) for l in range(L)])
    g["orthogonal_vs_he_measured"] = np.asarray(orth_profile, dtype=float)
    return g


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIGDIR, exist_ok=True)
    Z = np.load(NPZ, allow_pickle=False)
    etas = Z["etas"].tolist()
    out = {"conjugation": json.loads(str(Z["conj_json"])), "etas": etas, "batch": BATCH, "epochs": EPOCHS,
           "checkpoint_every": CKPT_EVERY, "hessian_batch": HESS_N, "hessian_iters": HESS_ITERS,
           "subset_for_full_gradient": SUB_N, "seeds_d3": Z["seeds_d3"].tolist(), "seeds_d3_pc": Z["seeds_d3_pc"].tolist(), "seeds_d4": Z["seeds_d4"].tolist(),
           "seeds_d4_pc": Z["seeds_d4_pc"].tolist(), "depths": {}}
    rng = np.random.default_rng(12345)
    fig_data = {}
    for depth in (3, 4):
        P = f"d{depth}_sum_"
        A, N1, N2, LOSS = Z[P + "A"], Z[P + "N1"], Z[P + "N2"], Z[P + "LOSS"]
        S, E, T, L = A.shape
        ck_step = Z[P + "ck_step"][0, 0]
        K = len(ck_step)
        D = {"n_seeds": S, "steps": T, "checkpoints": ck_step.tolist(), "layers": L,
             "orth_profile": Z[f"d{depth}_orth_profile"].tolist(),
             "seconds_per_run_median": float(np.median(Z[P + "seconds"])), "per_eta": {}}
        finite = np.isfinite(LOSS).all(axis=(2, 3))
        D["runs_with_nonfinite_loss"] = int((~finite).sum())
        cos = cos_from(A, N1, N2)                                  # (S,E,T) minibatch per-step cosine
        neg = cos < 0
        spe = T // EPOCHS
        for ei, eta in enumerate(etas):
            key = str(eta)
            R = {}
            R["conflict_rate"] = boot_ci([neg[s, ei] for s in range(S)], frac)
            R["conflict_rate_post_step0"] = boot_ci([neg[s, ei, 1:] for s in range(S)], frac)
            R["conflict_rate_by_epoch"] = [boot_ci([neg[s, ei, e * spe:(e + 1) * spe] for s in range(S)], frac) for e in range(EPOCHS)]
            R["conflict_rate_by_100_steps"] = [float(neg[:, ei, k:k + 100].mean()) for k in range(0, T, 100)]
            R["median_cos_by_epoch"] = [float(np.median(cos[:, ei, e * spe:(e + 1) * spe])) for e in range(EPOCHS)]
            R["median_cos_all_steps"] = float(np.median(cos[:, ei]))
            R["final_minibatch_losses_median"] = np.median(LOSS[:, ei, -1], axis=0).tolist()
            R["final_test_acc_mean"] = Z[P + "final_test_acc"][:, ei].mean(0).tolist()
            R["test_acc_at_checkpoints_mean"] = Z[P + "ck_test_acc"][:, ei].mean(0).tolist()
            R["mixed_fraction_all_steps"] = boot_ci([mixed_mask(A[s, ei]) for s in range(S)], frac)
            R["mixed_fraction_all_steps_by_epoch"] = [float(mixed_mask(A[:, ei, e * spe:(e + 1) * spe]).mean()) for e in range(EPOCHS)]
            for tag in ("mb", "full"):
                a = Z[P + f"ck_a_{tag}"][:, ei]
                n1 = Z[P + f"ck_n1_{tag}"][:, ei]
                n2 = Z[P + f"ck_n2_{tag}"][:, ei]   # (S,K,L)
                post = np.zeros((S, K), bool)
                post[:, 1:] = True
                mix = mixed_mask(a)
                negc = a.sum(-1) < 0
                cc = cos_from(a, n1, n2)
                C = {"n_cells": int(S * K),
                     "mixed_fraction": boot_ci([mix[s] for s in range(S)], frac),
                     "mixed_fraction_post_init": boot_ci([mix[s, 1:] for s in range(S)], frac),
                     "mixed_by_checkpoint": mix.mean(0).tolist(),
                     "conflict_rate_cells": boot_ci([negc[s] for s in range(S)], frac),
                     "conflict_rate_cells_post_init": boot_ci([negc[s, 1:] for s in range(S)], frac),
                     "conflict_by_checkpoint": negc.mean(0).tolist(),
                     "median_cos_by_checkpoint": np.median(cc, axis=0).tolist(),
                     "median_abs_cos_mixed": float(np.median(np.abs(cc[mix]))) if mix.any() else None,
                     "median_abs_cos_single_sign": float(np.median(np.abs(cc[~mix]))) if (~mix).any() else None}
                pats, cnts = np.unique((np.sign(a) < 0).reshape(-1, L).astype(int), axis=0, return_counts=True)
                C["layer_sign_pattern_counts"] = {"".join("-" if x else "+" for x in p): int(c) for p, c in zip(pats, cnts)}
                C["mean_abs_a_by_layer"] = np.mean(np.abs(a), axis=(0, 1)).tolist()
                C["frac_a_negative_by_layer"] = (a < 0).mean(axis=(0, 1)).tolist()
                af, n1f, n2f = a.reshape(-1, L), n1.reshape(-1, L), n2.reshape(-1, L)
                seed_of = np.repeat(np.arange(S), K)
                postf = post.ravel()
                G = {}
                for name, c in named_gauges(L, D["orth_profile"]).items():
                    flip, cosc, cos0 = gauge_sign_flips(af, n1f, n2f, c[None])
                    fl = flip[:, 0]
                    G[name] = {"c": c.tolist(), "flips": int(fl.sum()), "flips_post_init": int(fl[postf].sum()),
                               "flip_fraction": boot_ci([fl[seed_of == s] for s in range(S)], frac),
                               "conflict_rate_in_gauge": float((cosc[:, 0] < 0).mean())}
                for lim, nm in ((SQ2, "sqrt2"), (2.0, "2"), (4.0, "4")):
                    Cr = np.exp(rng.uniform(-np.log(lim), np.log(lim), size=(2000, L)))
                    flip, cosc, cos0 = gauge_sign_flips(af, n1f, n2f, Cr)
                    per_cell = flip.mean(1)
                    either = np.maximum(np.abs(cosc), np.abs(cos0))[flip]
                    G[f"random_logU_within_{nm}"] = {
                        "mean_flip_fraction": boot_ci([per_cell[seed_of == s] for s in range(S)], frac),
                        "mean_flip_fraction_post_init": float(per_cell[postf].mean()),
                        "max_flip_fraction_over_draws": float(flip.mean(0).max()),
                        "conflict_rate_range_over_draws": [float((cosc < 0).mean(0).min()), float((cosc < 0).mean(0).max())],
                        "cells_ever_flipped": int(flip.any(1).sum()),
                        "abs_cos_either_side_where_flipped": {"max": float(either.max()) if either.size else None,
                                                             "median": float(np.median(either)) if either.size else None}}
                npts = 25 if L == 3 else 13
                g = np.linspace(-0.5 * np.log(2), 0.5 * np.log(2), npts)
                Cb = np.exp(np.stack([m.ravel() for m in np.meshgrid(*([g] * L), indexing="ij")], 1))
                sup, cells_flip, per_cell_max = 0.0, 0, []
                for lo_ in range(0, af.shape[0], 64):
                    flip, cosc, cos0 = gauge_sign_flips(af[lo_:lo_ + 64], n1f[lo_:lo_ + 64], n2f[lo_:lo_ + 64], Cb)
                    either = np.where(flip, np.maximum(np.abs(cosc), np.abs(cos0)), 0.0)
                    m = either.max(1)
                    per_cell_max += m[flip.any(1)].tolist()
                    sup = max(sup, float(m.max()))
                    cells_flip += int(flip.any(1).sum())
                G["box_sqrt2_supremum_abs_cos_either_side"] = {"sup": sup, "cells_flipping_in_box": cells_flip,
                                                              "per_cell_max_median": float(np.median(per_cell_max)) if per_cell_max else None}
                C["induced_gauges"] = G
                mixf = mix.ravel()
                near = {"two_layer": [], "unconstrained": [], "constrained": [], "seed": [], "post": []}
                for i in np.where(mixf)[0]:
                    near["two_layer"].append(two_layer_nearest(af[i]))
                    near["unconstrained"].append(unconstrained_nearest(af[i]))
                    near["constrained"].append(constrained_nearest(af[i]))
                    near["seed"].append(int(seed_of[i]))
                    near["post"].append(bool(postf[i]))
                NF = {"n_mixed": int(mixf.sum())}
                for k in ("two_layer", "unconstrained", "constrained"):
                    v = np.array(near[k])
                    if v.size:
                        NF[k] = {"median_factor": float(np.median(v)),
                                 "quartiles": [float(np.percentile(v, 25)), float(np.percentile(v, 75))]}
                        for lim, nm in ((SQ2, "sqrt2"), (2.0, "2"), (3.0, "3"), (10.0, "10")):
                            NF[k][f"cells_within_{nm}"] = int((v <= lim).sum())
                            hits = np.zeros(af.shape[0], bool)
                            hits[np.where(mixf)[0]] = v <= lim
                            NF[k][f"fraction_of_all_cells_within_{nm}"] = boot_ci([hits[seed_of == s] for s in range(S)], frac)
                    else:
                        NF[k] = None
                C["nearest_flip"] = NF
                fig_data[(depth, eta, tag)] = np.array(near["constrained"]) if near["constrained"] else np.array([])
                b = Z[P + f"ck_b_{tag}"][:, ei]                     # (S,K,L,width)
                allsame = np.array([[bool((b[s, k] >= 0).all() or (b[s, k] <= 0).all()) for k in range(K)] for s in range(S)])
                C["neuron_precondition_holds_cells"] = int(allsame.sum())
                C["neuron_frac_negative_incoming_median"] = float(np.median((b < 0).mean(axis=(2, 3))))
                C["neuron_frac_negative_incoming_range"] = [float((b < 0).mean(axis=(2, 3)).min()), float((b < 0).mean(axis=(2, 3)).max())]
                R["cells_" + tag] = C
            smb = np.sign(Z[P + "ck_a_mb"][:, ei].sum(-1))
            sfl = np.sign(Z[P + "ck_a_full"][:, ei].sum(-1))
            R["sign_agreement_minibatch_vs_full_at_checkpoints"] = float((smb == sfl).mean())
            lam = Z[P + "ck_lam"][:, ei]
            el = eta * lam
            R["eta_lambda_max_median_by_checkpoint"] = np.median(el, axis=0).tolist()
            R["eta_lambda_max_post_init"] = {"median": float(np.median(el[:, 1:])), "q10": float(np.percentile(el[:, 1:], 10)),
                                             "q90": float(np.percentile(el[:, 1:], 90)),
                                             "frac_in_1.8_2.2": float(((el[:, 1:] > 1.8) & (el[:, 1:] < 2.2)).mean()),
                                             "frac_above_2.2": float((el[:, 1:] > 2.2).mean())}
            R["hessian_top_vector_trunk_energy_median"] = float(np.median(Z[P + "ck_vt_energy"][:, ei]))
            for tag in ("full", "mb"):
                f1 = Z[P + f"ck_f1_{tag}"][:, ei]
                f2 = Z[P + f"ck_f2_{tag}"][:, ei]
                fs = Z[P + f"ck_fsum_{tag}"][:, ei]
                cr = Z[P + f"ck_cos_rm_{tag}"][:, ei]
                c0 = Z[P + f"ck_cos_{tag}"][:, ei]
                same = Z[P + f"ck_same_{tag}"][:, ei]
                R[f"eigvec_{tag}"] = {"energy_along_v_task1_median_post_init": float(np.median(f1[:, 1:])),
                                      "energy_along_v_task2_median_post_init": float(np.median(f2[:, 1:])),
                                      "summed_gradient_energy_along_v_median_post_init": float(np.median(fs[:, 1:])),
                                      "same_side_post_init": float(same[:, 1:].mean()),
                                      "conflict_post_init_raw": float((c0[:, 1:] < 0).mean()),
                                      "conflict_post_init_v_removed": float((cr[:, 1:] < 0).mean()),
                                      "median_cos_post_init_raw": float(np.median(c0[:, 1:])),
                                      "median_cos_post_init_v_removed": float(np.median(cr[:, 1:])),
                                      "cells_changing_sign_post_init": int((np.sign(c0[:, 1:]) != np.sign(cr[:, 1:])).sum())}
            fro = Z[P + "ck_layer_fro"][:, ei]
            R["layer_fro_max_over_min_median"] = float(np.median(fro.max(-1) / fro.min(-1)))
            R["layer_fro_max_over_min_step0"] = float(np.median((fro.max(-1) / fro.min(-1))[:, 0]))
            R["layer_fro_final_over_initial_median"] = np.median(fro[:, -1] / fro[:, 0], axis=0).tolist()
            Q = f"d{depth}_pc_"
            Spc = Z[Q + "final_test_acc"].shape[0]
            pc_acc = Z[Q + "final_test_acc"][:, ei]
            sm_acc = Z[P + "final_test_acc"][:Spc, ei]
            diff = pc_acc - sm_acc
            R["pcgrad"] = {"n_seeds": int(Spc),
                           "summed_test_acc_mean": sm_acc.mean(0).tolist(), "pcgrad_test_acc_mean": pc_acc.mean(0).tolist(),
                           "paired_diff_digit_pct_points": [x * 100 for x in boot_ci([diff[s, 0] for s in range(Spc)], frac)],
                           "paired_diff_parity_pct_points": [x * 100 for x in boot_ci([diff[s, 1] for s in range(Spc)], frac)],
                           "seeds_pcgrad_better_digit": int((diff[:, 0] > 0).sum()), "seeds_pcgrad_better_parity": int((diff[:, 1] > 0).sum()),
                           "pcgrad_active_fraction": boot_ci([Z[Q + "ACTIVE"][s, ei] for s in range(Spc)], frac),
                           "pcgrad_test_acc_at_checkpoints_mean": Z[Q + "ck_test_acc"][:, ei].mean(0).tolist(),
                           "summed_test_acc_at_checkpoints_mean_same_seeds": Z[P + "ck_test_acc"][:Spc, ei].mean(0).tolist(),
                           "pcgrad_final_sub_loss_median": np.median(Z[Q + "final_sub_loss"][:, ei], 0).tolist(),
                           "summed_final_sub_loss_median_same_seeds": np.median(Z[P + "final_sub_loss"][:Spc, ei], 0).tolist()}
            D["per_eta"][key] = R
        pooled = {}
        for tag in ("mb", "full"):
            a = Z[P + f"ck_a_{tag}"]
            mix = mixed_mask(a)
            pooled[tag] = {"n_cells": int(mix.size), "mixed_fraction": boot_ci([mix[s] for s in range(S)], frac),
                           "mixed_fraction_post_init": boot_ci([mix[s, :, 1:] for s in range(S)], frac),
                           "conflict_rate_cells": boot_ci([(a.sum(-1) < 0)[s] for s in range(S)], frac)}
        pooled["conflict_rate_all_steps"] = boot_ci([neg[s] for s in range(S)], frac)
        D["pooled_over_eta"] = pooled
        out["depths"][str(depth)] = D
        if depth == 3:
            fig_data["cos"] = cos
            fig_data["lam"] = Z[P + "ck_lam"]
            fig_data["ck_step"] = ck_step
    make_figures(out, fig_data, etas, plt)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    print(f"wrote {JSON_OUT}")
    return out


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=TXT2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def make_figures(out, F, etas, plt):
    cos, lam, ck = F["cos"], F["lam"], F["ck_step"]
    S, E, T = cos.shape
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 2.9), dpi=160, facecolor="#fcfcfb")
    for ei, eta in enumerate(etas):
        col = PAL[str(eta)]
        w = 100
        r = np.array([(cos[:, ei, k:k + w] < 0).mean() for k in range(0, T, w)])
        axs[0].plot(np.arange(0, T, w) + w // 2, np.maximum(r, 2e-4), color=col, lw=2, marker="o", ms=3.5, label=f"eta = {eta}")
        el = eta * lam[:, ei]
        med = np.median(el, 0)
        axs[1].plot(ck, med, color=col, lw=2, marker="o", ms=3.5, label=f"eta = {eta}")
        axs[1].fill_between(ck, np.percentile(el, 25, 0), np.percentile(el, 75, 0), color=col, alpha=0.15, lw=0)
    axs[0].set_xlim(0, T)
    axs[0].set_yscale("log")
    axs[0].set_ylim(2e-4, 1)
    axs[0].legend(frameon=False, fontsize=8)
    axs[0].set_xlabel("SGD step (batch 128; 468 steps per epoch)", color=TXT2, fontsize=8)
    axs[0].set_ylabel("conflict rate per 100-step window", color=TXT2, fontsize=8)
    axs[0].set_title("Minibatch trunk cosine < 0 over 20 seeds\n(log scale; points at the 2e-4 floor are zero)", fontsize=9, color=TXT, loc="left")
    axs[1].axhline(2.0, color=TXT2, lw=0.8, ls=":")
    axs[1].text(ck[-1], 2.05, "eta*lambda = 2", fontsize=7, color=TXT2, ha="right")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("SGD step", color=TXT2, fontsize=8)
    axs[1].set_ylabel("eta * lambda_max (fixed 512-example batch)", color=TXT2, fontsize=8)
    axs[1].set_title("Sharpness: median and IQR over seeds", fontsize=9, color=TXT, loc="left")
    axs[1].legend(frameon=False, fontsize=8)
    for ax in axs:
        _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "mnist_real_conflict.png"))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.2, 2.9), dpi=160, facecolor="#fcfcfb")
    grid = np.exp(np.linspace(0, np.log(30), 300))
    n_cells = out["depths"]["3"]["per_eta"][str(etas[0])]["cells_mb"]["n_cells"]
    for eta in etas:
        v = F[(3, eta, "mb")]
        y = np.array([(v <= g).sum() / n_cells for g in grid])
        ax.plot(grid, y, color=PAL[str(eta)], lw=2, label=f"eta = {eta}")
    for x, nm in ((SQ2, "sqrt2"), (2.0, "2")):
        ax.axvline(x, color=TXT2, lw=0.8, ls=":")
        ax.text(x * 1.03, 0.96, nm, fontsize=7, color=TXT2, transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_xlim(1, 45)
    ax.set_ylim(0, 0.2)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlabel("largest per-layer factor of the nearest flipping gauge (prod c = 1)", color=TXT2, fontsize=8)
    ax.set_ylabel("fraction of checkpoint cells flipped", color=TXT2, fontsize=8)
    ax.set_title("MNIST depth 3, minibatch gradients, 300 cells per eta", fontsize=9, color=TXT, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "mnist_real_nearest_flip.png"))
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=160, facecolor="#fcfcfb")
    for ti, name in enumerate(("digit (10-way CE)", "parity (logistic)")):
        ax = axs[ti]
        for ei, eta in enumerate(etas):
            pc = out["depths"]["3"]["per_eta"][str(eta)]["pcgrad"]
            d = pc["paired_diff_digit_pct_points" if ti == 0 else "paired_diff_parity_pct_points"]
            ax.errorbar([ei], [d[0]], yerr=[[d[0] - d[1]], [d[2] - d[0]]], fmt="o", color=PAL[str(eta)], ms=6, capsize=3, lw=1.5)
            acc = pc["summed_test_acc_mean"][ti] * 100
            ax.annotate(f"\nsummed {acc:.2f}%", (ei, d[1]), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7, color=TXT2)
        ax.axhline(0, color=TXT2, lw=0.8)
        ax.set_xlim(-0.5, len(etas) - 0.5)
        lo_, hi_ = min(x["paired_diff_digit_pct_points" if ti == 0 else "paired_diff_parity_pct_points"][1] for x in
                       (out["depths"]["3"]["per_eta"][str(e)]["pcgrad"] for e in etas)), 0
        ax.set_ylim(lo_ - 0.04, None)
        ax.set_xticks(range(len(etas)))
        ax.set_xticklabels([f"eta = {e}" for e in etas])
        ax.set_ylabel("PCGrad - summed test accuracy (pct points)", color=TXT2, fontsize=8)
        ax.set_title(f"{name}: paired over seeds, 95% CI", fontsize=9, color=TXT, loc="left")
        _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "mnist_real_pcgrad.png"))
    plt.close(fig)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("run", "all") and not os.path.exists(NPZ):
        run_all()
    if stage in ("analyze", "all"):
        analyze()
