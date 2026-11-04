#!/usr/bin/env python3
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import io
import json
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mgp.benchmark import _instantiate_autoencoder, _instantiate_scaler

torch.set_default_dtype(torch.float64)
EXPLORATORY_INPUTS = REPO_ROOT / "artifacts" / "exploratory_decoders"
SEALED = REPO_ROOT / "eval" / "prospective_weight_validation_2026-07-17" / "first_look_input" / "sealed"
SUPPLEMENT = REPO_ROOT / "supplementary_data"

SILU_ARGMIN = -1.2784645427610738
DSILU_ARGMAX = 2.3993572805154676
SILU_RELU_GAP = 0.2784645427610738
ROUNDOFF = 1e-9


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def silu(x):
    return x * sigmoid(x)


def dsilu(x):
    s = sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def silu_interval(lo, hi):
    return silu(np.clip(SILU_ARGMIN, lo, hi)), np.maximum(silu(lo), silu(hi))


def dsilu_interval(lo, hi):
    values = [dsilu(lo), dsilu(hi)]
    for x in (DSILU_ARGMAX, -DSILU_ARGMAX):
        values.append(np.where((lo <= x) & (x <= hi), dsilu(x), np.nan))
    values = np.stack(values)
    return np.nanmin(values, 0), np.nanmax(values, 0)


def decoder_layers(ae, scaler) -> list[tuple[np.ndarray, np.ndarray]]:
    with torch.no_grad():
        if ae.architecture_name == "mlp":
            linear = [m for m in ae.decoder if isinstance(m, torch.nn.Linear)]
            layers = [(m.weight.numpy().copy(), m.bias.numpy().copy()) for m in linear]
        else:
            def first(z):
                features = ae.decoder_projection(z).reshape(z.shape[0], ae.decoder_start_channels, ae.input_dim)
                return ae.decoder_conv[0](features).reshape(z.shape[0], -1)

            a = first(torch.zeros(1, ae.latent_dim))[0].numpy()
            A = np.stack([first(torch.eye(ae.latent_dim)[j:j + 1])[0].numpy() - a for j in range(ae.latent_dim)], 1)
            last = ae.decoder_conv[2]
            width = A.shape[0]
            b = last(torch.zeros(1, ae.decoder_start_channels, ae.input_dim)).reshape(-1).numpy()
            basis = torch.eye(width).reshape(width, ae.decoder_start_channels, ae.input_dim)
            W = (last(basis).reshape(width, -1).numpy() - b).T
            layers = [(A, a), (W, b)]
    std, mean = scaler.std.numpy(), scaler.mean.numpy()
    W, b = layers[-1]
    layers[-1] = (std[:, None] * W, std * b + mean)
    return layers


def forward(layers, Z):
    h = Z.T
    for k, (W, b) in enumerate(layers):
        h = W @ h + b[:, None]
        if k < len(layers) - 1:
            h = silu(h)
    return h.T


def forward_jacobian(layers, z):
    h, J = z.copy(), np.eye(len(z))
    for k, (W, b) in enumerate(layers):
        a = W @ h + b
        J = W @ J
        if k < len(layers) - 1:
            J = dsilu(a)[:, None] * J
            h = silu(a)
        else:
            h = a
    return h, J


def box_bounds(layers, center, half, u, norm_u):
    m, r = center.T, half.T
    n_box = center.shape[0]
    Jm = np.broadcast_to(np.eye(2)[:, :, None], (2, 2, n_box)).copy()
    Jr = np.zeros_like(Jm)
    hc, Jc = center.T.copy(), Jm.copy()
    for k, (W, b) in enumerate(layers):
        am, ar = W @ m + b[:, None], np.abs(W) @ r
        Jm, Jr = np.tensordot(W, Jm, axes=1), np.tensordot(np.abs(W), Jr, axes=1)
        ac, Jc = W @ hc + b[:, None], np.tensordot(W, Jc, axes=1)
        if k < len(layers) - 1:
            lo, hi = silu_interval(am - ar, am + ar)
            m, r = 0.5 * (lo + hi), 0.5 * (hi - lo)
            dlo, dhi = dsilu_interval(am - ar, am + ar)
            dm, dr = 0.5 * (dlo + dhi), 0.5 * (dhi - dlo)
            Jr = np.abs(dm)[:, None, :] * Jr + dr[:, None, :] * (np.abs(Jm) + Jr)
            Jm = dm[:, None, :] * Jm
            Jc = dsilu(ac)[:, None, :] * Jc
            hc = silu(ac)
        else:
            m, r, hc = am, ar, ac
    gap = np.maximum(0.0, np.maximum((m - r) - u[:, None], u[:, None] - (m + r)))
    lower_interval = np.sqrt((gap ** 2).sum(0)) / norm_u
    res = hc - u[:, None]
    rho = np.einsum("ikb,kb->ib", np.abs(Jm - Jc) + Jr, half.T)
    lower_taylor = (linearized_box_minimum(Jc, res, half) - np.sqrt((rho ** 2).sum(0))) / norm_u
    return np.maximum(lower_interval, lower_taylor), np.sqrt((res ** 2).sum(0)) / norm_u


def linearized_box_minimum(J, res, half):
    Q, Rm = np.linalg.qr(np.moveaxis(J, 2, 0))
    resb = res.T
    b = np.einsum("bnk,bn->bk", Q, resb)
    perp = resb - np.einsum("bnk,bk->bn", Q, b)
    perp2 = (perp ** 2).sum(1)
    hx, hy = half[:, 0], half[:, 1]
    c0, c1 = Rm[:, :, 0], Rm[:, :, 1]

    def value(x, y):
        v = b + c0 * x[:, None] + c1 * y[:, None]
        return (v ** 2).sum(1)

    best = np.full(len(b), np.inf)
    det = Rm[:, 0, 0] * Rm[:, 1, 1] - Rm[:, 0, 1] * Rm[:, 1, 0]
    ok = np.abs(det) > 1e-14 * np.maximum((Rm ** 2).sum((1, 2)), 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        x0 = np.where(ok, (-b[:, 0] * Rm[:, 1, 1] + b[:, 1] * Rm[:, 0, 1]) / det, 0.0)
        y0 = np.where(ok, (b[:, 0] * Rm[:, 1, 0] - b[:, 1] * Rm[:, 0, 0]) / det, 0.0)
    inside = ok & (np.abs(x0) <= hx) & (np.abs(y0) <= hy)
    best = np.where(inside, value(x0, y0), best)
    n0, n1 = np.maximum((c0 ** 2).sum(1), 1e-300), np.maximum((c1 ** 2).sum(1), 1e-300)
    for s in (-1.0, 1.0):
        y = s * hy
        x = np.clip(-((b + c1 * y[:, None]) * c0).sum(1) / n0, -hx, hx)
        best = np.minimum(best, value(x, y))
        x = s * hx
        y = np.clip(-((b + c0 * x[:, None]) * c1).sum(1) / n1, -hy, hy)
        best = np.minimum(best, value(x, y))
    return np.sqrt(perp2 + best)


def polish(layers, u, z0):
    def objective(z):
        D, J = forward_jacobian(layers, z)
        r = D - u
        return r @ r, 2.0 * J.T @ r

    return minimize(objective, z0, jac=True, method="L-BFGS-B",
                    options={"maxiter": 200, "ftol": 1e-15, "gtol": 1e-12}).x


def outer_constant(layers) -> float:
    delta = 0.0
    for W, b in layers[:-1]:
        delta = np.linalg.norm(W, 2) * delta + np.linalg.norm(b) + SILU_RELU_GAP * np.sqrt(len(b))
    return float(np.linalg.norm(layers[-1][0], 2) * delta)


def certified_gamma(layers, n_dir: int = 1_000_000) -> float:
    lipschitz = float(np.prod([np.linalg.norm(W, 2) for W, _ in layers]))
    theta = np.linspace(0.0, 2.0 * np.pi, n_dir, endpoint=False)
    smallest = np.inf
    for i in range(0, n_dir, 20_000):
        h = np.stack([np.cos(theta[i:i + 20_000]), np.sin(theta[i:i + 20_000])])
        for W, _ in layers[:-1]:
            h = np.maximum(W @ h, 0.0)
        smallest = min(smallest, float(np.linalg.norm(layers[-1][0] @ h, axis=0).min()))
    return smallest - lipschitz * np.pi / n_dir


def branch_and_bound(layers, u, lo, hi, delta, atol=0.0, n0=64, max_boxes=5_000_000):
    norm_u = np.linalg.norm(u)
    batch = max(256, 3_000_000 // max(len(b) for _, b in layers))
    edges = [np.linspace(lo[j], hi[j], n0 + 1) for j in range(2)]
    cx, cy = np.meshgrid(0.5 * (edges[0][1:] + edges[0][:-1]), 0.5 * (edges[1][1:] + edges[1][:-1]), indexing="ij")
    center = np.stack([cx.ravel(), cy.ravel()], 1)
    half = np.tile([(hi[0] - lo[0]) / (2 * n0), (hi[1] - lo[1]) / (2 * n0)], (len(center), 1))
    best, z_best = np.inf, None
    lower_done, n_boxes = np.inf, 0
    while len(center):
        parts = [box_bounds(layers, center[i:i + batch], half[i:i + batch], u, norm_u)
                 for i in range(0, len(center), batch)]
        lower = np.concatenate([p[0] for p in parts]) * (1.0 - ROUNDOFF)
        at_center = np.concatenate([p[1] for p in parts])
        n_boxes += len(center)
        k = int(np.argmin(at_center))
        if at_center[k] < best:
            z = polish(layers, u, center[k])
            e = float(np.linalg.norm(forward(layers, z[None])[0] - u) / norm_u)
            best, z_best = (e, z) if e < at_center[k] else (float(at_center[k]), center[k].copy())
        keep = lower < (1.0 - delta) * best - atol
        if (~keep).any():
            lower_done = min(lower_done, float(lower[~keep].min()))
        center, half = center[keep], half[keep]
        if n_boxes > max_boxes and len(center):
            return dict(best=best, z=z_best, lower=min(lower_done, float(lower[keep].min())), boxes=n_boxes, capped=True)
        rows = np.arange(len(center))
        axis = np.argmax(half, 1)
        shift = np.zeros_like(half)
        shift[rows, axis] = half[rows, axis] / 2
        half = half.copy()
        half[rows, axis] /= 2
        center, half = np.concatenate([center - shift, center + shift]), np.concatenate([half, half])
    return dict(best=best, z=z_best, lower=lower_done, boxes=n_boxes, capped=False)


def certify_case(layers, u, target, gamma, C, delta, max_radius=1e6):
    norm_u = np.linalg.norm(u)
    radius = (C + np.linalg.norm(layers[-1][1] - u) + target * norm_u) / gamma if gamma > 0 else np.inf
    out = dict(radius=radius, outer_ok=bool(np.isfinite(radius) and radius <= max_radius))
    R = radius if out["outer_ok"] else max_radius
    bb = branch_and_bound(layers, u, np.array([-R, -R]), np.array([R, R]), delta)
    lower = min(bb["lower"], target) if out["outer_ok"] else bb["lower"]
    out.update(e_best=bb["best"], e_lower=lower, z1=float(bb["z"][0]), z2=float(bb["z"][1]),
               boxes=bb["boxes"], capped=bb["capped"])
    return out


def check_layers(layers, ae, scaler, Z):
    with torch.no_grad():
        ref = scaler.inverse_transform_tensor(ae.decode(torch.as_tensor(Z))).numpy()
    err = np.abs(forward(layers, Z) - ref).max() / np.abs(ref).max()
    if err > 1e-10:
        raise RuntimeError(f"decoder export mismatch {err:.2e}")


def load_exploratory(path: Path):
    with np.load(path, allow_pickle=False) as npz:
        meta = json.loads(str(npz["meta"]))
        state = {k[3:]: torch.as_tensor(npz[k]) for k in npz.files if k.startswith("sd/")}
        artifact = {"config": meta["config"], "autoencoder_state_dict": state,
                    "scaler": {"mean": npz["scaler_mean"], "std": npz["scaler_std"]}}
        refs = npz["references"]
    known = {}
    with open(SUPPLEMENT / "latent_search_exploratory_cases.csv") as fh:
        for row in csv.DictReader(fh):
            if (row["family"], row["regime"], int(row["seed"])) == (meta["family"], meta["regime"], meta["seed"]):
                known[int(row["sample"])] = row
    cases = []
    for i, u in enumerate(refs):
        row = known[i]
        cases.append((i, u, float(row["e_glob"]), {"e_rep": row["e_rep"], "e_glob": row["e_glob"], "mg": row["mg"]}))
    return meta["family"], meta["regime"], meta["seed"], artifact, cases


def load_block1(path: Path):
    with tarfile.open(path, "r:gz") as tar:
        members = tar.getmembers()

        def read(suffix):
            return tar.extractfile(next(m for m in members if m.name.endswith(suffix))).read()

        result = json.loads(read(".result.json"))
        artifact = torch.load(io.BytesIO(read(".offline_artifact.pt")), map_location="cpu", weights_only=False)
    cfg = result["config"]
    family, regime, seed = cfg["problem_name"], cfg["test_regime"], int(cfg["seed"])
    known = {}
    with open(SUPPLEMENT / "representation_capacity_cases.csv") as fh:
        for row in csv.DictReader(fh):
            if (row["family"], row["regime"], int(row["seed"])) == (family, regime, seed):
                known[int(row["sample"])] = row
    cases = []
    for i, case in enumerate(result["cases"]):
        row = known[i]
        u = np.asarray(case["full_order"]["state"], dtype=np.float64)
        cases.append((i, u, float(row["e_pol"]),
                      {k: row[k] for k in ("e_pol", "e_proj", "pod_g", "mg", "precond_m_lspg")}))
    return family, regime, seed, artifact, cases


def run(job):
    which, path, delta = job
    t0 = time.time()
    family, regime, seed, artifact, cases = (load_exploratory if which == "exploratory" else load_block1)(Path(path))
    ae, scaler = _instantiate_autoencoder(artifact), _instantiate_scaler(artifact)
    layers = decoder_layers(ae, scaler)
    check_layers(layers, ae, scaler, np.random.default_rng(0).normal(0, 3, (16, 2)))
    gamma, C = certified_gamma(layers), outer_constant(layers)
    rows = []
    for i, u, target, extra in cases:
        out = certify_case(layers, u, target, gamma, C, delta)
        rows.append({"family": family, "regime": regime, "seed": seed, "sample": i, "architecture": ae.architecture_name,
                     **extra, "e_best": out["e_best"], "e_lower": out["e_lower"],
                     "lower_over_best": out["e_lower"] / out["e_best"], "z1": out["z1"], "z2": out["z2"],
                     "radius": out["radius"], "gamma": gamma, "outer_ok": out["outer_ok"],
                     "boxes": out["boxes"], "capped": out["capped"]})
    print(f"{Path(path).name} {time.time() - t0:.0f}s", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("exploratory", "block1"), required=True)
    ap.add_argument("--delta", type=float, default=1e-3)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    paths = sorted(EXPLORATORY_INPUTS.glob("*.npz")) if args.set == "exploratory" else sorted(SEALED.glob("*.tar.gz"))
    if args.limit:
        paths = paths[: args.limit]
    jobs = [(args.set, str(p), args.delta) for p in paths]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            results = list(pool.map(run, jobs))
    else:
        results = [run(j) for j in jobs]
    rows = [r for rs in results for r in rs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
