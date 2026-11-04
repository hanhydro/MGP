#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import compute_projected_residuals as prj
from mgp.problems import create_problem

torch.set_default_dtype(torch.float64)
SHARDS = REPO_ROOT / "artifacts" / "cache" / "manifold_oracle_shards"


def run(spec: dict, rows_recorded: list, n_grid: int, margin: float, n_polish: int) -> list[dict]:
    config = prj.lm1d._config_from_embedded(spec["config"])
    config.offline_cache_dir = str(prj.OFFLINE_CACHE_DIR)
    config.use_offline_cache = True
    config.refresh_offline_cache = False
    artifact, info = prj.load_or_build_offline_artifact(config)
    if not info["cache_hit"]:
        raise RuntimeError(f"decoder not cached: {info['cache_path']}")
    problem = create_problem(artifact["problem_name"])
    ae = prj._instantiate_autoencoder(artifact)
    sc = prj._instantiate_scaler(artifact)

    def dec(z):
        return sc.inverse_transform_tensor(ae.decode(z))

    def enc(u):
        return ae.encode(sc.transform_tensor(torch.as_tensor(u)))

    train = torch.as_tensor(np.asarray(artifact["train_snapshots"], dtype=np.float64))
    with torch.no_grad():
        lat = enc(train)
    lo, hi = lat.min(0).values, lat.max(0).values
    rng = torch.clamp(hi - lo, min=1e-6)
    lo, hi = lo - margin * rng, hi + margin * rng
    axes = [torch.linspace(float(lo[i]), float(hi[i]), n_grid) for i in range(lat.shape[1])]
    Z = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, lat.shape[1])
    with torch.no_grad():
        U = torch.cat([dec(Z[i:i + 20000]) for i in range(0, len(Z), 20000)])

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    _, test_parameters = prj._resolve_test_parameters(problem, config)

    out = []
    for idx, mu in enumerate(np.asarray(test_parameters, dtype=np.float64)):
        rec = rows_recorded[idx]
        assert rec["sample"] == idx
        u = problem.solve_full(mu).solution.detach().to(torch.float64)
        nrm = torch.linalg.norm(u)
        err = torch.linalg.norm(U - u, dim=1) / nrm
        order = torch.argsort(err)[:n_polish]
        k = int(order[0])
        ij = np.unravel_index(k, (n_grid,) * lat.shape[1])
        on_edge = any(c in (0, n_grid - 1) for c in ij)
        with torch.no_grad():
            starts = [Z[j] for j in order] + [enc(u)]
        best, zbest = float(err[k]), Z[k]
        for z0 in starts:
            z = z0.clone().requires_grad_(True)
            opt = torch.optim.LBFGS([z], max_iter=80, tolerance_grad=1e-12, tolerance_change=1e-14,
                                    line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                loss = torch.sum((dec(z) - u) ** 2)
                loss.backward()
                return loss

            opt.step(closure)
            with torch.no_grad():
                e = float(torch.linalg.norm(dec(z) - u) / nrm)
            if e < best:
                best, zbest = e, z.detach()
        outside = bool(((zbest < lo) | (zbest > hi)).any())
        out.append({
            "family": spec["problem_name"], "regime": spec["test_regime"], "seed": spec["seed"], "sample": idx,
            "e_rep": rec["e_rep"], "e_grid": float(err[k]), "e_glob": best,
            "grid_min_on_box_edge": on_edge, "minimizer_outside_box": outside,
            "mg": rec["arms"]["mg"]["rel_err"],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=401)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--polish", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    shards = sorted(SHARDS.glob("*.json"))[:: args.stride]
    if args.limit:
        shards = shards[: args.limit]
    rows = []
    for sp in shards:
        d = json.loads(sp.read_text())
        rows += run(d["spec"], d["rows"], args.grid, args.margin, args.polish)
        print(sp.name, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
