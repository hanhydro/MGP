#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import os

for _cand in (Path(os.environ["MGP_SRC"]) if os.environ.get("MGP_SRC") else None, Path(__file__).resolve().parent / "src", Path(__file__).resolve().parents[1] / "src"):
    if _cand is not None and _cand.exists():
        sys.path.insert(0, str(_cand))
        break
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mgp.benchmark import _instantiate_autoencoder, _instantiate_scaler
from mgp.manifold_rom import ManifoldGalerkinROM, ManifoldLSPGLMROM
from mgp.problems import create_problem
from solvers import ManifoldLSPGGaussNewtonROM, ManifoldLSPGNewtonVariantROM

ARCHIVED_NAME = {"newton:25:on": "newton_m_lspg", "mg:25": "mg", "lm:25": "lm_m_lspg"}


def read_archive(tar_path: Path):
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()

        def read(suffix: str) -> bytes:
            member = next(m for m in members if m.name.endswith(suffix))
            return tar.extractfile(member).read()

        result = json.loads(read("result.json"))
        artifact = torch.load(io.BytesIO(read("offline_artifact.pt")), map_location="cpu", weights_only=False)
    return result, artifact


def build_rom(spec: str, autoencoder, scaler, problem, base_kwargs: dict):
    parts = spec.split(":")
    kind, cap = parts[0], int(parts[1])
    kwargs = dict(base_kwargs, solver_max_iter=cap)
    if kind == "mg":
        return ManifoldGalerkinROM(autoencoder, scaler, problem, **kwargs)
    if kind == "lm":
        return ManifoldLSPGLMROM(autoencoder, scaler, problem, **kwargs)
    fallback = parts[2] == "on"
    if kind == "newton":
        return ManifoldLSPGNewtonVariantROM(autoencoder, scaler, problem, fallback=fallback, **kwargs)
    if kind == "gn":
        return ManifoldLSPGGaussNewtonROM(autoencoder, scaler, problem, fallback=fallback, **kwargs)
    raise ValueError(spec)


def relative_l2(reference: np.ndarray, state: np.ndarray) -> float:
    return float(np.linalg.norm(reference - state) / max(np.linalg.norm(reference), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--variants", default="newton:25:on,gn:25:on,newton:25:off,newton:200:on,gn:200:off,mg:200,lm:200")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=None, help="limit the number of held-out cases (smoke tests only)")
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    result, artifact = read_archive(args.archive)
    config = result["config"]
    if config["warm_start_strategy"] != "mean_train":
        raise ValueError("this runner implements the training-mean start of the first prospective block")
    problem = create_problem(artifact["problem_name"])
    autoencoder = _instantiate_autoencoder(artifact)
    scaler = _instantiate_scaler(artifact)
    train_snapshots = np.asarray(artifact["train_snapshots"], dtype=np.float64)
    base_kwargs = {
        "solver_tol": config["nonlinear_solver_tol"],
        "solver_max_iter": config["nonlinear_solver_max_iter"],
        "solver_line_search_steps": config["nonlinear_solver_line_search_steps"],
        "solver_min_step_scale": config["nonlinear_solver_min_step_scale"],
    }
    mg = ManifoldGalerkinROM(autoencoder, scaler, problem, **base_kwargs)
    warm = train_snapshots.mean(axis=0)
    latent0 = mg.encode(warm)

    specs = args.variants.split(",")
    roms = {spec: build_rom(spec, autoencoder, scaler, problem, base_kwargs) for spec in specs}

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    cases_out = []
    for case in result["cases"][: args.cases]:
        mu = np.asarray(case["mu"], dtype=np.float64)
        reference = np.asarray(case["full_order"]["state"], dtype=np.float64)
        row = {"sample": case["sample"], "mu": mu.tolist(), "archived": {k: case["arms"][k]["relative_state_error"] for k in case["arms"]}, "variants": {}}
        for spec, rom in roms.items():
            started = time.perf_counter()
            try:
                res = rom.solve(mu, initial=latent0)
                elapsed = time.perf_counter() - started
                coords = res.solution.detach().cpu().numpy()
                state = rom.decode(coords)
                finite = bool(np.all(np.isfinite(coords)) and np.all(np.isfinite(state)))
                trace = getattr(res, "trace", [])
                entry = {
                    "status": "returned" if finite else "nonfinite_return",
                    "converged": bool(res.converged),
                    "stagnated": bool(getattr(res, "stagnated", False)),
                    "iterations": int(res.iterations),
                    "solver_residual_norm": float(res.residual_norm),
                    "time_sec": float(elapsed),
                    "n_fallback": int(sum(1 for t in trace if t["fallback"])),
                    "n_nondescent": int(sum(1 for t in trace if t["directional_derivative"] >= 0.0)),
                    "n_trace": len(trace),
                }
                if finite:
                    residual = problem.residual_torch(torch.as_tensor(state, dtype=torch.float64), mu)
                    entry.update(
                        relative_state_error=relative_l2(reference, state),
                        ambient_residual_norm=float(torch.linalg.norm(residual).item()),
                        coordinates=coords.tolist(),
                    )
            except Exception as exc:
                entry = {"status": "solver_exception", "exception_type": type(exc).__name__, "exception_message": str(exc)[:500], "time_sec": float(time.perf_counter() - started)}
            row["variants"][spec] = entry
        cases_out.append(row)
        print(args.archive.name, "sample", case["sample"], {s: round(v.get("relative_state_error", float("nan")), 4) for s, v in row["variants"].items()}, flush=True)

    gate = {}
    for spec, archived in ARCHIVED_NAME.items():
        if spec in roms:
            diffs = [abs(r["variants"][spec].get("relative_state_error", float("nan")) - r["archived"][archived]) for r in cases_out]
            gate[spec] = {"archived_arm": archived, "max_abs_error_difference": float(np.nanmax(diffs)), "n_cases": len(diffs)}
    payload = {
        "archive": args.archive.name,
        "case_id": result["case_id"],
        "problem": artifact["problem_name"],
        "regime": config["test_regime"],
        "seed": config["seed"],
        "variants": specs,
        "reproduction_gate": gate,
        "runtime": {"torch": torch.__version__, "numpy": np.__version__, "platform": sys.platform},
        "cases": cases_out,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / (result["case_id"] + ".variants.json")
    out_path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False))
    print("gate:", json.dumps(gate), "\nwrote", out_path)


if __name__ == "__main__":
    main()
