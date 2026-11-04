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

from mgp.benchmark import _instantiate_autoencoder, _instantiate_scaler, _select_warm_start_state
from mgp.manifold_rom import ManifoldGalerkinROM
from mgp.problems import create_problem
from run_variant import build_rom, relative_l2

ARCHIVED_NAME = {"newton:25:on": "newton_m_lspg", "mg:25": "mg"}
ARCHIVED_KEYS = {"mg": "mgp_error_l2", "newton_m_lspg": "mgp_lspg_error_l2", "pod_g": "pod_error_l2", "pod_lspg": "lspg_error_l2"}


def read_archive(tar_path: Path):
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()

        def read(suffix: str) -> bytes:
            member = next(m for m in members if m.name.endswith(suffix))
            return tar.extractfile(member).read()

        result = json.loads(read("result.json"))
        artifact = torch.load(io.BytesIO(read("offline_artifact.pt")), map_location="cpu", weights_only=False)
    return result, artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--strategy", default="nearest_train")
    parser.add_argument("--variants", default="newton:25:on,gn:25:on,newton:25:off,newton:200:on,gn:200:off")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    result, artifact = read_archive(args.archive)
    block = result["strategies"][args.strategy]
    config = block["config"]
    if config["warm_start_strategy"] != args.strategy:
        raise ValueError("strategy mismatch between archive and request")
    problem = create_problem(artifact["problem_name"])
    autoencoder = _instantiate_autoencoder(artifact)
    scaler = _instantiate_scaler(artifact)
    train_parameters = np.asarray(artifact["train_parameters"], dtype=np.float64)
    train_snapshots = np.asarray(artifact["train_snapshots"], dtype=np.float64)
    base_kwargs = {
        "solver_tol": config["nonlinear_solver_tol"],
        "solver_max_iter": config["nonlinear_solver_max_iter"],
        "solver_line_search_steps": config["nonlinear_solver_line_search_steps"],
        "solver_min_step_scale": config["nonlinear_solver_min_step_scale"],
    }
    mg = ManifoldGalerkinROM(autoencoder, scaler, problem, **base_kwargs)
    specs = args.variants.split(",")
    roms = {spec: build_rom(spec, autoencoder, scaler, problem, base_kwargs) for spec in specs}

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    cases_out = []
    for sample, case in enumerate(block["cases"][: args.cases]):
        mu = np.asarray(case["mu"], dtype=np.float64)
        t0 = time.perf_counter()
        full = problem.solve_full(mu)
        reference = full.solution.detach().cpu().numpy()
        if not bool(full.converged) or not np.all(np.isfinite(reference)):
            cases_out.append({"sample": sample, "mu": mu.tolist(), "reference": {"status": "reference_failure"}, "variants": {}})
            continue
        ref_qoi = float(problem.quantity_of_interest(reference))
        warm = _select_warm_start_state(args.strategy, mu, train_parameters, train_snapshots)
        latent0 = mg.encode(warm)
        row = {
            "sample": sample,
            "mu": mu.tolist(),
            "reference": {"status": "returned", "time_sec": time.perf_counter() - t0, "qoi": ref_qoi, "archived_qoi": case.get("full_qoi"), "qoi_abs_difference": abs(ref_qoi - case["full_qoi"]) if case.get("full_qoi") is not None else None},
            "archived": {name: case[key] for name, key in ARCHIVED_KEYS.items()},
            "variants": {},
        }
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
                    entry.update(relative_state_error=relative_l2(reference, state), ambient_residual_norm=float(torch.linalg.norm(residual).item()), coordinates=coords.tolist())
            except Exception as exc:
                entry = {"status": "solver_exception", "exception_type": type(exc).__name__, "exception_message": str(exc)[:500], "time_sec": float(time.perf_counter() - started)}
            row["variants"][spec] = entry
        cases_out.append(row)
        print(args.archive.name, "sample", sample, "ref dQoI %.2e" % (row["reference"]["qoi_abs_difference"] or 0.0), {s: round(v.get("relative_state_error", float("nan")), 4) for s, v in row["variants"].items()}, flush=True)

    gate = {}
    for spec, archived in ARCHIVED_NAME.items():
        if spec in roms:
            diffs = [abs(r["variants"][spec].get("relative_state_error", float("nan")) - r["archived"][archived]) for r in cases_out if r["variants"]]
            gate[spec] = {"archived_arm": archived, "max_abs_error_difference": float(np.nanmax(diffs)) if diffs else None, "n_cases": len(diffs)}
    payload = {
        "archive": args.archive.name,
        "case_id": result["case_id"],
        "problem": artifact["problem_name"],
        "regime": result["regime"],
        "seed": result["seed"],
        "strategy": args.strategy,
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
