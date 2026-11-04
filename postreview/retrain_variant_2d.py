#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import io
import inspect
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _cand in (Path(os.environ["MGP_SRC"]) if os.environ.get("MGP_SRC") else None, Path(__file__).resolve().parent / "src"):
    if _cand is not None and _cand.exists():
        sys.path.insert(0, str(_cand))
        break
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mgp.autoencoder import SnapshotAutoencoder, resolve_activation
from mgp.benchmark import (
    BenchmarkConfig,
    _effective_projected_weight,
    _instantiate_scaler,
    _physics_penalty_terms,
    _select_warm_start_state,
    np_to_torch,
)
from mgp.manifold_rom import ManifoldGalerkinROM
from mgp.problems import create_problem
from retrain_variant import polish_error
from run_variant import build_rom, relative_l2
from train_variants import train_autoencoder_variant


def read_archive(tar_path: Path):
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()

        def read(suffix: str) -> bytes:
            member = next(m for m in members if m.name.endswith(suffix))
            return tar.extractfile(member).read()

        result = json.loads(read("result.json"))
        artifact = torch.load(io.BytesIO(read("offline_artifact.pt")), map_location="cpu", weights_only=False)
    return result, artifact


def new_model(config, problem):
    return SnapshotAutoencoder(
        input_dim=problem.interior_dim,
        latent_dim=config.latent_dim,
        architecture_name=config.architecture_name,
        hidden_dims=tuple(config.hidden_dims),
        decoder_hidden_dims=tuple(config.decoder_hidden_dims),
        activation=resolve_activation(config.activation_name),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True, help="in-domain second-block archive")
    parser.add_argument("--extra-archive", type=Path, nargs="*", default=[], help="OOD partner archive(s) of the same seed")
    parser.add_argument("--schedule", choices=["const", "cosine"], required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--variants", default="mg:25,newton:25:on")
    parser.add_argument("--strategy", default="nearest_train")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    result, artifact = read_archive(args.archive)
    config = BenchmarkConfig(**artifact["config"])
    problem = create_problem(config.problem_name)
    train_parameters = np.asarray(artifact["train_parameters"], dtype=np.float64)
    train_snapshots = np.asarray(artifact["train_snapshots"], dtype=np.float64)
    scaler = _instantiate_scaler(artifact)
    normalized = scaler.transform(train_snapshots)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = new_model(config, problem)
    projected_weight = _effective_projected_weight(config)
    fast = {"compute_projected_graph": projected_weight != 0.0} if "compute_projected_graph" in inspect.signature(_physics_penalty_terms).parameters else {}
    started = time.perf_counter()
    history, best = train_autoencoder_variant(
        model,
        normalized,
        epochs=args.epochs,
        learning_rate=config.autoencoder_lr,
        projected_residual_penalty_weight=projected_weight,
        ambient_residual_penalty_weight=config.ambient_residual_penalty_weight,
        residual_penalty_schedule=config.residual_penalty_schedule,
        parameters=train_parameters,
        physics_penalty_fn=lambda r, z, p: _physics_penalty_terms(r, z, p, problem, scaler, model, config.residual_penalty_metric, **fast),
        seed=config.seed,
        lr_schedule=None if args.schedule == "const" else "cosine",
    )
    train_time = time.perf_counter() - started
    model.eval()
    best_model = new_model(config, problem)
    best_model.load_state_dict(best["state_dict"])
    best_model.eval()
    decoders = {"final": model, "best": best_model}

    archived_sd = artifact["autoencoder_state_dict"]
    training = {"time_sec": train_time, "final_loss": history["loss"][-1], "best_loss": best["loss"], "best_epoch": best["epoch"],
                "final_over_best_loss": history["loss"][-1] / best["loss"]}
    for name, m in decoders.items():
        rec = scaler.inverse_transform(m(np_to_torch(normalized)).detach().cpu().numpy())
        training[f"{name}_train_reconstruction_error"] = relative_l2(train_snapshots.ravel(), rec.ravel())
        training[f"{name}_max_abs_weight_difference_vs_archived"] = max(float((m.state_dict()[k] - archived_sd[k]).abs().max()) for k in archived_sd)
    training["archived_train_reconstruction_error"] = artifact["autoencoder_reconstruction_error"]

    u_c = train_snapshots.mean(axis=0)
    _, _, vt = np.linalg.svd(train_snapshots - u_c, full_matrices=False)
    V = vt[: config.pod_dim].T

    base_kwargs = {
        "solver_tol": config.nonlinear_solver_tol,
        "solver_max_iter": config.nonlinear_solver_max_iter,
        "solver_line_search_steps": config.nonlinear_solver_line_search_steps,
        "solver_min_step_scale": config.nonlinear_solver_min_step_scale,
    }
    specs = args.variants.split(",")
    evaluation_sets = [(result["regime"], result)]
    for extra in args.extra_archive:
        extra_result, extra_artifact = read_archive(extra)
        if extra_result["seed"] != result["seed"] or not np.array_equal(np.asarray(extra_artifact["train_parameters"]), train_parameters):
            raise ValueError("extra archive must share the seed and the training data")
        evaluation_sets.append((extra_result["regime"], extra_result))

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    cases_out = []
    for regime, eval_result in evaluation_sets:
        block = eval_result["strategies"][args.strategy]
        for sample, case in enumerate(block["cases"][: args.cases]):
            mu = np.asarray(case["mu"], dtype=np.float64)
            full = problem.solve_full(mu)
            reference = full.solution.detach().cpu().numpy()
            d = reference - u_c
            row = {
                "regime": regime,
                "sample": sample,
                "mu": mu.tolist(),
                "reference_qoi_abs_difference": abs(float(problem.quantity_of_interest(reference)) - case["full_qoi"]),
                "e_proj": float(np.linalg.norm(d - V @ (V.T @ d)) / np.linalg.norm(reference)),
                "archived": {"mg": case["mgp_error_l2"], "newton_m_lspg": case["mgp_lspg_error_l2"], "pod_g": case["pod_error_l2"], "pod_lspg": case["lspg_error_l2"]},
                "decoders": {},
            }
            warm = _select_warm_start_state(args.strategy, mu, train_parameters, train_snapshots)
            for name, m in decoders.items():
                def decode(z, m=m):
                    return scaler.inverse_transform_tensor(m.decode(z))

                def encode(u, m=m):
                    with torch.no_grad():
                        return m.encode(scaler.transform_tensor(torch.tensor(u, dtype=torch.float64)))

                e_ae, e_pol = polish_error(decode, encode, reference)
                mg = ManifoldGalerkinROM(m, scaler, problem, **base_kwargs)
                latent0 = mg.encode(warm)
                entry = {"e_ae": e_ae, "e_pol": e_pol, "arms": {}}
                for spec in specs:
                    rom = build_rom(spec, m, scaler, problem, base_kwargs)
                    t0 = time.perf_counter()
                    try:
                        res = rom.solve(mu, initial=latent0)
                        coords = res.solution.detach().cpu().numpy()
                        state = rom.decode(coords)
                        finite = bool(np.all(np.isfinite(coords)) and np.all(np.isfinite(state)))
                        arm = {"status": "returned" if finite else "nonfinite_return", "converged": bool(res.converged),
                               "iterations": int(res.iterations), "time_sec": float(time.perf_counter() - t0)}
                        if finite:
                            arm["relative_state_error"] = relative_l2(reference, state)
                    except Exception as exc:
                        arm = {"status": "solver_exception", "exception_type": type(exc).__name__, "exception_message": str(exc)[:500]}
                    entry["arms"][spec] = arm
                row["decoders"][name] = entry
            cases_out.append(row)
            print(args.archive.name, regime, "sample", sample, {n: {s: round(a.get("relative_state_error", float("nan")), 4) for s, a in e["arms"].items()} for n, e in row["decoders"].items()}, flush=True)

    tag = f"{args.schedule}{args.epochs}"
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"config": artifact["config"], "schedule": args.schedule, "epochs": args.epochs, "history": history,
                "best_epoch": best["epoch"], "final_state_dict": model.state_dict(), "best_state_dict": best["state_dict"], "scaler": artifact["scaler"]},
               args.out / f"{result['case_id']}.{tag}.decoder.pt")
    payload = {"archive": args.archive.name, "case_id": result["case_id"], "problem": config.problem_name, "seed": config.seed,
               "schedule": args.schedule, "epochs": args.epochs, "strategy": args.strategy, "training": training, "variants": specs,
               "runtime": {"torch": torch.__version__, "numpy": np.__version__, "platform": sys.platform}, "cases": cases_out}
    out_path = args.out / f"{result['case_id']}.{tag}.retrain2d.json"
    out_path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False))
    print("training:", json.dumps(training), "\nwrote", out_path)


if __name__ == "__main__":
    main()
