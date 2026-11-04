#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
from dataclasses import asdict
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

from mgp.autoencoder import SnapshotAutoencoder, resolve_activation, train_autoencoder
from mgp.benchmark import (
    BenchmarkConfig,
    _effective_projected_weight,
    _instantiate_scaler,
    _physics_penalty_terms,
    np_to_torch,
)
from mgp.manifold_rom import ManifoldGalerkinROM
from mgp.problems import create_problem
from run_variant import build_rom, relative_l2


def read_archive(tar_path: Path):
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()

        def read(suffix: str) -> bytes:
            member = next(m for m in members if m.name.endswith(suffix))
            return tar.extractfile(member).read()

        result = json.loads(read("result.json"))
        artifact = torch.load(io.BytesIO(read("offline_artifact.pt")), map_location="cpu", weights_only=False)
        with np.load(io.BytesIO(read("offline_matrices.npz"))) as npz:
            matrices = {k: npz[k] for k in npz.files}
    return result, artifact, matrices


def polish_error(decode, encode, u_ref: np.ndarray) -> tuple[float, float]:
    z0 = encode(u_ref)
    with torch.no_grad():
        e_ae = relative_l2(u_ref, decode(z0).numpy())
    z = z0.clone().requires_grad_(True)
    target = torch.tensor(u_ref, dtype=torch.float64)
    opt = torch.optim.LBFGS([z], max_iter=60, tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.sum((decode(z) - target) ** 2)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        e_pol = min(e_ae, relative_l2(u_ref, decode(z).numpy()))
    return e_ae, e_pol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extra-archive", type=Path, nargs="*", default=[], help="further archives (e.g. the OOD partner) whose cases are evaluated with the same retrained decoder")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--init-seed", type=int, default=None, help="network-initialization seed (default: the data seed)")
    parser.add_argument("--variants", default="mg:25,newton:25:on,gn:25:on,lm:25")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    result, artifact, mats = read_archive(args.archive)
    config = BenchmarkConfig(**artifact["config"])
    config.autoencoder_epochs = args.epochs
    init_seed = config.seed if args.init_seed is None else args.init_seed
    problem = create_problem(config.problem_name)
    train_parameters = np.asarray(artifact["train_parameters"], dtype=np.float64)
    train_snapshots = np.asarray(artifact["train_snapshots"], dtype=np.float64)
    scaler = _instantiate_scaler(artifact)
    normalized = scaler.transform(train_snapshots)

    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    autoencoder = SnapshotAutoencoder(
        input_dim=problem.interior_dim,
        latent_dim=config.latent_dim,
        architecture_name=config.architecture_name,
        hidden_dims=tuple(config.hidden_dims),
        decoder_hidden_dims=tuple(config.decoder_hidden_dims),
        activation=resolve_activation(config.activation_name),
    )
    started = time.perf_counter()
    history = train_autoencoder(
        autoencoder,
        normalized,
        epochs=config.autoencoder_epochs,
        learning_rate=config.autoencoder_lr,
        smoothness_weight=config.smoothness_weight,
        projected_residual_penalty_weight=_effective_projected_weight(config),
        ambient_residual_penalty_weight=config.ambient_residual_penalty_weight,
        residual_penalty_schedule=config.residual_penalty_schedule,
        training_objective_mode=config.training_objective_mode,
        online_residual_sample_count=config.online_residual_sample_count,
        online_residual_sample_scale=config.online_residual_sample_scale,
        online_residual_sample_seed_offset=config.online_residual_sample_seed_offset,
        online_residual_sample_source=config.online_residual_sample_source,
        parameters=train_parameters,
        physics_penalty_fn=lambda reconstructed, latent_batch, parameter_batch: _physics_penalty_terms(
            reconstructed, latent_batch, parameter_batch, problem, scaler, autoencoder, config.residual_penalty_metric
        ),
        seed=config.seed,
    )
    train_time = time.perf_counter() - started
    autoencoder.eval()
    reconstructed = scaler.inverse_transform(autoencoder(np_to_torch(normalized)).detach().cpu().numpy())
    train_recon = relative_l2(train_snapshots.ravel(), reconstructed.ravel())
    archived_sd = artifact["autoencoder_state_dict"]
    max_dw = max(float((autoencoder.state_dict()[k] - archived_sd[k]).abs().max()) for k in archived_sd)

    base_kwargs = {
        "solver_tol": config.nonlinear_solver_tol,
        "solver_max_iter": config.nonlinear_solver_max_iter,
        "solver_line_search_steps": config.nonlinear_solver_line_search_steps,
        "solver_min_step_scale": config.nonlinear_solver_min_step_scale,
    }
    mg = ManifoldGalerkinROM(autoencoder, scaler, problem, **base_kwargs)
    latent0 = mg.encode(train_snapshots.mean(axis=0))
    specs = args.variants.split(",")
    roms = {spec: build_rom(spec, autoencoder, scaler, problem, base_kwargs) for spec in specs}
    evaluation_sets = [(config.test_regime, result, mats)]
    for extra in args.extra_archive:
        extra_result, extra_artifact, extra_mats = read_archive(extra)
        if extra_artifact["config"]["seed"] != config.seed or extra_artifact["problem_name"] != config.problem_name:
            raise ValueError("extra archive must share the data seed and family")
        evaluation_sets.append((extra_result["config"]["test_regime"], extra_result, extra_mats))

    def decode(z):
        return scaler.inverse_transform_tensor(autoencoder.decode(z))

    def encode(u):
        with torch.no_grad():
            return autoencoder.encode(scaler.transform_tensor(torch.tensor(u, dtype=torch.float64)))

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    cases_out = []
    for regime, eval_result, eval_mats in evaluation_sets:
      V, u_c = eval_mats["pod_basis"], eval_mats["pod_mean"]
      for case in eval_result["cases"][: args.cases]:
        mu = np.asarray(case["mu"], dtype=np.float64)
        reference = np.asarray(case["full_order"]["state"], dtype=np.float64)
        d = reference - u_c
        e_ae, e_pol = polish_error(decode, encode, reference)
        row = {
            "regime": regime,
            "sample": case["sample"],
            "mu": mu.tolist(),
            "archived": {k: case["arms"][k]["relative_state_error"] for k in case["arms"]},
            "e_proj": float(np.linalg.norm(d - V @ (V.T @ d)) / np.linalg.norm(reference)),
            "e_ae": e_ae,
            "e_pol": e_pol,
            "variants": {},
        }
        for spec, rom in roms.items():
            t0 = time.perf_counter()
            try:
                res = rom.solve(mu, initial=latent0)
                coords = res.solution.detach().cpu().numpy()
                state = rom.decode(coords)
                finite = bool(np.all(np.isfinite(coords)) and np.all(np.isfinite(state)))
                entry = {
                    "status": "returned" if finite else "nonfinite_return",
                    "converged": bool(res.converged),
                    "iterations": int(res.iterations),
                    "solver_residual_norm": float(res.residual_norm),
                    "time_sec": float(time.perf_counter() - t0),
                }
                if finite:
                    residual = problem.residual_torch(torch.as_tensor(state, dtype=torch.float64), mu)
                    entry.update(relative_state_error=relative_l2(reference, state), ambient_residual_norm=float(torch.linalg.norm(residual).item()))
            except Exception as exc:
                entry = {"status": "solver_exception", "exception_type": type(exc).__name__, "exception_message": str(exc)[:500]}
            row["variants"][spec] = entry
        cases_out.append(row)
        print(args.archive.name, regime, "sample", case["sample"], "e_pol %.4f e_proj %.4f" % (e_pol, row["e_proj"]), {s: round(v.get("relative_state_error", float("nan")), 4) for s, v in row["variants"].items()}, flush=True)

    tag = f"ep{args.epochs}_init{init_seed}"
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(config),
            "init_seed": init_seed,
            "autoencoder_state_dict": autoencoder.state_dict(),
            "autoencoder_history": history,
            "autoencoder_reconstruction_error": train_recon,
            "scaler": artifact["scaler"],
        },
        args.out / f"{result['case_id']}.{tag}.decoder.pt",
    )
    loss = np.asarray(history["loss"], dtype=float)
    k = max(2, len(loss) // 10)
    payload = {
        "archive": args.archive.name,
        "case_id": result["case_id"],
        "problem": config.problem_name,
        "regime": config.test_regime,
        "seed": config.seed,
        "init_seed": init_seed,
        "epochs": args.epochs,
        "training": {
            "time_sec": train_time,
            "final_loss": float(loss[-1]),
            "min_loss": float(loss.min()),
            "windowed_decrease": float((loss[-2 * k:-k].mean() - loss[-k:].mean()) / loss[-2 * k:-k].mean()),
            "train_reconstruction_error": train_recon,
            "archived_train_reconstruction_error": artifact["autoencoder_reconstruction_error"],
            "max_abs_weight_difference_vs_archived": max_dw,
        },
        "variants": specs,
        "runtime": {"torch": torch.__version__, "numpy": np.__version__, "platform": sys.platform},
        "cases": cases_out,
    }
    out_path = args.out / f"{result['case_id']}.{tag}.retrain.json"
    out_path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False))
    print("training:", json.dumps(payload["training"]), "\nwrote", out_path)


if __name__ == "__main__":
    main()
