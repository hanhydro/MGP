from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .autoencoder import SnapshotAutoencoder, SnapshotScaler, resolve_activation, train_autoencoder
from .manifold_rom import ManifoldGalerkinROM, ManifoldLSPGROM
from .pod_rom import PODGalerkinROM, PODLSPGROM
from .problems import NonlinearDiffusionProblem, create_problem


@dataclass
class BenchmarkConfig:
    train_size: int = 30
    test_size: int = 12
    test_full_size: int | None = None
    test_case_start: int = 0
    test_case_count: int | None = None
    problem_name: str = "nonlinear_diffusion"
    test_regime: str = "in_domain"
    latent_dim: int = 4
    pod_dim: int = 4
    architecture_name: str = "mlp"
    hidden_dims: tuple[int, ...] = (48, 48)
    decoder_hidden_dims: tuple[int, ...] = ()
    activation_name: str = "tanh"
    autoencoder_epochs: int = 800
    autoencoder_lr: float = 1e-3
    smoothness_weight: float = 0.0
    residual_penalty_weight: float = 0.02
    projected_residual_penalty_weight: float | None = None
    ambient_residual_penalty_weight: float = 0.005
    residual_penalty_schedule: str = "constant"
    residual_penalty_metric: str = "l2_norm"
    training_objective_mode: str = "standard"
    online_residual_sample_count: int = 0
    online_residual_sample_scale: float = 0.05
    online_residual_sample_seed_offset: int = 10007
    online_residual_sample_source: str = "encoded_training_latent_perturbation"
    warm_start_strategy: str = "nearest_train"
    nonlinear_solver_tol: float = 1e-8
    nonlinear_solver_max_iter: int = 25
    nonlinear_solver_line_search_steps: int = 8
    nonlinear_solver_min_step_scale: float = 1e-3
    seed: int = 0
    deterministic_publication_mode: bool = False
    publication_num_threads: int = 1
    offline_cache_dir: str = "artifacts/cache"
    use_offline_cache: bool = False
    refresh_offline_cache: bool = False


def np_to_torch(values: np.ndarray):
    return torch.tensor(values, dtype=torch.float64)


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    numerator = np.linalg.norm(reference - candidate)
    denominator = np.linalg.norm(reference)
    return float(numerator / max(denominator, 1e-12))


def _solve_snapshots(problem: NonlinearDiffusionProblem, parameters: np.ndarray) -> np.ndarray:
    snapshots = []
    previous = None
    for mu in parameters:
        result = problem.solve_full(mu, initial=previous)
        snapshots.append(result.solution.detach().cpu().numpy())
        previous = snapshots[-1]
    return np.asarray(snapshots)


def _sample_test_parameters(
    problem: NonlinearDiffusionProblem, test_size: int, *, regime: str, seed: int
) -> np.ndarray:
    if regime == "in_domain":
        return problem.sample_parameters(test_size, seed=seed)
    if regime == "ood":
        return problem.sample_parameters_ood(test_size, seed=seed)
    raise ValueError(f"unknown test_regime: {regime}")


def _resolve_test_parameters(
    problem: NonlinearDiffusionProblem,
    config: BenchmarkConfig,
) -> tuple[np.ndarray, np.ndarray]:
    full_test_size = config.test_full_size if config.test_full_size is not None else config.test_size
    all_test_parameters = _sample_test_parameters(
        problem,
        full_test_size,
        regime=config.test_regime,
        seed=config.seed + 1,
    )
    if config.test_case_count is None:
        test_parameters = all_test_parameters[config.test_case_start :]
    else:
        test_parameters = all_test_parameters[
            config.test_case_start : config.test_case_start + config.test_case_count
        ]
    if len(test_parameters) == 0:
        raise ValueError("benchmark test subset is empty")
    return all_test_parameters, test_parameters


def _nearest_training_snapshot(
    mu: np.ndarray, train_parameters: np.ndarray, train_snapshots: np.ndarray
) -> np.ndarray:
    distances = np.linalg.norm(train_parameters - mu[None, :], axis=1)
    return train_snapshots[int(np.argmin(distances))]


def _select_warm_start_state(
    strategy: str,
    mu: np.ndarray,
    train_parameters: np.ndarray,
    train_snapshots: np.ndarray,
) -> np.ndarray:
    if strategy == "nearest_train":
        return _nearest_training_snapshot(mu, train_parameters, train_snapshots)
    if strategy == "mean_train":
        return np.mean(train_snapshots, axis=0)
    if strategy == "zero":
        return np.zeros(train_snapshots.shape[1], dtype=np.float64)
    raise ValueError(f"unknown warm_start_strategy: {strategy}")


def _metric_norm(values: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "mean_square":
        return torch.mean(values**2)
    if metric == "l2_norm":
        return torch.linalg.vector_norm(values)
    if metric == "linf":
        return torch.max(torch.abs(values))
    raise ValueError(f"unknown residual_penalty_metric: {metric}")


def _decode_physical(autoencoder: SnapshotAutoencoder, scaler: SnapshotScaler, latent: torch.Tensor) -> torch.Tensor:
    normalized_state = autoencoder.decode(latent)
    return scaler.inverse_transform_tensor(normalized_state)


def _physics_penalty_terms(
    reconstructed_normalized: torch.Tensor,
    latent_batch: torch.Tensor,
    parameter_batch: torch.Tensor,
    problem: NonlinearDiffusionProblem,
    scaler: SnapshotScaler,
    autoencoder: SnapshotAutoencoder,
    metric: str,
):
    projected_penalties = []
    ambient_penalties = []
    for reconstructed_sample, latent_sample, mu in zip(reconstructed_normalized, latent_batch, parameter_batch):
        physical_state = scaler.inverse_transform_tensor(reconstructed_sample)
        ambient_residual = problem.residual_torch(physical_state, mu)

        def decode_physical(z: torch.Tensor) -> torch.Tensor:
            return _decode_physical(autoencoder, scaler, z)

        tangent = torch.autograd.functional.jacobian(decode_physical, latent_sample, create_graph=True)
        projected_residual = tangent.T @ ambient_residual
        projected_penalties.append(_metric_norm(projected_residual, metric))
        ambient_penalties.append(_metric_norm(ambient_residual, metric))
    return {
        "projected": torch.stack(projected_penalties).mean(),
        "ambient": torch.stack(ambient_penalties).mean(),
    }


def _decoder_jacobian_diagnostics(autoencoder: SnapshotAutoencoder, latent_state: np.ndarray) -> tuple[float, float]:
    latent_tensor = torch.tensor(latent_state, dtype=torch.float64)
    jacobian = autoencoder.decoder_jacobian(latent_tensor).detach().cpu().numpy()
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    jacobian_fro_norm = float(np.linalg.norm(jacobian))
    jacobian_condition = float(singular_values[0] / max(singular_values[-1], 1e-12))
    return jacobian_fro_norm, jacobian_condition


def _projected_residual_norm(mgp_rom: ManifoldGalerkinROM, latent_state: np.ndarray, mu) -> float:
    latent_tensor = torch.tensor(latent_state, dtype=torch.float64)
    reduced_residual = mgp_rom.reduced_residual_torch(latent_tensor, mu)
    return float(torch.linalg.vector_norm(reduced_residual).item())


def _reconstruction_case_metrics(
    autoencoder: SnapshotAutoencoder,
    scaler: SnapshotScaler,
    problem: NonlinearDiffusionProblem,
    snapshots: np.ndarray,
    parameters: np.ndarray,
) -> list[dict]:
    metrics = []
    for snapshot, mu in zip(snapshots, parameters):
        normalized_snapshot = scaler.transform_tensor(np_to_torch(snapshot))
        latent = autoencoder.encode(normalized_snapshot)
        reconstructed_normalized = autoencoder.decode(latent)
        reconstructed_physical = scaler.inverse_transform_tensor(reconstructed_normalized)
        ambient_residual = problem.residual_torch(reconstructed_physical, mu)

        def decode_physical(z: torch.Tensor) -> torch.Tensor:
            return _decode_physical(autoencoder, scaler, z)

        tangent = torch.autograd.functional.jacobian(decode_physical, latent)
        projected_residual = tangent.T @ ambient_residual
        reconstructed_state = reconstructed_physical.detach().cpu().numpy()
        metrics.append(
            {
                "mu": np.asarray(mu).tolist(),
                "reconstruction_error_l2": _relative_l2(snapshot, reconstructed_state),
                "reconstruction_residual": float(torch.linalg.vector_norm(ambient_residual).item()),
                "reconstruction_projected_residual": float(
                    torch.linalg.vector_norm(projected_residual).item()
                ),
            }
        )
    return metrics


def _publication_runtime_metadata(config: BenchmarkConfig) -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "deterministic_publication_mode": config.deterministic_publication_mode,
        "publication_num_threads_requested": config.publication_num_threads,
        "torch_num_threads": torch.get_num_threads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def _effective_projected_weight(config: BenchmarkConfig) -> float:
    if config.projected_residual_penalty_weight is not None:
        return float(config.projected_residual_penalty_weight)
    return float(config.residual_penalty_weight)


def _offline_artifact_key_payload(config: BenchmarkConfig) -> dict:
    payload = asdict(config)
    for key in (
        "pod_dim",
        "warm_start_strategy",
        "offline_cache_dir",
        "use_offline_cache",
        "refresh_offline_cache",
        "test_full_size",
        "test_case_start",
        "test_case_count",
        "nonlinear_solver_tol",
        "nonlinear_solver_max_iter",
        "nonlinear_solver_line_search_steps",
        "nonlinear_solver_min_step_scale",
    ):
        payload.pop(key, None)
    return payload


def offline_artifact_cache_key(config: BenchmarkConfig) -> str:
    payload = _offline_artifact_key_payload(config)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def offline_artifact_cache_path(config: BenchmarkConfig) -> Path:
    return Path(config.offline_cache_dir) / f"offline_artifact_{offline_artifact_cache_key(config)}.pt"


def _instantiate_scaler(artifact: dict) -> SnapshotScaler:
    scaler_state = artifact["scaler"]
    return SnapshotScaler(
        mean=torch.tensor(scaler_state["mean"], dtype=torch.float64),
        std=torch.tensor(scaler_state["std"], dtype=torch.float64),
    )


def _instantiate_autoencoder(artifact: dict) -> SnapshotAutoencoder:
    artifact_config = artifact["config"]
    problem = create_problem(artifact_config["problem_name"])
    autoencoder = SnapshotAutoencoder(
        input_dim=problem.interior_dim,
        latent_dim=artifact_config["latent_dim"],
        architecture_name=artifact_config["architecture_name"],
        hidden_dims=tuple(artifact_config["hidden_dims"]),
        decoder_hidden_dims=tuple(artifact_config["decoder_hidden_dims"]),
        activation=resolve_activation(artifact_config["activation_name"]),
    )
    autoencoder.load_state_dict(artifact["autoencoder_state_dict"])
    autoencoder.eval()
    return autoencoder


def _build_offline_artifact_data(config: BenchmarkConfig, runtime_metadata: dict) -> dict:
    problem = create_problem(config.problem_name)
    train_parameters = problem.sample_parameters(config.train_size, seed=config.seed)

    train_snapshots = _solve_snapshots(problem, train_parameters)
    scaler = SnapshotScaler.fit(train_snapshots)
    normalized_snapshots = scaler.transform(train_snapshots)

    autoencoder = SnapshotAutoencoder(
        input_dim=problem.interior_dim,
        latent_dim=config.latent_dim,
        architecture_name=config.architecture_name,
        hidden_dims=config.hidden_dims,
        decoder_hidden_dims=config.decoder_hidden_dims,
        activation=resolve_activation(config.activation_name),
    )
    projected_residual_penalty_weight = _effective_projected_weight(config)
    history = train_autoencoder(
        autoencoder,
        normalized_snapshots,
        epochs=config.autoencoder_epochs,
        learning_rate=config.autoencoder_lr,
        smoothness_weight=config.smoothness_weight,
        projected_residual_penalty_weight=projected_residual_penalty_weight,
        ambient_residual_penalty_weight=config.ambient_residual_penalty_weight,
        residual_penalty_schedule=config.residual_penalty_schedule,
        training_objective_mode=config.training_objective_mode,
        online_residual_sample_count=config.online_residual_sample_count,
        online_residual_sample_scale=config.online_residual_sample_scale,
        online_residual_sample_seed_offset=config.online_residual_sample_seed_offset,
        online_residual_sample_source=config.online_residual_sample_source,
        parameters=train_parameters,
        physics_penalty_fn=lambda reconstructed, latent_batch, parameter_batch: _physics_penalty_terms(
            reconstructed,
            latent_batch,
            parameter_batch,
            problem,
            scaler,
            autoencoder,
            config.residual_penalty_metric,
        ),
        seed=config.seed,
    )
    reconstructed_train = autoencoder(np_to_torch(normalized_snapshots)).detach().cpu().numpy()
    reconstructed_train_physical = scaler.inverse_transform(reconstructed_train)
    ae_reconstruction_error = _relative_l2(train_snapshots.ravel(), reconstructed_train_physical.ravel())
    train_latents = autoencoder.encode(np_to_torch(normalized_snapshots)).detach().cpu().numpy()
    train_latent_centroid = np.mean(train_latents, axis=0)

    return {
        "artifact_version": 3,
        "artifact_type": "offline_manifold_artifact",
        "config": asdict(config),
        "metadata": runtime_metadata,
        "cache_key": offline_artifact_cache_key(config),
        "problem_name": config.problem_name,
        "qoi_name": problem.qoi_name,
        "train_parameters": train_parameters.tolist(),
        "train_snapshots": train_snapshots.tolist(),
        "train_snapshot_shape": list(train_snapshots.shape),
        "scaler": {
            "mean": scaler.mean.detach().cpu().numpy().tolist(),
            "std": scaler.std.detach().cpu().numpy().tolist(),
        },
        "autoencoder_state_dict": autoencoder.state_dict(),
        "autoencoder_final_loss": history["loss"][-1],
        "autoencoder_history": history,
        "autoencoder_reconstruction_error": ae_reconstruction_error,
        "autoencoder_final_residual_penalty": history["residual"][-1],
        "autoencoder_final_projected_residual_penalty": history["projected_residual"][-1],
        "autoencoder_final_ambient_residual_penalty": history["ambient_residual"][-1],
        "train_latent_centroid": train_latent_centroid.tolist(),
    }


def build_offline_artifact(config: BenchmarkConfig) -> dict:
    with _benchmark_runtime_context(config):
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        runtime_metadata = _publication_runtime_metadata(config)
        return _build_offline_artifact_data(config, runtime_metadata)


def load_or_build_offline_artifact(config: BenchmarkConfig) -> tuple[dict, dict]:
    cache_path = offline_artifact_cache_path(config)
    cache_info = {
        "cache_key": offline_artifact_cache_key(config),
        "cache_path": str(cache_path),
        "cache_hit": False,
        "cache_enabled": bool(config.use_offline_cache),
    }
    if config.use_offline_cache and not config.refresh_offline_cache and cache_path.exists():
        artifact = torch.load(cache_path, map_location="cpu")
        cache_info["cache_hit"] = True
        return artifact, cache_info

    artifact = build_offline_artifact(config)
    if config.use_offline_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, cache_path)
    return artifact, cache_info


def _evaluate_benchmark_from_artifact_data(artifact: dict, config: BenchmarkConfig, runtime_metadata: dict) -> dict:
    problem = create_problem(artifact["problem_name"])
    autoencoder = _instantiate_autoencoder(artifact)
    scaler = _instantiate_scaler(artifact)
    train_parameters = np.asarray(artifact["train_parameters"], dtype=np.float64)
    train_snapshots = np.asarray(artifact["train_snapshots"], dtype=np.float64)
    train_latent_centroid = np.asarray(artifact["train_latent_centroid"], dtype=np.float64)
    all_test_parameters, test_parameters = _resolve_test_parameters(problem, config)

    solver_kwargs = {
        "solver_tol": config.nonlinear_solver_tol,
        "solver_max_iter": config.nonlinear_solver_max_iter,
        "solver_line_search_steps": config.nonlinear_solver_line_search_steps,
        "solver_min_step_scale": config.nonlinear_solver_min_step_scale,
    }
    mgp_rom = ManifoldGalerkinROM(
        autoencoder=autoencoder,
        scaler=scaler,
        problem=problem,
        **solver_kwargs,
    )
    mgp_lspg_rom = ManifoldLSPGROM(
        autoencoder=autoencoder,
        scaler=scaler,
        problem=problem,
        **solver_kwargs,
    )
    pod_rom = PODGalerkinROM.fit(
        train_snapshots,
        reduced_dim=config.pod_dim,
        problem=problem,
        **solver_kwargs,
    )
    lspg_rom = PODLSPGROM.fit(
        train_snapshots,
        reduced_dim=config.pod_dim,
        problem=problem,
        **solver_kwargs,
    )

    full_cases = []
    for mu in test_parameters:
        full_start = perf_counter()
        full = problem.solve_full(mu)
        full_time = perf_counter() - full_start
        full_state = full.solution.detach().cpu().numpy()
        full_cases.append(
            {
                "mu": np.asarray(mu, dtype=np.float64).tolist(),
                "full_state": full_state.tolist(),
                "full_residual": float(full.residual_norm),
                "full_qoi": float(problem.quantity_of_interest(full_state)),
                "full_time_sec": float(full_time),
            }
        )

    per_case = []
    for case in full_cases:
        mu = np.asarray(case["mu"], dtype=np.float64)
        full_state = np.asarray(case["full_state"], dtype=np.float64)
        full_time = float(case["full_time_sec"])
        full_qoi = float(case["full_qoi"])

        warm_state = _select_warm_start_state(
            config.warm_start_strategy,
            mu,
            train_parameters,
            train_snapshots,
        )
        initial_latent = mgp_rom.encode(warm_state)
        initial_pod = pod_rom.project_state(warm_state)
        initial_lspg = lspg_rom.project_state(warm_state)

        mgp_start = perf_counter()
        mgp = mgp_rom.solve(mu, initial=initial_latent)
        mgp_time = perf_counter() - mgp_start
        mgp_latent = mgp.solution.detach().cpu().numpy()
        mgp_state = mgp_rom.decode(mgp_latent)
        mgp_qoi = problem.quantity_of_interest(mgp_state)
        mgp_projected_residual = _projected_residual_norm(mgp_rom, mgp_latent, mu)
        mgp_latent_distance = float(np.linalg.norm(mgp_latent - train_latent_centroid))
        mgp_decoder_jacobian_fro, mgp_decoder_jacobian_condition = _decoder_jacobian_diagnostics(
            autoencoder, mgp_latent
        )

        mgp_lspg_start = perf_counter()
        mgp_lspg = mgp_lspg_rom.solve(mu, initial=initial_latent)
        mgp_lspg_time = perf_counter() - mgp_lspg_start
        mgp_lspg_latent = mgp_lspg.solution.detach().cpu().numpy()
        mgp_lspg_state = mgp_lspg_rom.decode(mgp_lspg_latent)
        mgp_lspg_qoi = problem.quantity_of_interest(mgp_lspg_state)
        mgp_lspg_projected_residual = _projected_residual_norm(mgp_rom, mgp_lspg_latent, mu)
        mgp_lspg_latent_distance = float(np.linalg.norm(mgp_lspg_latent - train_latent_centroid))
        (
            mgp_lspg_decoder_jacobian_fro,
            mgp_lspg_decoder_jacobian_condition,
        ) = _decoder_jacobian_diagnostics(autoencoder, mgp_lspg_latent)

        pod_start = perf_counter()
        pod = pod_rom.solve(mu, initial=initial_pod)
        pod_time = perf_counter() - pod_start
        pod_state = pod_rom.reconstruct(pod.solution.detach().cpu().numpy())
        pod_qoi = problem.quantity_of_interest(pod_state)

        lspg_start = perf_counter()
        lspg = lspg_rom.solve(mu, initial=initial_lspg)
        lspg_time = perf_counter() - lspg_start
        lspg_state = lspg_rom.reconstruct(lspg.solution.detach().cpu().numpy())
        lspg_qoi = problem.quantity_of_interest(lspg_state)

        per_case.append(
            {
                "mu": mu.tolist(),
                "full_residual": float(case["full_residual"]),
                "full_qoi": full_qoi,
                "mgp_error_l2": _relative_l2(full_state, mgp_state),
                "mgp_residual": problem.residual_torch(np_to_torch(mgp_state), mu).norm().item(),
                "mgp_qoi": mgp_qoi,
                "mgp_qoi_error": problem.qoi_error(full_qoi, mgp_qoi),
                "mgp_projected_residual": mgp_projected_residual,
                "mgp_converged": mgp.converged,
                "mgp_iterations": mgp.iterations,
                "mgp_time_sec": mgp_time,
                "mgp_latent_distance": mgp_latent_distance,
                "mgp_decoder_jacobian_fro": mgp_decoder_jacobian_fro,
                "mgp_decoder_jacobian_condition": mgp_decoder_jacobian_condition,
                "mgp_lspg_error_l2": _relative_l2(full_state, mgp_lspg_state),
                "mgp_lspg_residual": problem.residual_torch(
                    np_to_torch(mgp_lspg_state), mu
                ).norm().item(),
                "mgp_lspg_qoi": mgp_lspg_qoi,
                "mgp_lspg_qoi_error": problem.qoi_error(full_qoi, mgp_lspg_qoi),
                "mgp_lspg_projected_residual": mgp_lspg_projected_residual,
                "mgp_lspg_converged": mgp_lspg.converged,
                "mgp_lspg_iterations": mgp_lspg.iterations,
                "mgp_lspg_time_sec": mgp_lspg_time,
                "mgp_lspg_latent_distance": mgp_lspg_latent_distance,
                "mgp_lspg_decoder_jacobian_fro": mgp_lspg_decoder_jacobian_fro,
                "mgp_lspg_decoder_jacobian_condition": mgp_lspg_decoder_jacobian_condition,
                "pod_error_l2": _relative_l2(full_state, pod_state),
                "pod_residual": problem.residual_torch(np_to_torch(pod_state), mu).norm().item(),
                "pod_qoi": pod_qoi,
                "pod_qoi_error": problem.qoi_error(full_qoi, pod_qoi),
                "pod_converged": pod.converged,
                "pod_iterations": pod.iterations,
                "pod_time_sec": pod_time,
                "lspg_error_l2": _relative_l2(full_state, lspg_state),
                "lspg_residual": problem.residual_torch(np_to_torch(lspg_state), mu).norm().item(),
                "lspg_qoi": lspg_qoi,
                "lspg_qoi_error": problem.qoi_error(full_qoi, lspg_qoi),
                "lspg_converged": lspg.converged,
                "lspg_iterations": lspg.iterations,
                "lspg_time_sec": lspg_time,
                "full_time_sec": full_time,
            }
        )

    mgp_errors = np.array([case["mgp_error_l2"] for case in per_case])
    mgp_lspg_errors = np.array([case["mgp_lspg_error_l2"] for case in per_case])
    pod_errors = np.array([case["pod_error_l2"] for case in per_case])
    lspg_errors = np.array([case["lspg_error_l2"] for case in per_case])
    mgp_residuals = np.array([case["mgp_residual"] for case in per_case])
    mgp_lspg_residuals = np.array([case["mgp_lspg_residual"] for case in per_case])
    mgp_projected_residuals = np.array([case["mgp_projected_residual"] for case in per_case])
    mgp_lspg_projected_residuals = np.array([case["mgp_lspg_projected_residual"] for case in per_case])
    pod_residuals = np.array([case["pod_residual"] for case in per_case])
    lspg_residuals = np.array([case["lspg_residual"] for case in per_case])
    mgp_qoi_errors = np.array([case["mgp_qoi_error"] for case in per_case])
    mgp_lspg_qoi_errors = np.array([case["mgp_lspg_qoi_error"] for case in per_case])
    pod_qoi_errors = np.array([case["pod_qoi_error"] for case in per_case])
    lspg_qoi_errors = np.array([case["lspg_qoi_error"] for case in per_case])
    mgp_times = np.array([case["mgp_time_sec"] for case in per_case])
    mgp_lspg_times = np.array([case["mgp_lspg_time_sec"] for case in per_case])
    pod_times = np.array([case["pod_time_sec"] for case in per_case])
    lspg_times = np.array([case["lspg_time_sec"] for case in per_case])
    full_times = np.array([case["full_time_sec"] for case in per_case])
    mgp_converged = np.array([case["mgp_converged"] for case in per_case], dtype=np.float64)
    mgp_lspg_converged = np.array([case["mgp_lspg_converged"] for case in per_case], dtype=np.float64)
    pod_converged = np.array([case["pod_converged"] for case in per_case], dtype=np.float64)
    lspg_converged = np.array([case["lspg_converged"] for case in per_case], dtype=np.float64)
    mgp_latent_distances = np.array([case["mgp_latent_distance"] for case in per_case])
    mgp_lspg_latent_distances = np.array([case["mgp_lspg_latent_distance"] for case in per_case])
    mgp_decoder_jacobian_fro = np.array([case["mgp_decoder_jacobian_fro"] for case in per_case])
    mgp_lspg_decoder_jacobian_fro = np.array(
        [case["mgp_lspg_decoder_jacobian_fro"] for case in per_case]
    )
    mgp_decoder_jacobian_condition = np.array(
        [case["mgp_decoder_jacobian_condition"] for case in per_case]
    )
    mgp_lspg_decoder_jacobian_condition = np.array(
        [case["mgp_lspg_decoder_jacobian_condition"] for case in per_case]
    )
    held_out = np.asarray([np.asarray(case["full_state"], dtype=np.float64) for case in full_cases])
    normalized_held_out = scaler.transform(held_out)
    reconstructed = autoencoder(np_to_torch(normalized_held_out)).detach().cpu().numpy()
    reconstructed_physical = scaler.inverse_transform(reconstructed)
    ae_reconstruction_error = _relative_l2(held_out.ravel(), reconstructed_physical.ravel())
    reconstruction_cases = _reconstruction_case_metrics(
        autoencoder,
        scaler,
        problem,
        held_out,
        test_parameters,
    )
    reconstruction_errors = np.array([case["reconstruction_error_l2"] for case in reconstruction_cases])
    reconstruction_residuals = np.array([case["reconstruction_residual"] for case in reconstruction_cases])
    reconstruction_projected_residuals = np.array(
        [case["reconstruction_projected_residual"] for case in reconstruction_cases]
    )

    return {
        "config": asdict(config),
        "metadata": runtime_metadata,
        "offline_artifact": {
            "cache_key": artifact.get("cache_key"),
            "artifact_version": artifact.get("artifact_version"),
        },
        "qoi_name": artifact["qoi_name"],
        "train_snapshot_shape": artifact["train_snapshot_shape"],
        "autoencoder_final_loss": artifact["autoencoder_final_loss"],
        "autoencoder_reconstruction_error": ae_reconstruction_error,
        "autoencoder_final_residual_penalty": artifact["autoencoder_final_residual_penalty"],
        "autoencoder_final_projected_residual_penalty": artifact["autoencoder_final_projected_residual_penalty"],
        "autoencoder_final_ambient_residual_penalty": artifact["autoencoder_final_ambient_residual_penalty"],
        "train_latent_centroid": artifact["train_latent_centroid"],
        "test_parameters": test_parameters.tolist(),
        "all_test_parameters": all_test_parameters.tolist(),
        "summary": {
            "mgp_mean_error_l2": float(mgp_errors.mean()),
            "mgp_lspg_mean_error_l2": float(mgp_lspg_errors.mean()),
            "pod_mean_error_l2": float(pod_errors.mean()),
            "lspg_mean_error_l2": float(lspg_errors.mean()),
            "mgp_median_error_l2": float(np.median(mgp_errors)),
            "mgp_lspg_median_error_l2": float(np.median(mgp_lspg_errors)),
            "pod_median_error_l2": float(np.median(pod_errors)),
            "lspg_median_error_l2": float(np.median(lspg_errors)),
            "mgp_mean_residual": float(mgp_residuals.mean()),
            "mgp_lspg_mean_residual": float(mgp_lspg_residuals.mean()),
            "mgp_mean_projected_residual": float(mgp_projected_residuals.mean()),
            "mgp_lspg_mean_projected_residual": float(mgp_lspg_projected_residuals.mean()),
            "pod_mean_residual": float(pod_residuals.mean()),
            "lspg_mean_residual": float(lspg_residuals.mean()),
            "mgp_mean_qoi_error": float(mgp_qoi_errors.mean()),
            "mgp_lspg_mean_qoi_error": float(mgp_lspg_qoi_errors.mean()),
            "pod_mean_qoi_error": float(pod_qoi_errors.mean()),
            "lspg_mean_qoi_error": float(lspg_qoi_errors.mean()),
            "mgp_convergence_rate": float(mgp_converged.mean()),
            "mgp_lspg_convergence_rate": float(mgp_lspg_converged.mean()),
            "pod_convergence_rate": float(pod_converged.mean()),
            "lspg_convergence_rate": float(lspg_converged.mean()),
            "mgp_mean_latent_distance": float(mgp_latent_distances.mean()),
            "mgp_lspg_mean_latent_distance": float(mgp_lspg_latent_distances.mean()),
            "mgp_mean_decoder_jacobian_fro": float(mgp_decoder_jacobian_fro.mean()),
            "mgp_lspg_mean_decoder_jacobian_fro": float(mgp_lspg_decoder_jacobian_fro.mean()),
            "mgp_mean_decoder_jacobian_condition": float(mgp_decoder_jacobian_condition.mean()),
            "mgp_lspg_mean_decoder_jacobian_condition": float(
                mgp_lspg_decoder_jacobian_condition.mean()
            ),
            "reconstruction_mean_error_l2": float(reconstruction_errors.mean()),
            "reconstruction_mean_residual": float(reconstruction_residuals.mean()),
            "reconstruction_mean_projected_residual": float(
                reconstruction_projected_residuals.mean()
            ),
            "mean_full_time_sec": float(full_times.mean()),
            "mean_mgp_time_sec": float(mgp_times.mean()),
            "mean_mgp_lspg_time_sec": float(mgp_lspg_times.mean()),
            "mean_pod_time_sec": float(pod_times.mean()),
            "mean_lspg_time_sec": float(lspg_times.mean()),
            "mgp_speedup_vs_full": float(full_times.mean() / max(mgp_times.mean(), 1e-12)),
            "mgp_lspg_speedup_vs_full": float(full_times.mean() / max(mgp_lspg_times.mean(), 1e-12)),
            "pod_speedup_vs_full": float(full_times.mean() / max(pod_times.mean(), 1e-12)),
            "lspg_speedup_vs_full": float(full_times.mean() / max(lspg_times.mean(), 1e-12)),
        },
        "cases": per_case,
        "reconstruction_cases": reconstruction_cases,
    }


def evaluate_benchmark_from_offline_artifact(artifact: dict, config: BenchmarkConfig) -> dict:
    with _benchmark_runtime_context(config):
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        runtime_metadata = _publication_runtime_metadata(config)
        return _evaluate_benchmark_from_artifact_data(artifact, config, runtime_metadata)


@contextmanager
def _benchmark_runtime_context(config: BenchmarkConfig):
    previous_num_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_omp_num_threads = os.environ.get("OMP_NUM_THREADS")
    if config.deterministic_publication_mode:
        os.environ["OMP_NUM_THREADS"] = str(config.publication_num_threads)
        torch.set_num_threads(config.publication_num_threads)
        torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        if config.deterministic_publication_mode:
            torch.use_deterministic_algorithms(previous_deterministic)
            torch.set_num_threads(previous_num_threads)
            if previous_omp_num_threads is None:
                os.environ.pop("OMP_NUM_THREADS", None)
            else:
                os.environ["OMP_NUM_THREADS"] = previous_omp_num_threads


def run_benchmark(config: BenchmarkConfig) -> dict:
    artifact, cache_info = load_or_build_offline_artifact(config)
    result = evaluate_benchmark_from_offline_artifact(artifact, config)
    result["offline_artifact"].update(cache_info)
    return result
