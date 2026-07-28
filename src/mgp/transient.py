from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .autoencoder import SnapshotAutoencoder, SnapshotScaler, resolve_activation
from .benchmark import _metric_norm, _relative_l2, np_to_torch
from .newton import NewtonResult, solve_nonlinear_system
from .pod_rom import PODGalerkinROM, PODLSPGROM
from .problems import FrontLayerProblem


@dataclass
class TransientBenchmarkConfig:
    train_size: int = 24
    test_size: int = 12
    test_full_size: int | None = None
    test_case_start: int = 0
    test_case_count: int | None = None
    test_regime: str = "in_domain"
    latent_dim: int = 2
    pod_dim: int = 2
    architecture_name: str = "conv1d"
    hidden_dims: tuple[int, ...] = (24, 24, 24)
    decoder_hidden_dims: tuple[int, ...] = (48, 48)
    activation_name: str = "silu"
    autoencoder_epochs: int = 250
    autoencoder_lr: float = 1e-3
    smoothness_weight: float = 0.0
    projected_residual_penalty_weight: float = 0.02
    ambient_residual_penalty_weight: float = 0.005
    temporal_smoothness_weight: float = 0.0
    residual_penalty_metric: str = "l2_norm"
    objective_mode: str = "snapshot_reference"
    seed: int = 0
    deterministic_publication_mode: bool = False
    publication_num_threads: int = 1
    dt: float = 0.01
    num_steps: int = 100
    num_nodes: int = 65


@dataclass
class TransientFrontLayerProblem:
    num_nodes: int = 65
    dt: float = 0.01
    num_steps: int = 100

    def __post_init__(self) -> None:
        self.steady_problem = FrontLayerProblem(num_nodes=self.num_nodes)
        self.interior_dim = self.steady_problem.interior_dim
        self.qoi_name = self.steady_problem.qoi_name

    def sample_parameters(self, num_samples: int, *, seed: int = 0) -> np.ndarray:
        return self.steady_problem.sample_parameters(num_samples, seed=seed)

    def sample_parameters_ood(self, num_samples: int, *, seed: int = 0) -> np.ndarray:
        return self.steady_problem.sample_parameters_ood(num_samples, seed=seed)

    def initial_state(self) -> np.ndarray:
        return np.zeros(self.interior_dim, dtype=np.float64)

    def snapshot_residual_torch(self, u_interior: torch.Tensor, mu) -> torch.Tensor:
        return self.steady_problem.residual_torch(u_interior, mu)

    def step_residual_torch(self, u_next: torch.Tensor, mu, previous_state) -> torch.Tensor:
        if not isinstance(previous_state, torch.Tensor):
            previous_state = torch.tensor(previous_state, dtype=torch.float64, device=u_next.device)
        else:
            previous_state = previous_state.to(dtype=torch.float64, device=u_next.device)
        return u_next - previous_state + self.dt * self.snapshot_residual_torch(u_next, mu)

    def solve_step(self, mu, previous_state: np.ndarray, *, initial: np.ndarray | None = None) -> NewtonResult:
        if initial is None:
            x0 = torch.tensor(previous_state, dtype=torch.float64)
        else:
            x0 = torch.tensor(initial, dtype=torch.float64)
        return solve_nonlinear_system(lambda x: self.step_residual_torch(x, mu, previous_state), x0)

    def solve_trajectory(self, mu, *, initial_state: np.ndarray | None = None) -> dict:
        previous_state = (
            self.initial_state() if initial_state is None else np.asarray(initial_state, dtype=np.float64)
        )
        states = [previous_state.copy()]
        step_residuals = []
        converged = []
        iterations = []
        step_times = []
        for _ in range(self.num_steps):
            start = perf_counter()
            result = self.solve_step(mu, previous_state, initial=previous_state)
            step_times.append(perf_counter() - start)
            current_state = result.solution.detach().cpu().numpy()
            states.append(current_state.copy())
            step_residuals.append(float(result.residual_norm))
            converged.append(bool(result.converged))
            iterations.append(int(result.iterations))
            previous_state = current_state
        return {
            "states": np.asarray(states, dtype=np.float64),
            "step_residuals": step_residuals,
            "converged": converged,
            "iterations": iterations,
            "step_times": step_times,
        }

    def quantity_of_interest(self, u_interior: np.ndarray) -> float:
        return self.steady_problem.quantity_of_interest(u_interior)

    def qoi_error(self, reference: float, candidate: float) -> float:
        return self.steady_problem.qoi_error(reference, candidate)


def _sample_test_parameters(
    problem: TransientFrontLayerProblem, test_size: int, *, regime: str, seed: int
) -> np.ndarray:
    if regime == "in_domain":
        return problem.sample_parameters(test_size, seed=seed)
    if regime == "ood":
        return problem.sample_parameters_ood(test_size, seed=seed)
    raise ValueError(f"unknown test_regime: {regime}")


def _runtime_metadata(config: TransientBenchmarkConfig) -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "deterministic_publication_mode": config.deterministic_publication_mode,
        "publication_num_threads_requested": config.publication_num_threads,
        "torch_num_threads": torch.get_num_threads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


@contextmanager
def _runtime_context(config: TransientBenchmarkConfig):
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


def _decode_physical(autoencoder: SnapshotAutoencoder, scaler: SnapshotScaler, latent: torch.Tensor) -> torch.Tensor:
    normalized_state = autoencoder.decode(latent)
    return scaler.inverse_transform_tensor(normalized_state)


def _build_training_pairs(
    trajectories: np.ndarray, parameters: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    previous_states = trajectories[:, :-1, :].reshape(-1, trajectories.shape[-1])
    next_states = trajectories[:, 1:, :].reshape(-1, trajectories.shape[-1])
    repeated_parameters = np.repeat(parameters, trajectories.shape[1] - 1, axis=0)
    return previous_states, next_states, repeated_parameters


def _latent_temporal_smoothness(
    model: SnapshotAutoencoder,
    scaler: SnapshotScaler,
    batch_next: torch.Tensor,
    batch_prev: torch.Tensor,
) -> torch.Tensor:
    prev_normalized = scaler.transform_tensor(batch_prev)
    prev_latent = model.encode(prev_normalized)
    next_latent = model.encode(batch_next)
    return torch.mean((next_latent - prev_latent) ** 2)


def _snapshot_penalty_terms(
    reconstructed_normalized: torch.Tensor,
    latent_batch: torch.Tensor,
    parameter_batch: torch.Tensor,
    problem: TransientFrontLayerProblem,
    scaler: SnapshotScaler,
    autoencoder: SnapshotAutoencoder,
    metric: str,
) -> dict[str, torch.Tensor]:
    projected_penalties = []
    ambient_penalties = []
    for reconstructed_sample, latent_sample, mu in zip(reconstructed_normalized, latent_batch, parameter_batch):
        physical_state = scaler.inverse_transform_tensor(reconstructed_sample)
        ambient_residual = problem.snapshot_residual_torch(physical_state, mu)

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


def _step_penalty_terms(
    reconstructed_normalized: torch.Tensor,
    latent_batch: torch.Tensor,
    parameter_batch: torch.Tensor,
    previous_state_batch: torch.Tensor,
    problem: TransientFrontLayerProblem,
    scaler: SnapshotScaler,
    autoencoder: SnapshotAutoencoder,
    metric: str,
) -> dict[str, torch.Tensor]:
    projected_penalties = []
    ambient_penalties = []
    for reconstructed_sample, latent_sample, mu, previous_state in zip(
        reconstructed_normalized, latent_batch, parameter_batch, previous_state_batch
    ):
        physical_state = scaler.inverse_transform_tensor(reconstructed_sample)
        ambient_residual = problem.step_residual_torch(physical_state, mu, previous_state)

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


def _train_transient_autoencoder(
    config: TransientBenchmarkConfig,
    problem: TransientFrontLayerProblem,
    scaler: SnapshotScaler,
    snapshots: np.ndarray,
    parameters: np.ndarray,
    *,
    previous_states: np.ndarray | None,
) -> tuple[SnapshotAutoencoder, dict[str, list[float]]]:
    model = SnapshotAutoencoder(
        input_dim=problem.interior_dim,
        latent_dim=config.latent_dim,
        architecture_name=config.architecture_name,
        hidden_dims=config.hidden_dims,
        decoder_hidden_dims=config.decoder_hidden_dims,
        activation=resolve_activation(config.activation_name),
    )

    normalized_snapshots = scaler.transform(snapshots)
    snapshot_tensor = torch.tensor(normalized_snapshots, dtype=torch.float64)
    parameter_tensor = torch.tensor(parameters, dtype=torch.float64)
    if previous_states is None:
        dataset = TensorDataset(snapshot_tensor, parameter_tensor)
    else:
        previous_tensor = torch.tensor(previous_states, dtype=torch.float64)
        dataset = TensorDataset(snapshot_tensor, parameter_tensor, previous_tensor)
    loader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.autoencoder_lr)

    history = {
        "loss": [],
        "reconstruction": [],
        "projected_residual": [],
        "ambient_residual": [],
        "temporal_smoothness": [],
    }

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    for _ in range(config.autoencoder_epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_projected = 0.0
        total_ambient = 0.0
        total_temporal = 0.0
        seen = 0
        for batch_items in loader:
            batch = batch_items[0]
            parameter_batch = batch_items[1]
            previous_batch = batch_items[2] if len(batch_items) > 2 else None

            optimizer.zero_grad()
            latent = model.encode(batch)
            reconstructed = model.decode(latent)
            recon_loss = torch.mean((reconstructed - batch) ** 2)
            projected_loss = torch.tensor(0.0, dtype=torch.float64)
            ambient_loss = torch.tensor(0.0, dtype=torch.float64)
            temporal_loss = torch.tensor(0.0, dtype=torch.float64)

            if config.objective_mode == "ambient_heavy_step_consistent":
                assert previous_batch is not None
                penalty_terms = _step_penalty_terms(
                    reconstructed,
                    latent,
                    parameter_batch,
                    previous_batch,
                    problem,
                    scaler,
                    model,
                    config.residual_penalty_metric,
                )
                if config.temporal_smoothness_weight > 0.0:
                    temporal_loss = _latent_temporal_smoothness(model, scaler, batch, previous_batch)
            else:
                penalty_terms = _snapshot_penalty_terms(
                    reconstructed,
                    latent,
                    parameter_batch,
                    problem,
                    scaler,
                    model,
                    config.residual_penalty_metric,
                )

            projected_loss = penalty_terms["projected"]
            ambient_loss = penalty_terms["ambient"]

            loss = recon_loss
            loss = loss + config.projected_residual_penalty_weight * projected_loss
            loss = loss + config.ambient_residual_penalty_weight * ambient_loss
            loss = loss + config.temporal_smoothness_weight * temporal_loss
            loss.backward()
            optimizer.step()

            batch_size = batch.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_recon += float(recon_loss.item()) * batch_size
            total_projected += float(projected_loss.item()) * batch_size
            total_ambient += float(ambient_loss.item()) * batch_size
            total_temporal += float(temporal_loss.item()) * batch_size
            seen += batch_size

        history["loss"].append(total_loss / max(seen, 1))
        history["reconstruction"].append(total_recon / max(seen, 1))
        history["projected_residual"].append(total_projected / max(seen, 1))
        history["ambient_residual"].append(total_ambient / max(seen, 1))
        history["temporal_smoothness"].append(total_temporal / max(seen, 1))

    return model, history


class _TransientManifoldGalerkinROM:
    def __init__(self, *, autoencoder: SnapshotAutoencoder, scaler: SnapshotScaler, problem: TransientFrontLayerProblem):
        self.autoencoder = autoencoder
        self.scaler = scaler
        self.problem = problem

    def encode(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            normalized = self.scaler.transform_tensor(torch.tensor(state, dtype=torch.float64))
            latent = self.autoencoder.encode(normalized)
        return latent.detach().cpu().numpy()

    def decode(self, latent: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return _decode_physical(
                self.autoencoder, self.scaler, torch.tensor(latent, dtype=torch.float64)
            ).detach().cpu().numpy()

    def reduced_step_residual_torch(self, latent: torch.Tensor, mu, previous_state) -> torch.Tensor:
        latent = latent.to(dtype=torch.float64)

        def decode_physical(z: torch.Tensor) -> torch.Tensor:
            return _decode_physical(self.autoencoder, self.scaler, z)

        state = decode_physical(latent)
        tangent = torch.autograd.functional.jacobian(decode_physical, latent, create_graph=True)
        ambient_residual = self.problem.step_residual_torch(state, mu, previous_state)
        return tangent.T @ ambient_residual

    def solve_step(self, mu, previous_state: np.ndarray, *, initial: np.ndarray | None = None) -> NewtonResult:
        if initial is None:
            x0 = self.encode(previous_state)
        else:
            x0 = np.asarray(initial, dtype=np.float64)
        return solve_nonlinear_system(
            lambda z: self.reduced_step_residual_torch(z, mu, previous_state),
            torch.tensor(x0, dtype=torch.float64),
        )


class _TransientManifoldLSPGROM(_TransientManifoldGalerkinROM):
    def ambient_step_residual_torch(self, latent: torch.Tensor, mu, previous_state) -> torch.Tensor:
        latent = latent.to(dtype=torch.float64)
        state = _decode_physical(self.autoencoder, self.scaler, latent)
        return self.problem.step_residual_torch(state, mu, previous_state)

    def optimality_residual_torch(self, latent: torch.Tensor, mu, previous_state) -> torch.Tensor:
        def ambient_residual(local_latent: torch.Tensor) -> torch.Tensor:
            return self.ambient_step_residual_torch(local_latent, mu, previous_state)

        jacobian = torch.autograd.functional.jacobian(ambient_residual, latent, create_graph=True)
        residual = ambient_residual(latent)
        return jacobian.T @ residual

    def ambient_residual_norm_torch(self, latent: torch.Tensor, mu, previous_state) -> torch.Tensor:
        residual = self.ambient_step_residual_torch(latent, mu, previous_state)
        return 0.5 * torch.dot(residual, residual)

    def solve_step(self, mu, previous_state: np.ndarray, *, initial: np.ndarray | None = None) -> NewtonResult:
        if initial is None:
            x0 = self.encode(previous_state)
        else:
            x0 = np.asarray(initial, dtype=np.float64)
        return solve_nonlinear_system(
            lambda z: self.optimality_residual_torch(z, mu, previous_state),
            torch.tensor(x0, dtype=torch.float64),
            merit_fn=lambda z: self.ambient_residual_norm_torch(z, mu, previous_state),
        )


class _TransientPODGalerkinROM:
    def __init__(self, *, basis: np.ndarray, mean: np.ndarray, problem: TransientFrontLayerProblem):
        self.basis = basis
        self.mean = mean
        self.problem = problem

    @classmethod
    def fit(cls, snapshots: np.ndarray, reduced_dim: int, problem: TransientFrontLayerProblem):
        steady = PODGalerkinROM.fit(snapshots, reduced_dim, problem.steady_problem)
        return cls(basis=steady.basis, mean=steady.mean, problem=problem)

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        return self.mean + self.basis @ coefficients

    def project_state(self, state: np.ndarray) -> np.ndarray:
        return self.basis.T @ (state - self.mean)

    def reduced_step_residual_torch(self, coefficients: torch.Tensor, mu, previous_state) -> torch.Tensor:
        basis = torch.tensor(self.basis, dtype=torch.float64, device=coefficients.device)
        mean = torch.tensor(self.mean, dtype=torch.float64, device=coefficients.device)
        state = mean + basis @ coefficients
        residual = self.problem.step_residual_torch(state, mu, previous_state)
        return basis.T @ residual

    def solve_step(self, mu, previous_state: np.ndarray, *, initial: np.ndarray | None = None) -> NewtonResult:
        if initial is None:
            x0 = self.project_state(previous_state)
        else:
            x0 = np.asarray(initial, dtype=np.float64)
        return solve_nonlinear_system(
            lambda coeffs: self.reduced_step_residual_torch(coeffs, mu, previous_state),
            torch.tensor(x0, dtype=torch.float64),
        )


class _TransientPODLSPGROM(_TransientPODGalerkinROM):
    def residual_torch(self, coefficients: torch.Tensor, mu, previous_state) -> torch.Tensor:
        basis = torch.tensor(self.basis, dtype=torch.float64, device=coefficients.device)
        mean = torch.tensor(self.mean, dtype=torch.float64, device=coefficients.device)
        state = mean + basis @ coefficients
        return self.problem.step_residual_torch(state, mu, previous_state)

    def optimality_residual_torch(self, coefficients: torch.Tensor, mu, previous_state) -> torch.Tensor:
        def ambient_residual(local_coefficients: torch.Tensor) -> torch.Tensor:
            return self.residual_torch(local_coefficients, mu, previous_state)

        jacobian = torch.autograd.functional.jacobian(ambient_residual, coefficients, create_graph=True)
        residual = ambient_residual(coefficients)
        return jacobian.T @ residual

    def residual_norm_torch(self, coefficients: torch.Tensor, mu, previous_state) -> torch.Tensor:
        residual = self.residual_torch(coefficients, mu, previous_state)
        return 0.5 * torch.dot(residual, residual)

    def solve_step(self, mu, previous_state: np.ndarray, *, initial: np.ndarray | None = None) -> NewtonResult:
        if initial is None:
            x0 = self.project_state(previous_state)
        else:
            x0 = np.asarray(initial, dtype=np.float64)
        return solve_nonlinear_system(
            lambda coeffs: self.optimality_residual_torch(coeffs, mu, previous_state),
            torch.tensor(x0, dtype=torch.float64),
            merit_fn=lambda coeffs: self.residual_norm_torch(coeffs, mu, previous_state),
        )


def _trajectory_metrics(
    problem: TransientFrontLayerProblem,
    full_trajectory: np.ndarray,
    candidate_trajectory: np.ndarray,
    candidate_step_residuals: list[float],
) -> dict:
    step_errors = []
    step_qoi_errors = []
    reference_qoi = []
    candidate_qoi = []
    for reference_state, candidate_state in zip(full_trajectory[1:], candidate_trajectory[1:]):
        step_errors.append(_relative_l2(reference_state, candidate_state))
        full_qoi = problem.quantity_of_interest(reference_state)
        pred_qoi = problem.quantity_of_interest(candidate_state)
        step_qoi_errors.append(problem.qoi_error(full_qoi, pred_qoi))
        reference_qoi.append(full_qoi)
        candidate_qoi.append(pred_qoi)

    return {
        "mean_state_error_l2": float(np.mean(step_errors)),
        "terminal_state_error_l2": float(step_errors[-1]),
        "mean_qoi_error": float(np.mean(step_qoi_errors)),
        "terminal_qoi_error": float(step_qoi_errors[-1]),
        "mean_step_residual": float(np.mean(candidate_step_residuals)),
        "terminal_step_residual": float(candidate_step_residuals[-1]),
        "reference_qoi_trajectory": reference_qoi,
        "candidate_qoi_trajectory": candidate_qoi,
    }


def _solve_rom_trajectory(
    rom,
    problem: TransientFrontLayerProblem,
    mu: np.ndarray,
) -> dict:
    previous_state = problem.initial_state()
    if hasattr(rom, "encode"):
        previous_reduced = rom.encode(previous_state)
    else:
        previous_reduced = rom.project_state(previous_state)

    states = [previous_state.copy()]
    step_residuals = []
    converged = []
    iterations = []
    step_times = []
    for _ in range(problem.num_steps):
        start = perf_counter()
        result = rom.solve_step(mu, previous_state, initial=previous_reduced)
        step_times.append(perf_counter() - start)
        reduced_state = result.solution.detach().cpu().numpy()
        if hasattr(rom, "decode"):
            current_state = rom.decode(reduced_state)
        else:
            current_state = rom.reconstruct(reduced_state)
        states.append(current_state.copy())
        step_residuals.append(float(problem.step_residual_torch(np_to_torch(current_state), mu, previous_state).norm().item()))
        converged.append(bool(result.converged))
        iterations.append(int(result.iterations))
        previous_state = current_state
        previous_reduced = reduced_state

    return {
        "states": np.asarray(states, dtype=np.float64),
        "step_residuals": step_residuals,
        "converged": converged,
        "iterations": iterations,
        "step_times": step_times,
    }


def run_transient_benchmark(config: TransientBenchmarkConfig) -> dict:
    with _runtime_context(config):
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        if config.objective_mode not in {
            "snapshot_reference",
            "ambient_heavy_snapshot",
            "ambient_heavy_step_consistent",
        }:
            raise ValueError(f"unknown objective_mode: {config.objective_mode}")

        problem = TransientFrontLayerProblem(
            num_nodes=config.num_nodes,
            dt=config.dt,
            num_steps=config.num_steps,
        )
        train_parameters = problem.sample_parameters(config.train_size, seed=config.seed)
        full_test_size = config.test_full_size or config.test_size
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
            raise ValueError("transient test subset is empty")

        train_trajectories = np.asarray(
            [problem.solve_trajectory(mu)["states"] for mu in train_parameters],
            dtype=np.float64,
        )
        previous_states, next_states, repeated_parameters = _build_training_pairs(train_trajectories, train_parameters)
        scaler = SnapshotScaler.fit(next_states)
        autoencoder, history = _train_transient_autoencoder(
            config,
            problem,
            scaler,
            next_states,
            repeated_parameters,
            previous_states=previous_states if config.objective_mode == "ambient_heavy_step_consistent" else None,
        )
        autoencoder.eval()

        with torch.no_grad():
            normalized_next_states = scaler.transform(next_states)
            reconstructed = autoencoder(np_to_torch(normalized_next_states)).detach().cpu().numpy()
        reconstructed_physical = scaler.inverse_transform(reconstructed)
        reconstruction_error = _relative_l2(next_states.ravel(), reconstructed_physical.ravel())
        reconstruction_step_residuals = []
        reconstruction_snapshot_residuals = []
        for reconstructed_state, previous_state, mu in zip(reconstructed_physical, previous_states, repeated_parameters):
            reconstruction_step_residuals.append(
                float(problem.step_residual_torch(np_to_torch(reconstructed_state), mu, previous_state).norm().item())
            )
            reconstruction_snapshot_residuals.append(
                float(problem.snapshot_residual_torch(np_to_torch(reconstructed_state), mu).norm().item())
            )

        mgp_rom = _TransientManifoldGalerkinROM(autoencoder=autoencoder, scaler=scaler, problem=problem)
        mgp_lspg_rom = _TransientManifoldLSPGROM(autoencoder=autoencoder, scaler=scaler, problem=problem)
        pod_rom = _TransientPODGalerkinROM.fit(next_states, config.pod_dim, problem)
        lspg_rom = _TransientPODLSPGROM.fit(next_states, config.pod_dim, problem)

        cases = []
        for mu in test_parameters:
            full_start = perf_counter()
            full = problem.solve_trajectory(mu)
            full_time = perf_counter() - full_start
            mgp = _solve_rom_trajectory(mgp_rom, problem, mu)
            mgp_lspg = _solve_rom_trajectory(mgp_lspg_rom, problem, mu)
            pod = _solve_rom_trajectory(pod_rom, problem, mu)
            lspg = _solve_rom_trajectory(lspg_rom, problem, mu)

            full_metrics = _trajectory_metrics(problem, full["states"], full["states"], full["step_residuals"])
            mgp_metrics = _trajectory_metrics(problem, full["states"], mgp["states"], mgp["step_residuals"])
            mgp_lspg_metrics = _trajectory_metrics(problem, full["states"], mgp_lspg["states"], mgp_lspg["step_residuals"])
            pod_metrics = _trajectory_metrics(problem, full["states"], pod["states"], pod["step_residuals"])
            lspg_metrics = _trajectory_metrics(problem, full["states"], lspg["states"], lspg["step_residuals"])

            cases.append(
                {
                    "mu": np.asarray(mu, dtype=np.float64).tolist(),
                    "full_states": full["states"].tolist(),
                    "full_step_residuals": full["step_residuals"],
                    "full_time_sec": float(full_time),
                    "full_mean_state_error_l2": full_metrics["mean_state_error_l2"],
                    "mgp_mean_trajectory_state_error_l2": mgp_metrics["mean_state_error_l2"],
                    "mgp_terminal_state_error_l2": mgp_metrics["terminal_state_error_l2"],
                    "mgp_mean_trajectory_qoi_error": mgp_metrics["mean_qoi_error"],
                    "mgp_terminal_qoi_error": mgp_metrics["terminal_qoi_error"],
                    "mgp_mean_step_residual": mgp_metrics["mean_step_residual"],
                    "mgp_terminal_step_residual": mgp_metrics["terminal_step_residual"],
                    "mgp_failed_step_count": int(sum(not flag for flag in mgp["converged"])),
                    "mgp_time_sec": float(sum(mgp["step_times"])),
                    "mgp_lspg_mean_trajectory_state_error_l2": mgp_lspg_metrics["mean_state_error_l2"],
                    "mgp_lspg_terminal_state_error_l2": mgp_lspg_metrics["terminal_state_error_l2"],
                    "mgp_lspg_mean_trajectory_qoi_error": mgp_lspg_metrics["mean_qoi_error"],
                    "mgp_lspg_terminal_qoi_error": mgp_lspg_metrics["terminal_qoi_error"],
                    "mgp_lspg_mean_step_residual": mgp_lspg_metrics["mean_step_residual"],
                    "mgp_lspg_terminal_step_residual": mgp_lspg_metrics["terminal_step_residual"],
                    "mgp_lspg_failed_step_count": int(sum(not flag for flag in mgp_lspg["converged"])),
                    "mgp_lspg_time_sec": float(sum(mgp_lspg["step_times"])),
                    "pod_mean_trajectory_state_error_l2": pod_metrics["mean_state_error_l2"],
                    "pod_terminal_state_error_l2": pod_metrics["terminal_state_error_l2"],
                    "pod_mean_trajectory_qoi_error": pod_metrics["mean_qoi_error"],
                    "pod_terminal_qoi_error": pod_metrics["terminal_qoi_error"],
                    "pod_mean_step_residual": pod_metrics["mean_step_residual"],
                    "pod_terminal_step_residual": pod_metrics["terminal_step_residual"],
                    "pod_failed_step_count": int(sum(not flag for flag in pod["converged"])),
                    "pod_time_sec": float(sum(pod["step_times"])),
                    "lspg_mean_trajectory_state_error_l2": lspg_metrics["mean_state_error_l2"],
                    "lspg_terminal_state_error_l2": lspg_metrics["terminal_state_error_l2"],
                    "lspg_mean_trajectory_qoi_error": lspg_metrics["mean_qoi_error"],
                    "lspg_terminal_qoi_error": lspg_metrics["terminal_qoi_error"],
                    "lspg_mean_step_residual": lspg_metrics["mean_step_residual"],
                    "lspg_terminal_step_residual": lspg_metrics["terminal_step_residual"],
                    "lspg_failed_step_count": int(sum(not flag for flag in lspg["converged"])),
                    "lspg_time_sec": float(sum(lspg["step_times"])),
                }
            )

        def _mean_case(key: str) -> float:
            return float(np.mean([case[key] for case in cases]))

        evaluated_test_size = len(test_parameters)
        total_steps = evaluated_test_size * config.num_steps
        summary = {
            "mgp_mean_trajectory_state_error_l2": _mean_case("mgp_mean_trajectory_state_error_l2"),
            "mgp_terminal_state_error_l2": _mean_case("mgp_terminal_state_error_l2"),
            "mgp_mean_trajectory_qoi_error": _mean_case("mgp_mean_trajectory_qoi_error"),
            "mgp_terminal_qoi_error": _mean_case("mgp_terminal_qoi_error"),
            "mgp_mean_step_residual": _mean_case("mgp_mean_step_residual"),
            "mgp_terminal_step_residual": _mean_case("mgp_terminal_step_residual"),
            "mgp_lspg_mean_trajectory_state_error_l2": _mean_case("mgp_lspg_mean_trajectory_state_error_l2"),
            "mgp_lspg_terminal_state_error_l2": _mean_case("mgp_lspg_terminal_state_error_l2"),
            "mgp_lspg_mean_trajectory_qoi_error": _mean_case("mgp_lspg_mean_trajectory_qoi_error"),
            "mgp_lspg_terminal_qoi_error": _mean_case("mgp_lspg_terminal_qoi_error"),
            "mgp_lspg_mean_step_residual": _mean_case("mgp_lspg_mean_step_residual"),
            "mgp_lspg_terminal_step_residual": _mean_case("mgp_lspg_terminal_step_residual"),
            "pod_mean_trajectory_state_error_l2": _mean_case("pod_mean_trajectory_state_error_l2"),
            "pod_terminal_state_error_l2": _mean_case("pod_terminal_state_error_l2"),
            "pod_mean_trajectory_qoi_error": _mean_case("pod_mean_trajectory_qoi_error"),
            "pod_terminal_qoi_error": _mean_case("pod_terminal_qoi_error"),
            "pod_mean_step_residual": _mean_case("pod_mean_step_residual"),
            "pod_terminal_step_residual": _mean_case("pod_terminal_step_residual"),
            "lspg_mean_trajectory_state_error_l2": _mean_case("lspg_mean_trajectory_state_error_l2"),
            "lspg_terminal_state_error_l2": _mean_case("lspg_terminal_state_error_l2"),
            "lspg_mean_trajectory_qoi_error": _mean_case("lspg_mean_trajectory_qoi_error"),
            "lspg_terminal_qoi_error": _mean_case("lspg_terminal_qoi_error"),
            "lspg_mean_step_residual": _mean_case("lspg_mean_step_residual"),
            "lspg_terminal_step_residual": _mean_case("lspg_terminal_step_residual"),
            "mgp_failed_step_count": int(sum(case["mgp_failed_step_count"] for case in cases)),
            "mgp_lspg_failed_step_count": int(sum(case["mgp_lspg_failed_step_count"] for case in cases)),
            "pod_failed_step_count": int(sum(case["pod_failed_step_count"] for case in cases)),
            "lspg_failed_step_count": int(sum(case["lspg_failed_step_count"] for case in cases)),
            "mgp_convergence_rate": float(1.0 - sum(case["mgp_failed_step_count"] for case in cases) / max(total_steps, 1)),
            "mgp_lspg_convergence_rate": float(1.0 - sum(case["mgp_lspg_failed_step_count"] for case in cases) / max(total_steps, 1)),
            "pod_convergence_rate": float(1.0 - sum(case["pod_failed_step_count"] for case in cases) / max(total_steps, 1)),
            "lspg_convergence_rate": float(1.0 - sum(case["lspg_failed_step_count"] for case in cases) / max(total_steps, 1)),
            "mean_full_time_sec": _mean_case("full_time_sec"),
            "mean_mgp_time_sec": _mean_case("mgp_time_sec"),
            "mean_mgp_lspg_time_sec": _mean_case("mgp_lspg_time_sec"),
            "mean_pod_time_sec": _mean_case("pod_time_sec"),
            "mean_lspg_time_sec": _mean_case("lspg_time_sec"),
            "reconstruction_mean_error_l2": reconstruction_error,
            "reconstruction_mean_step_residual": float(np.mean(reconstruction_step_residuals)),
            "reconstruction_mean_snapshot_residual": float(np.mean(reconstruction_snapshot_residuals)),
        }

        return {
            "config": asdict(config),
            "metadata": _runtime_metadata(config),
            "problem_name": "front_layer_transient",
            "qoi_name": problem.qoi_name,
            "summary": summary,
            "cases": cases,
            "autoencoder_history": history,
            "train_parameters": train_parameters.tolist(),
            "test_parameters": test_parameters.tolist(),
            "all_test_parameters": all_test_parameters.tolist(),
        }
