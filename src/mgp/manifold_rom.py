from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .autoencoder import SnapshotAutoencoder, SnapshotScaler
from .newton import NewtonResult, solve_nonlinear_system


@dataclass
class ManifoldGalerkinROM:
    autoencoder: SnapshotAutoencoder
    scaler: SnapshotScaler
    problem: object
    solver_tol: float = 1e-8
    solver_max_iter: int = 25
    solver_line_search_steps: int = 8
    solver_min_step_scale: float = 1e-3

    @property
    def reduced_dim(self) -> int:
        return self.autoencoder.latent_dim

    def encode(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            normalized = self.scaler.transform_tensor(torch.tensor(state, dtype=torch.float64))
            latent = self.autoencoder.encode(normalized)
        return latent.detach().cpu().numpy()

    def decode(self, latent: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            decoded = self._decode_tensor(torch.tensor(latent, dtype=torch.float64))
        return decoded.detach().cpu().numpy()

    def _decode_tensor(self, latent: torch.Tensor) -> torch.Tensor:
        normalized_state = self.autoencoder.decode(latent)
        return self.scaler.inverse_transform_tensor(normalized_state)

    def reduced_residual_torch(self, latent: torch.Tensor, mu) -> torch.Tensor:
        latent = latent.to(dtype=torch.float64)

        def decode_physical(z: torch.Tensor) -> torch.Tensor:
            return self._decode_tensor(z)

        state = decode_physical(latent)
        tangent = torch.autograd.functional.jacobian(decode_physical, latent, create_graph=True)
        ambient_residual = self.problem.residual_torch(state, mu)
        return tangent.T @ ambient_residual

    def solve(self, mu, *, initial: np.ndarray | None = None) -> NewtonResult:
        if initial is None:
            x0 = torch.zeros(self.reduced_dim, dtype=torch.float64)
        else:
            x0 = torch.tensor(initial, dtype=torch.float64)
        return solve_nonlinear_system(
            lambda z: self.reduced_residual_torch(z, mu),
            x0,
            tol=self.solver_tol,
            max_iter=self.solver_max_iter,
            line_search_steps=self.solver_line_search_steps,
            min_step_scale=self.solver_min_step_scale,
        )


@dataclass
class ManifoldLSPGROM:
    autoencoder: SnapshotAutoencoder
    scaler: SnapshotScaler
    problem: object
    solver_tol: float = 1e-8
    solver_max_iter: int = 25
    solver_line_search_steps: int = 8
    solver_min_step_scale: float = 1e-3

    @property
    def reduced_dim(self) -> int:
        return self.autoencoder.latent_dim

    def encode(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            normalized = self.scaler.transform_tensor(torch.tensor(state, dtype=torch.float64))
            latent = self.autoencoder.encode(normalized)
        return latent.detach().cpu().numpy()

    def decode(self, latent: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            decoded = self._decode_tensor(torch.tensor(latent, dtype=torch.float64))
        return decoded.detach().cpu().numpy()

    def _decode_tensor(self, latent: torch.Tensor) -> torch.Tensor:
        normalized_state = self.autoencoder.decode(latent)
        return self.scaler.inverse_transform_tensor(normalized_state)

    def ambient_residual_torch(self, latent: torch.Tensor, mu) -> torch.Tensor:
        latent = latent.to(dtype=torch.float64)
        state = self._decode_tensor(latent)
        return self.problem.residual_torch(state, mu)

    def optimality_residual_torch(self, latent: torch.Tensor, mu) -> torch.Tensor:
        latent = latent.to(dtype=torch.float64)

        def ambient_residual(local_latent: torch.Tensor) -> torch.Tensor:
            return self.ambient_residual_torch(local_latent, mu)

        jacobian = torch.autograd.functional.jacobian(ambient_residual, latent, create_graph=True)
        residual = ambient_residual(latent)
        return jacobian.T @ residual

    def ambient_residual_norm_torch(self, latent: torch.Tensor, mu) -> torch.Tensor:
        residual = self.ambient_residual_torch(latent, mu)
        return 0.5 * torch.dot(residual, residual)

    def solve(self, mu, *, initial: np.ndarray | None = None) -> NewtonResult:
        if initial is None:
            x0 = torch.zeros(self.reduced_dim, dtype=torch.float64)
        else:
            x0 = torch.tensor(initial, dtype=torch.float64)
        return solve_nonlinear_system(
            lambda z: self.optimality_residual_torch(z, mu),
            x0,
            merit_fn=lambda z: self.ambient_residual_norm_torch(z, mu),
            tol=self.solver_tol,
            max_iter=self.solver_max_iter,
            line_search_steps=self.solver_line_search_steps,
            min_step_scale=self.solver_min_step_scale,
        )


@dataclass
class ManifoldLSPGLMROM:
    """Levenberg--Marquardt algorithm for the manifold least-squares problem.

    The objective is ``0.5 * ||R(D_phys(z); mu)||_2**2``, identical to the
    objective associated with :class:`ManifoldLSPGROM`.  This class changes
    the nonlinear algorithm: it uses a diagonally scaled, damped
    Gauss--Newton step and multiplicative damping updates.  It declares
    algorithmic convergence when the common stationarity norm
    ``||J_r(z).T @ r(z)||_2`` is below ``solver_tol``.  Equal stopping
    quantities make the recorded flags interpretable side by side, but do
    not make Newton and LM convergence behavior algorithm-independent.
    """

    autoencoder: SnapshotAutoencoder
    scaler: SnapshotScaler
    problem: object
    solver_tol: float = 1e-8
    solver_max_iter: int = 25
    solver_line_search_steps: int = 8
    solver_min_step_scale: float = 1e-3
    lm_initial_lambda: float = 1e-3
    lm_lambda_decrease: float = 0.3
    lm_lambda_increase: float = 5.0
    lm_min_lambda: float = 1e-15
    lm_max_lambda: float = 1e12
    lm_step_stagnation_rtol: float = 1e-12
    lm_merit_stagnation_rtol: float = 1e-14

    @property
    def reduced_dim(self) -> int:
        return self.autoencoder.latent_dim

    def encode(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            normalized = self.scaler.transform_tensor(torch.tensor(state, dtype=torch.float64))
            latent = self.autoencoder.encode(normalized)
        return latent.detach().cpu().numpy()

    def decode(self, latent: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            decoded = self._decode_tensor(torch.tensor(latent, dtype=torch.float64))
        return decoded.detach().cpu().numpy()

    def _decode_tensor(self, latent: torch.Tensor) -> torch.Tensor:
        normalized_state = self.autoencoder.decode(latent)
        return self.scaler.inverse_transform_tensor(normalized_state)

    def ambient_residual_torch(self, latent: torch.Tensor, mu) -> torch.Tensor:
        latent = latent.to(dtype=torch.float64)
        state = self._decode_tensor(latent)
        return self.problem.residual_torch(state, mu)

    def optimality_residual_torch(self, latent: torch.Tensor, mu) -> torch.Tensor:
        latent = latent.to(dtype=torch.float64)

        def ambient_residual(local_latent: torch.Tensor) -> torch.Tensor:
            return self.ambient_residual_torch(local_latent, mu)

        jacobian = torch.autograd.functional.jacobian(ambient_residual, latent, create_graph=True)
        residual = ambient_residual(latent)
        return jacobian.T @ residual

    def solve(self, mu, *, initial: np.ndarray | None = None) -> NewtonResult:
        if initial is None:
            z = torch.zeros(self.reduced_dim, dtype=torch.float64)
        else:
            z = torch.tensor(initial, dtype=torch.float64)
        z = z.detach().clone().to(dtype=torch.float64)

        def residual_fn(latent: torch.Tensor) -> torch.Tensor:
            return self.ambient_residual_torch(latent, mu)

        lam = float(self.lm_initial_lambda)
        history: list[float] = []
        residual = residual_fn(z).detach()
        merit = 0.5 * float(torch.dot(residual, residual).item())

        iterations_executed = 0
        for iteration in range(self.solver_max_iter):
            iterations_executed = iteration
            jacobian = torch.autograd.functional.jacobian(residual_fn, z)
            gradient = jacobian.T @ residual
            gradient_norm = torch.linalg.norm(gradient).item()
            history.append(gradient_norm)
            if gradient_norm < self.solver_tol:
                return NewtonResult(
                    solution=z.detach(),
                    residual_norm=gradient_norm,
                    iterations=iteration,
                    converged=True,
                    history=history,
                )

            gram = jacobian.T @ jacobian
            damping_diag = torch.diagonal(gram).clamp(min=1e-32)
            accepted = False
            stagnated = False
            while lam <= self.lm_max_lambda:
                lhs = gram + lam * torch.diag(damping_diag)
                try:
                    step = torch.linalg.solve(lhs, -gradient)
                except RuntimeError:
                    lam *= self.lm_lambda_increase
                    continue
                if not bool(torch.all(torch.isfinite(step))):
                    lam *= self.lm_lambda_increase
                    continue
                candidate = (z + step).detach()
                candidate_residual = residual_fn(candidate).detach()
                candidate_merit = 0.5 * float(
                    torch.dot(candidate_residual, candidate_residual).item()
                )
                if np.isfinite(candidate_merit) and candidate_merit < merit:
                    step_norm = float(torch.linalg.norm(step).item())
                    z_norm = float(torch.linalg.norm(z).item())
                    merit_drop = merit - candidate_merit
                    z = candidate
                    residual = candidate_residual
                    previous_merit = merit
                    merit = candidate_merit
                    lam = max(lam * self.lm_lambda_decrease, self.lm_min_lambda)
                    accepted = True
                    if step_norm <= self.lm_step_stagnation_rtol * (
                        z_norm + self.lm_step_stagnation_rtol
                    ):
                        stagnated = True
                    if merit_drop <= self.lm_merit_stagnation_rtol * max(
                        previous_merit, 1e-300
                    ):
                        stagnated = True
                    break
                lam *= self.lm_lambda_increase
            if not accepted:
                stagnated = True
            if stagnated:
                iterations_executed = iteration + 1
                break
        else:
            iterations_executed = self.solver_max_iter

        final_jacobian = torch.autograd.functional.jacobian(residual_fn, z)
        final_norm = torch.linalg.norm(final_jacobian.T @ residual).item()
        history.append(final_norm)
        return NewtonResult(
            solution=z.detach(),
            residual_norm=final_norm,
            iterations=iterations_executed,
            converged=final_norm < self.solver_tol,
            history=history,
        )


@dataclass
class ManifoldPreconditionedLSPGROM(ManifoldLSPGLMROM):
    """LM solve of manifold LSPG under a frozen operator-informed metric.

    The objective is ``0.5 * ||B0 @ R(D_phys(z); mu)||_2**2``.  ``B0`` is
    the fixed offline Jacobian-inverse map used in the paper; it is neither
    identified with an L2 Riesz map nor presented as an online-efficient
    preconditioner.  LM mechanics and the algorithmic stationarity stopping
    rule are inherited from :class:`ManifoldLSPGLMROM`.  Callers compute and
    report the unpreconditioned ambient residual from the returned state.
    """

    preconditioner: torch.Tensor | None = None

    def ambient_residual_torch(self, latent: torch.Tensor, mu) -> torch.Tensor:
        residual = super().ambient_residual_torch(latent, mu)
        if self.preconditioner is None:
            return residual
        return self.preconditioner.to(dtype=torch.float64) @ residual
