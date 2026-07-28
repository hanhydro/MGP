from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .newton import NewtonResult, solve_nonlinear_system


@dataclass
class PODGalerkinROM:
    basis: np.ndarray
    mean: np.ndarray
    problem: object
    solver_tol: float = 1e-8
    solver_max_iter: int = 25
    solver_line_search_steps: int = 8
    solver_min_step_scale: float = 1e-3

    @classmethod
    def fit(
        cls,
        snapshots: np.ndarray,
        reduced_dim: int,
        problem: object,
        **solver_kwargs,
    ) -> "PODGalerkinROM":
        mean = np.mean(snapshots, axis=0)
        centered = snapshots - mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:reduced_dim].T
        return cls(basis=basis, mean=mean, problem=problem, **solver_kwargs)

    @property
    def reduced_dim(self) -> int:
        return self.basis.shape[1]

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        return self.mean + self.basis @ coefficients

    def project_state(self, state: np.ndarray) -> np.ndarray:
        return self.basis.T @ (state - self.mean)

    def reduced_residual_torch(self, coefficients: torch.Tensor, mu) -> torch.Tensor:
        basis = torch.tensor(self.basis, dtype=torch.float64, device=coefficients.device)
        mean = torch.tensor(self.mean, dtype=torch.float64, device=coefficients.device)
        state = mean + basis @ coefficients
        ambient_residual = self.problem.residual_torch(state, mu)
        return basis.T @ ambient_residual

    def solve(self, mu, *, initial: np.ndarray | None = None) -> NewtonResult:
        if initial is None:
            x0 = torch.zeros(self.reduced_dim, dtype=torch.float64)
        else:
            x0 = torch.tensor(initial, dtype=torch.float64)
        return solve_nonlinear_system(
            lambda x: self.reduced_residual_torch(x, mu),
            x0,
            tol=self.solver_tol,
            max_iter=self.solver_max_iter,
            line_search_steps=self.solver_line_search_steps,
            min_step_scale=self.solver_min_step_scale,
        )


@dataclass
class PODLSPGROM:
    basis: np.ndarray
    mean: np.ndarray
    problem: object
    solver_tol: float = 1e-8
    solver_max_iter: int = 25
    solver_line_search_steps: int = 8
    solver_min_step_scale: float = 1e-3

    @classmethod
    def fit(
        cls,
        snapshots: np.ndarray,
        reduced_dim: int,
        problem: object,
        **solver_kwargs,
    ) -> "PODLSPGROM":
        galerkin = PODGalerkinROM.fit(snapshots, reduced_dim, problem)
        return cls(basis=galerkin.basis, mean=galerkin.mean, problem=problem, **solver_kwargs)

    @property
    def reduced_dim(self) -> int:
        return self.basis.shape[1]

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        return self.mean + self.basis @ coefficients

    def project_state(self, state: np.ndarray) -> np.ndarray:
        return self.basis.T @ (state - self.mean)

    def residual_torch(self, coefficients: torch.Tensor, mu) -> torch.Tensor:
        basis = torch.tensor(self.basis, dtype=torch.float64, device=coefficients.device)
        mean = torch.tensor(self.mean, dtype=torch.float64, device=coefficients.device)
        state = mean + basis @ coefficients
        return self.problem.residual_torch(state, mu)

    def optimality_residual_torch(self, coefficients: torch.Tensor, mu) -> torch.Tensor:
        def ambient_residual(local_coefficients: torch.Tensor) -> torch.Tensor:
            return self.residual_torch(local_coefficients, mu)

        jacobian = torch.autograd.functional.jacobian(ambient_residual, coefficients, create_graph=True)
        residual = ambient_residual(coefficients)
        return jacobian.T @ residual

    def solve(self, mu, *, initial: np.ndarray | None = None) -> NewtonResult:
        if initial is None:
            x0 = torch.zeros(self.reduced_dim, dtype=torch.float64)
        else:
            x0 = torch.tensor(initial, dtype=torch.float64)
        return solve_nonlinear_system(
            lambda x: self.optimality_residual_torch(x, mu),
            x0,
            tol=self.solver_tol,
            max_iter=self.solver_max_iter,
            line_search_steps=self.solver_line_search_steps,
            min_step_scale=self.solver_min_step_scale,
        )


@dataclass
class PreconditionedPODLSPGROM(PODLSPGROM):
    """Newton POD-LSPG under the frozen operator-informed residual map.

    The objective is ``0.5 * ||B0 @ R(u_c + V @ a; mu)||_2**2``.  This arm
    is the affine companion to preconditioned manifold LSPG, but the two
    preconditioned arms use different nonlinear algorithms; they therefore
    form a partially crossed residual-metric extension rather than another
    cell of the Newton 2-by-2 trial-set/equation core.
    """

    preconditioner: np.ndarray | None = None

    def residual_torch(self, coefficients: torch.Tensor, mu) -> torch.Tensor:
        residual = super().residual_torch(coefficients, mu)
        if self.preconditioner is None:
            return residual
        b0 = torch.tensor(
            self.preconditioner, dtype=torch.float64, device=coefficients.device
        )
        return b0 @ residual
