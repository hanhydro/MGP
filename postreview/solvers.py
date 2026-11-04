from __future__ import annotations

from dataclasses import dataclass, field

import torch

from mgp.manifold_rom import ManifoldLSPGROM
from mgp.newton import NewtonResult


@dataclass
class VariantResult(NewtonResult):
    trace: list = field(default_factory=list)
    stagnated: bool = False


def _finish(func, x, history, trace, iterations, tol, stagnated):
    final_residual = func(x)
    final_norm = torch.linalg.norm(final_residual).item()
    return VariantResult(
        solution=x.detach(),
        residual_norm=final_norm,
        iterations=iterations,
        converged=final_norm < tol,
        history=history + [final_norm],
        trace=trace,
        stagnated=stagnated,
    )


def solve_newton_instrumented(
    func,
    x0: torch.Tensor,
    *,
    merit_fn,
    tol: float,
    max_iter: int,
    line_search_steps: int,
    min_step_scale: float,
    fallback: bool = True,
) -> VariantResult:
    x = x0.detach().clone().to(dtype=torch.float64)
    history: list[float] = []
    trace: list[dict] = []

    for iteration in range(max_iter):
        with torch.enable_grad():
            x_var = x.detach().clone().requires_grad_(True)
            residual = func(x_var)
            residual_norm = torch.linalg.norm(residual).item()
            history.append(residual_norm)
            if residual_norm < tol:
                return VariantResult(
                    solution=x.detach(),
                    residual_norm=residual_norm,
                    iterations=iteration,
                    converged=True,
                    history=history,
                    trace=trace,
                )
            jacobian = torch.autograd.functional.jacobian(func, x_var)

        try:
            step = torch.linalg.solve(jacobian, -residual.detach())
        except RuntimeError:
            step = torch.linalg.lstsq(jacobian, -residual.detach()).solution

        current_merit = float(merit_fn(x).item())
        directional = float(torch.dot(residual.detach(), step).item())
        step_scale = 1.0
        accepted = False
        for _ in range(line_search_steps):
            if step_scale < min_step_scale:
                break
            candidate = x + step_scale * step
            candidate_measure = float(merit_fn(candidate).item())
            if candidate_measure < current_merit:
                x = candidate.detach()
                accepted = True
                break
            step_scale *= 0.5
        trace.append(
            {
                "iteration": iteration,
                "gradient_norm": residual_norm,
                "directional_derivative": directional,
                "accepted_scale": step_scale if accepted else None,
                "fallback": not accepted,
            }
        )
        if not accepted:
            if not fallback:
                return _finish(func, x, history, trace, iteration + 1, tol, stagnated=True)
            x = (x + min_step_scale * step).detach()

    return _finish(func, x, history, trace, max_iter, tol, stagnated=False)


def solve_gauss_newton(
    residual_fn,
    x0: torch.Tensor,
    *,
    tol: float,
    max_iter: int,
    line_search_steps: int,
    min_step_scale: float,
    fallback: bool = True,
) -> VariantResult:
    x = x0.detach().clone().to(dtype=torch.float64)
    history: list[float] = []
    trace: list[dict] = []

    def gradient_fn(point: torch.Tensor) -> torch.Tensor:
        jac = torch.autograd.functional.jacobian(residual_fn, point)
        return jac.T @ residual_fn(point).detach()

    def merit(point: torch.Tensor) -> float:
        r = residual_fn(point).detach()
        return 0.5 * float(torch.dot(r, r).item())

    for iteration in range(max_iter):
        jacobian = torch.autograd.functional.jacobian(residual_fn, x)
        residual = residual_fn(x).detach()
        gradient = jacobian.T @ residual
        gradient_norm = torch.linalg.norm(gradient).item()
        history.append(gradient_norm)
        if gradient_norm < tol:
            return VariantResult(
                solution=x.detach(),
                residual_norm=gradient_norm,
                iterations=iteration,
                converged=True,
                history=history,
                trace=trace,
            )
        gram = jacobian.T @ jacobian
        try:
            step = torch.linalg.solve(gram, -gradient)
        except RuntimeError:
            step = torch.linalg.lstsq(gram, -gradient).solution

        current_merit = 0.5 * float(torch.dot(residual, residual).item())
        directional = float(torch.dot(gradient, step).item())
        step_scale = 1.0
        accepted = False
        for _ in range(line_search_steps):
            if step_scale < min_step_scale:
                break
            candidate = x + step_scale * step
            if merit(candidate) < current_merit:
                x = candidate.detach()
                accepted = True
                break
            step_scale *= 0.5
        trace.append(
            {
                "iteration": iteration,
                "gradient_norm": gradient_norm,
                "directional_derivative": directional,
                "accepted_scale": step_scale if accepted else None,
                "fallback": not accepted,
            }
        )
        if not accepted:
            if not fallback:
                return _finish(gradient_fn, x, history, trace, iteration + 1, tol, stagnated=True)
            x = (x + min_step_scale * step).detach()

    return _finish(gradient_fn, x, history, trace, max_iter, tol, stagnated=False)


@dataclass
class ManifoldLSPGNewtonVariantROM(ManifoldLSPGROM):

    fallback: bool = True

    def solve(self, mu, *, initial=None) -> VariantResult:
        x0 = torch.zeros(self.reduced_dim, dtype=torch.float64) if initial is None else torch.tensor(initial, dtype=torch.float64)
        return solve_newton_instrumented(
            lambda z: self.optimality_residual_torch(z, mu),
            x0,
            merit_fn=lambda z: self.ambient_residual_norm_torch(z, mu),
            tol=self.solver_tol,
            max_iter=self.solver_max_iter,
            line_search_steps=self.solver_line_search_steps,
            min_step_scale=self.solver_min_step_scale,
            fallback=self.fallback,
        )


@dataclass
class ManifoldLSPGGaussNewtonROM(ManifoldLSPGROM):

    fallback: bool = True

    def solve(self, mu, *, initial=None) -> VariantResult:
        x0 = torch.zeros(self.reduced_dim, dtype=torch.float64) if initial is None else torch.tensor(initial, dtype=torch.float64)
        return solve_gauss_newton(
            lambda z: self.ambient_residual_torch(z, mu),
            x0,
            tol=self.solver_tol,
            max_iter=self.solver_max_iter,
            line_search_steps=self.solver_line_search_steps,
            min_step_scale=self.solver_min_step_scale,
            fallback=self.fallback,
        )
