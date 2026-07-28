from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NewtonResult:
    solution: torch.Tensor
    residual_norm: float
    iterations: int
    converged: bool
    history: list[float]


def solve_nonlinear_system(
    func,
    x0: torch.Tensor,
    *,
    merit_fn=None,
    tol: float = 1e-8,
    max_iter: int = 25,
    line_search_steps: int = 8,
    min_step_scale: float = 1e-3,
) -> NewtonResult:
    x = x0.detach().clone().to(dtype=torch.float64)
    history: list[float] = []

    for iteration in range(max_iter):
        with torch.enable_grad():
            x_var = x.detach().clone().requires_grad_(True)
            residual = func(x_var)
            residual_norm = torch.linalg.norm(residual).item()
            history.append(residual_norm)
            if residual_norm < tol:
                return NewtonResult(
                    solution=x.detach(),
                    residual_norm=residual_norm,
                    iterations=iteration,
                    converged=True,
                    history=history,
                )
            jacobian = torch.autograd.functional.jacobian(func, x_var)

        try:
            step = torch.linalg.solve(jacobian, -residual.detach())
        except RuntimeError:
            step = torch.linalg.lstsq(jacobian, -residual.detach()).solution

        current_merit = (
            float(merit_fn(x).item())
            if merit_fn is not None
            else residual_norm
        )
        step_scale = 1.0
        accepted = False
        for _ in range(line_search_steps):
            if step_scale < min_step_scale:
                break
            candidate = x + step_scale * step
            candidate_measure = (
                float(merit_fn(candidate).item())
                if merit_fn is not None
                else torch.linalg.norm(func(candidate)).item()
            )
            if candidate_measure < current_merit:
                x = candidate.detach()
                accepted = True
                break
            step_scale *= 0.5

        if not accepted:
            x = (x + min_step_scale * step).detach()

    final_residual = func(x)
    final_norm = torch.linalg.norm(final_residual).item()
    return NewtonResult(
        solution=x.detach(),
        residual_norm=final_norm,
        iterations=max_iter,
        converged=final_norm < tol,
        history=history + [final_norm],
    )
