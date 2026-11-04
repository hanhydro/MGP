from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mgp.newton import solve_nonlinear_system
from solvers import solve_gauss_newton, solve_newton_instrumented

torch.set_default_dtype(torch.float64)
A = torch.tensor([[3.0, 1.0], [1.0, 2.0], [0.5, -1.0]])
b = torch.tensor([1.0, -2.0, 0.5])


def residual(z):
    return A @ z - b


def stationarity(z):
    return A.T @ (A @ z - b)


def merit(z):
    r = residual(z)
    return 0.5 * torch.dot(r, r)


def test_gauss_newton_solves_linear_least_squares_in_one_step():
    res = solve_gauss_newton(residual, torch.zeros(2), tol=1e-10, max_iter=25, line_search_steps=8, min_step_scale=1e-3)
    expected = torch.linalg.lstsq(A, b).solution
    assert res.converged and res.iterations == 1
    assert torch.allclose(res.solution, expected, atol=1e-10)
    assert res.trace[0]["directional_derivative"] < 0 and res.trace[0]["accepted_scale"] == 1.0


def test_instrumented_newton_matches_reference_solver_bitwise():
    kwargs = dict(tol=1e-10, max_iter=25, line_search_steps=8, min_step_scale=1e-3)
    ref = solve_nonlinear_system(stationarity, torch.tensor([0.3, -0.7]), merit_fn=merit, **kwargs)
    new = solve_newton_instrumented(stationarity, torch.tensor([0.3, -0.7]), merit_fn=merit, **kwargs)
    assert torch.equal(ref.solution, new.solution)
    assert ref.history == new.history and ref.iterations == new.iterations and ref.converged == new.converged


def test_fallback_off_terminates_as_stagnated():
    start = torch.tensor([2.0, 2.0])

    def bowl(z):
        return 0.5 * torch.dot(z - start, z - start)

    res = solve_newton_instrumented(stationarity, start.clone(), merit_fn=bowl, tol=1e-10, max_iter=25, line_search_steps=8, min_step_scale=1e-3, fallback=False)
    assert res.stagnated and not res.converged and res.iterations == 1
    assert res.trace[0]["fallback"] is True
