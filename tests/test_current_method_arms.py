from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mgp import (
    FrontLayerProblem,
    ManifoldLSPGLMROM,
    ManifoldPreconditionedLSPGROM,
    NonlinearDiffusionProblem,
    PreconditionedPODLSPGROM,
    TwoDBurgersProblem,
)


class _IdentityScaler:
    def transform_tensor(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def inverse_transform_tensor(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _OneDimensionalDecoder:
    latent_dim = 1

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return state[:1]

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.stack((latent[0], latent[0]))


class _LinearResidualProblem:
    def residual_torch(self, state: torch.Tensor, _mu) -> torch.Tensor:
        return torch.stack((state[0] - 1.0, 2.0 * (state[1] + 1.0)))


def test_current_arms_are_part_of_public_package() -> None:
    assert ManifoldLSPGLMROM.__name__ == "ManifoldLSPGLMROM"
    assert ManifoldPreconditionedLSPGROM.__name__ == "ManifoldPreconditionedLSPGROM"
    assert PreconditionedPODLSPGROM.__name__ == "PreconditionedPODLSPGROM"


def test_preconditioned_pod_arm_applies_operator_map() -> None:
    problem = NonlinearDiffusionProblem(num_nodes=5)
    basis = np.eye(problem.interior_dim, 1, dtype=np.float64)
    mean = np.zeros(problem.interior_dim, dtype=np.float64)
    coefficients = torch.zeros(1, dtype=torch.float64)
    mu = np.array([1.0, 0.5, 0.15], dtype=np.float64)

    raw = PreconditionedPODLSPGROM(
        basis=basis,
        mean=mean,
        problem=problem,
        preconditioner=None,
    ).residual_torch(coefficients, mu)
    mapped = PreconditionedPODLSPGROM(
        basis=basis,
        mean=mean,
        problem=problem,
        preconditioner=2.0 * np.eye(problem.interior_dim),
    ).residual_torch(coefficients, mu)

    torch.testing.assert_close(mapped, 2.0 * raw)


def test_lm_and_preconditioned_lm_optimize_the_declared_metrics() -> None:
    common = dict(
        autoencoder=_OneDimensionalDecoder(),
        scaler=_IdentityScaler(),
        problem=_LinearResidualProblem(),
        solver_tol=1e-10,
        solver_max_iter=50,
    )
    raw = ManifoldLSPGLMROM(**common).solve(mu=None, initial=np.zeros(1))
    mapped = ManifoldPreconditionedLSPGROM(
        **common,
        preconditioner=torch.diag(torch.tensor([2.0, 0.5], dtype=torch.float64)),
    ).solve(mu=None, initial=np.zeros(1))

    assert raw.converged
    assert mapped.converged
    assert float(raw.solution[0]) == pytest.approx(-0.6, abs=1e-7)
    assert float(mapped.solution[0]) == pytest.approx(0.6, abs=1e-7)


def test_two_d_problem_and_midrange_qoi_are_available() -> None:
    problem = TwoDBurgersProblem(num_nodes=6)
    state = torch.zeros(problem.interior_dim, dtype=torch.float64)
    residual = problem.residual_torch(state, np.array([1.0, 0.5, 0.5]))
    assert residual.shape == (problem.interior_dim,)
    assert np.isfinite(problem.quantity_of_interest(state.numpy()))

    front = FrontLayerProblem(num_nodes=9, qoi_variant="midrange")
    profile = np.linspace(-1.0, 2.0, front.interior_dim)
    value = front.quantity_of_interest(profile)
    assert 0.0 <= value <= 1.0


def test_acceptance_statistics_rederive_declared_families(tmp_path: Path) -> None:
    package_root = Path(__file__).resolve().parents[1]
    output_json = tmp_path / "acceptance_statistics.json"
    output_markdown = tmp_path / "acceptance_statistics.md"
    subprocess.run(
        [
            sys.executable,
            str(package_root / "scripts" / "derive_acceptance_revision_statistics.py"),
            "--input",
            str(
                package_root
                / "artifacts"
                / "acceptance_revision_retrospective_inputs.json"
            ),
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
        ],
        check=True,
        cwd=package_root,
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["m1_holm_family"]["configured_contrasts"] == 20
    assert result["m1_holm_family"]["below_0p05_count"] == 9
    assert result["m4_metric_full_cohort"]["manifold_sample_left_lower"] == 1199
    assert result["m4_metric_full_cohort"]["affine_sample_left_lower"] == 1187
    assert (
        sum(
            row["interaction_seed_summary"]["holm_p_10"] < 0.05
            for row in result["factorial_core"]["cells"]
        )
        == 4
    )


def test_prospective_public_export_reproduces_first_and_post_look(tmp_path: Path) -> None:
    package_root = Path(__file__).resolve().parents[1]
    study = package_root / "eval" / "prospective_weight_validation_2026-07-17"
    reanalysis_json = tmp_path / "prospective_reanalysis.json"
    reanalysis_markdown = tmp_path / "prospective_reanalysis.md"
    subprocess.run(
        [
            sys.executable,
            str(study / "analyze_public_results.py"),
            "--manifest",
            str(study / "manifest.tsv"),
            "--public-export",
            str(study / "public_export"),
            "--canonical-summary",
            str(study / "first_look_output" / "FIRST_LOOK_SUMMARY.json"),
            "--output-json",
            str(reanalysis_json),
            "--output-markdown",
            str(reanalysis_markdown),
        ],
        check=True,
        cwd=study,
    )
    first = json.loads(reanalysis_json.read_text(encoding="utf-8"))
    assert len(first["primary_holm5"]) == 5
    assert all(row["favorable_negative"] == 20 for row in first["primary_holm5"])
    assert sum(row["holm_adjusted_p"] < 0.05 for row in first["secondary_cellwise_holm40"]) == 38

    descriptive_json = tmp_path / "prospective_descriptives.json"
    subprocess.run(
        [
            sys.executable,
            str(study / "analyze_public_descriptives.py"),
            "--manifest",
            str(study / "manifest.tsv"),
            "--public-export",
            str(study / "public_export"),
            "--canonical-descriptives",
            str(study / "first_look_output" / "POST_LOOK_DESCRIPTIVES_V1.json"),
            "--output-json",
            str(descriptive_json),
        ],
        check=True,
        cwd=study,
    )
    descriptive = json.loads(descriptive_json.read_text(encoding="utf-8"))
    assert descriptive["design"]["arm_sample_record_count"] == 13440
    assert sum(
        row["catastrophic_failure_count"]
        for row in descriptive["reliability_by_cell_arm"]
    ) == 0


def test_prospective_frozen_scientific_source_inventory() -> None:
    package_root = Path(__file__).resolve().parents[1]
    verifier = (
        package_root
        / "eval"
        / "prospective_weight_validation_2026-07-17"
        / "verify_frozen_scientific_source.py"
    )
    subprocess.run([sys.executable, str(verifier)], check=True, cwd=package_root)


def test_prospective_latex_tables_rederive_byte_identically(tmp_path: Path) -> None:
    package_root = Path(__file__).resolve().parents[1]
    tracked = package_root / "artifacts" / "prospective"
    subprocess.run(
        [
            sys.executable,
            str(package_root / "scripts" / "render_prospective_first_look_tables.py"),
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        cwd=package_root,
    )
    expected_names = {
        "prospective_primary_table.tex",
        "prospective_cellwise_table.tex",
        "prospective_secondary_table.tex",
        "prospective_descriptive_effects_table.tex",
        "prospective_reliability_table.tex",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_names
    for name in expected_names:
        assert (tmp_path / name).read_bytes() == (tracked / name).read_bytes()
