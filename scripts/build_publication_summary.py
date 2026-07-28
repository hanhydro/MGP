from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
sys.path.insert(0, str(ROOT / "src"))

from mgp.solver_tradeoff import pick_solver_recommendation


BENCHMARK_LABELS = {
    "reference_bratu_source": "Bratu-source problem",
    "reference_front_layer": "front-layer problem",
    "reference_hydrologic_conductivity": "hydraulic-conductivity test",
    "reference_nonlinear_diffusion": "nonlinear-diffusion problem",
}

PROBLEM_LABELS = {
    "bratu_source": "Bratu-source",
    "front_layer": "front-layer",
    "hydrologic_conductivity": "hydraulic-conductivity",
    "nonlinear_diffusion": "nonlinear-diffusion",
}

REGIME_LABELS = {
    "in_domain": "in-domain",
    "ood": "out-of-domain",
}

SOLVER_DECISION_LABELS = {
    "retain_manifold_galerkin": "MG selected",
    "prefer_manifold_lspg": "M-LSPG selected",
}


def _benchmark_label(value: str) -> str:
    return BENCHMARK_LABELS.get(value, value.replace("_", " "))


def _problem_label(value: str) -> str:
    return PROBLEM_LABELS.get(value, value.replace("_", "-"))


def _regime_label(value: str) -> str:
    return REGIME_LABELS.get(value, value.replace("_", "-"))


def _solver_decision_label(value: str) -> str:
    return SOLVER_DECISION_LABELS.get(value, value.replace("_", " "))


def _display_row(row: dict) -> dict:
    return {
        **row,
        "benchmark_label": _benchmark_label(row.get("preset_name", "")),
        "problem_label": _problem_label(row.get("problem_name", "")),
        "regime_label": _regime_label(row.get("test_regime", "")),
        "solver_decision_label": _solver_decision_label(
            row.get("solver_recommendation", row.get("recommended_solver", ""))
        ),
    }


def build_publication_summary_artifacts(
    artifacts_dir: Path = ARTIFACTS,
    *,
    reference_path: Path | None = None,
    solver_path: Path | None = None,
    preset_tuning_path: Path | None = None,
    objective_weight_path: Path | None = None,
    objective_ablation_path: Path | None = None,
    summary_stem: str = "publication_summary",
    csv_prefix: str = "publication",
) -> dict:
    hydrologic_only = _is_hydrologic_summary(summary_stem=summary_stem, solver_path=solver_path)
    reference = _load_json(reference_path or (artifacts_dir / "reference_architecture_study.json"))
    solver = _load_json(solver_path or (artifacts_dir / "solver_tradeoff_study.json"))
    preset_tuning = _load_json(preset_tuning_path or (artifacts_dir / "preset_tuning_bratu.json"))
    objective_weight = _load_json(objective_weight_path or (artifacts_dir / "objective_weight_study.json"))
    objective_ablation = _load_json(
        objective_ablation_path or (artifacts_dir / "objective_ablation_study.json")
    )

    payload = {
        "solver_table": _build_solver_table(solver),
        "seedwise_solver_rows": solver.get("seedwise_solver_rows", []),
        "solver_win_count_table": _build_solver_win_count_table(solver),
        "solver_failure_table": _build_solver_failure_table(solver),
        "solver_time_table": _build_solver_time_table(solver),
        "convergence_context_table": _build_convergence_context_table(solver),
        "solver_threshold_sensitivity_table": _build_solver_threshold_sensitivity_table(solver),
    }
    provenance_payload = None
    if hydrologic_only:
        payload["narrative"] = _build_hydrologic_narrative(solver)
    else:
        provenance_payload = {
            "bratu_resolution": _build_bratu_resolution(preset_tuning),
            "objective_table": _build_objective_table(objective_weight),
            "objective_ablation_table": _build_objective_ablation_table(objective_ablation),
            "narrative": _build_narrative(
                reference, solver, preset_tuning, objective_weight, objective_ablation
            ),
        }
        payload["narrative"] = _build_final_summary_narrative(solver)
        (artifacts_dir / f"{summary_stem}_provenance.json").write_text(json.dumps(provenance_payload, indent=2))
        (artifacts_dir / f"{summary_stem}_provenance.md").write_text(_format_provenance_markdown(provenance_payload))

    (artifacts_dir / f"{summary_stem}.json").write_text(json.dumps(payload, indent=2))
    (artifacts_dir / f"{summary_stem}.md").write_text(_format_markdown(payload))
    _write_publication_csvs(payload, artifacts_dir=artifacts_dir, csv_prefix=csv_prefix)
    return payload


def build_final_results_table_artifacts(artifacts_dir: Path = ARTIFACTS) -> dict:
    expanded = _load_json(artifacts_dir / "publication_summary_expanded.json")
    hydrologic = _load_json(artifacts_dir / "publication_summary_hydrologic.json")
    rows = [
        _final_results_row(row)
        for row in [*expanded["solver_table"], *hydrologic["solver_table"]]
    ]
    rows.sort(key=lambda row: (row["problem_name"], row["test_regime"], row["preset_name"]))

    markdown_lines = [
        "# Final Results Table",
        "",
        "These benchmark configurations reproduce the three-problem and hydraulic-conductivity settings. The targeted residual check is counted separately and is not substituted into this table. In particular, the front-layer in-domain configuration below remains at the Table 2 weights `(0, 0.03)` rather than the later alternative weights `(0.02, 0.03)`. The summary figures likewise use the Table 2 values rather than the alternative weights.",
        "",
        "Manifold Galerkin = tangent-space residual projection equation on the learned manifold. Manifold-LSPG = stationarity solve for the full-order-residual least-squares problem on the same learned manifold.",
        "",
        "For the front-layer out-of-domain configuration, the front-location QoI is tied at the displayed precision for manifold Galerkin and POD-LSPG under the fallback front-location rule, so the implemented surrogate has limited discriminatory value for that configuration.",
        "",
        "The Selection column applies only to the same-decoder manifold-solver rule; it is not a global recommendation over POD-LSPG.",
        "",
        "Gap columns are computed from unrounded seed-level aggregates, so they can differ from differences of the rounded printed means at the last displayed digit.",
        "",
        "| Case | Evaluation regime | Selection | Manifold Galerkin error | Manifold Galerkin QoI | Manifold Galerkin residual | POD-LSPG error | POD-LSPG QoI | POD-LSPG residual | Manifold Galerkin lower-error seeds | Manifold Galerkin lower-QoI-error seeds | Manifold Galerkin failures | Manifold-LSPG failures |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        display = _display_row(row)
        markdown_lines.append(
            "| {benchmark_label} | {regime_label} | {solver_decision_label} | {mgp_mean_error_l2:.4f} | {mgp_mean_qoi_error:.4f} | {mgp_mean_residual:.4f} | {lspg_mean_error_l2:.4f} | {lspg_mean_qoi_error:.4f} | {lspg_mean_residual:.4f} | {mgp_error_wins_vs_lspg}/{num_runs} | {mgp_qoi_wins_vs_lspg}/{num_runs} | {mgp_failed_case_count}/{total_case_count} | {mgp_lspg_failed_case_count}/{total_case_count} |".format(
                **display
            )
        )

    (artifacts_dir / "FINAL_RESULTS_TABLE.md").write_text("\n".join(markdown_lines) + "\n")
    _write_csv(
        artifacts_dir / "final_results_table.csv",
        rows,
        [
            "preset_name",
            "problem_name",
            "test_regime",
            "recommended_solver",
            "mgp_mean_error_l2",
            "mgp_mean_qoi_error",
            "mgp_mean_residual",
            "lspg_mean_error_l2",
            "lspg_mean_qoi_error",
            "lspg_mean_residual",
            "mgp_error_wins_vs_lspg",
            "mgp_qoi_wins_vs_lspg",
            "num_runs",
            "mgp_failed_case_count",
            "mgp_lspg_failed_case_count",
            "total_case_count",
        ],
    )
    return {"rows": rows}


def build_highlighted_row_uncertainty_artifacts(artifacts_dir: Path = ARTIFACTS) -> dict:
    expanded = _load_json(artifacts_dir / "publication_summary_expanded.json")
    hydrologic = _load_json(artifacts_dir / "publication_summary_hydrologic.json")
    summary_rows = [*expanded["solver_table"], *hydrologic["solver_table"]]
    seedwise_rows = _load_seedwise_rows(artifacts_dir / "publication_expanded_seedwise_solver_rows.csv") + _load_seedwise_rows(
        artifacts_dir / "publication_hydrologic_seedwise_solver_rows.csv"
    )
    targets = [
        ("reference_front_layer", "in_domain"),
        ("reference_front_layer", "ood"),
        ("reference_bratu_source", "ood"),
        ("reference_hydrologic_conductivity", "in_domain"),
        ("reference_hydrologic_conductivity", "ood"),
    ]
    rows = []
    for preset_name, test_regime in targets:
        summary_row = next(
            row
            for row in summary_rows
            if row["preset_name"] == preset_name and row["test_regime"] == test_regime
        )
        seedwise = [
            row
            for row in seedwise_rows
            if row["preset_name"] == preset_name
            and row["test_regime"] == test_regime
            and row["solver_variant"] == "manifold_galerkin"
        ]
        error_gap_values = [row["error_gap_vs_lspg"] for row in seedwise]
        qoi_gap_values = [row["qoi_gap_vs_lspg"] for row in seedwise]
        residual_gap_values = [row["residual_gap_vs_lspg"] for row in seedwise]
        rows.append(
            {
                "preset_name": preset_name,
                "problem_name": summary_row["problem_name"],
                "test_regime": test_regime,
                "qoi_name": summary_row["qoi_name"],
                "mean_error_gap_vs_pod_lspg": summary_row["mgp_error_gap_vs_lspg"],
                "error_gap_ci95_low": _percentile(error_gap_values, 2.5),
                "error_gap_ci95_high": _percentile(error_gap_values, 97.5),
                "mean_qoi_gap_vs_pod_lspg": summary_row["mgp_qoi_gap_vs_lspg"],
                "qoi_gap_ci95_low": _percentile(qoi_gap_values, 2.5),
                "qoi_gap_ci95_high": _percentile(qoi_gap_values, 97.5),
                "mean_residual_gap_vs_pod_lspg": summary_row["mgp_residual_gap_vs_lspg"],
                "residual_gap_ci95_low": _percentile(residual_gap_values, 2.5),
                "residual_gap_ci95_high": _percentile(residual_gap_values, 97.5),
                "mgp_error_wins_vs_lspg": summary_row["mgp_error_wins_vs_lspg"],
                "mgp_qoi_wins_vs_lspg": summary_row["mgp_qoi_wins_vs_lspg"],
                "num_runs": summary_row["num_runs"],
                "mgp_failed_case_count": summary_row["mgp_failed_case_count"],
                "mgp_lspg_failed_case_count": summary_row["mgp_lspg_failed_case_count"],
                "total_case_count": summary_row["total_case_count"],
            }
        )

    markdown_lines = [
        "# Supplementary Table S1. Highlighted Case Uncertainty",
        "",
        "The gap sign convention is method minus POD-LSPG on the same benchmark configuration, so negative error/QoI gaps mean lower error and positive residual gaps mean higher residual.",
        "",
        "Intervals are empirical 2.5--97.5% seed-percentile intervals over seed-level means, not inferential confidence intervals. Gap columns are computed from unrounded seed-level aggregates, so they can differ from differences of the rounded printed means at the last displayed digit.",
        "",
        "For the front-layer out-of-domain configuration, the front-location QoI is tied at the displayed precision for manifold Galerkin and POD-LSPG under the fallback front-location rule, so the implemented surrogate has limited discriminatory value for that configuration.",
        "",
        "| Case | Evaluation regime | QoI | mean error gap | error-gap 2.5--97.5% seed pct. | mean QoI gap | QoI-gap 2.5--97.5% seed pct. | mean residual gap | residual-gap 2.5--97.5% seed pct. | Manifold Galerkin lower-error seeds | Manifold Galerkin lower-QoI-error seeds | Manifold Galerkin failures | Manifold-LSPG failures |",
        "| --- | --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        display = _display_row(row)
        markdown_lines.append(
            "| {benchmark_label} | {regime_label} | {qoi_name} | {mean_error_gap_vs_pod_lspg:.4f} | [{error_gap_ci95_low:.4f}, {error_gap_ci95_high:.4f}] | {mean_qoi_gap_vs_pod_lspg:.4f} | [{qoi_gap_ci95_low:.4f}, {qoi_gap_ci95_high:.4f}] | {mean_residual_gap_vs_pod_lspg:.4f} | [{residual_gap_ci95_low:.4f}, {residual_gap_ci95_high:.4f}] | {mgp_error_wins_vs_lspg}/{num_runs} | {mgp_qoi_wins_vs_lspg}/{num_runs} | {mgp_failed_case_count}/{total_case_count} | {mgp_lspg_failed_case_count}/{total_case_count} |".format(
                **display
            )
        )
    (artifacts_dir / "HIGHLIGHTED_ROW_UNCERTAINTY.md").write_text("\n".join(markdown_lines) + "\n")
    _write_csv(
        artifacts_dir / "highlighted_row_uncertainty.csv",
        rows,
        [
            "preset_name",
            "problem_name",
            "test_regime",
            "qoi_name",
            "mean_error_gap_vs_pod_lspg",
            "error_gap_ci95_low",
            "error_gap_ci95_high",
            "mean_qoi_gap_vs_pod_lspg",
            "qoi_gap_ci95_low",
            "qoi_gap_ci95_high",
            "mean_residual_gap_vs_pod_lspg",
            "residual_gap_ci95_low",
            "residual_gap_ci95_high",
            "mgp_error_wins_vs_lspg",
            "mgp_qoi_wins_vs_lspg",
            "num_runs",
            "mgp_failed_case_count",
            "mgp_lspg_failed_case_count",
            "total_case_count",
        ],
    )
    return {"rows": rows}


def build_results_traceability_map_artifacts(artifacts_dir: Path = ARTIFACTS) -> dict:
    rows = [
        {
            "result_id": "front_layer_in_domain_state_reduction",
            "result_text": "Front-layer in-domain has the largest expanded state-error reduction for manifold Galerkin.",
            "manuscript_location": "Results §2",
            "final_results_rows": "front-layer / in-domain",
            "uncertainty_rows": "front-layer / in-domain",
            "solver_delta_rows": "front-layer / in-domain",
            "figures": "Figure 1; Figure 2",
        },
        {
            "result_id": "front_layer_ood_mixed",
            "result_text": "Front-layer out-of-domain has mixed error and QoI behavior rather than a uniformly lower-error result.",
            "manuscript_location": "Results §2; uncertainty paragraph",
            "final_results_rows": "front-layer / out-of-domain",
            "uncertainty_rows": "front-layer / out-of-domain",
            "solver_delta_rows": "front-layer / out-of-domain",
            "figures": "Figure 1; Figure 2",
        },
        {
            "result_id": "bratu_ood_qoi_reduction",
            "result_text": "Bratu-source out-of-domain has the largest QoI-error reduction, with only a modest mean state-error reduction.",
            "manuscript_location": "Results §3",
            "final_results_rows": "Bratu-source / out-of-domain",
            "uncertainty_rows": "Bratu-source / out-of-domain",
            "solver_delta_rows": "Bratu-source / out-of-domain",
            "figures": "Figure 1; Figure 2",
        },
        {
            "result_id": "hydraulic_in_domain_error_reduction",
            "result_text": "Hydraulic conductivity in-domain lowers mean state and mean outlet-flux proxy error while remaining residual-inferior to POD-LSPG.",
            "manuscript_location": "Results §5",
            "final_results_rows": "hydraulic-conductivity / in-domain",
            "uncertainty_rows": "hydraulic-conductivity / in-domain",
            "solver_delta_rows": "hydraulic-conductivity / in-domain",
            "figures": "Figure 1; Figure 3",
        },
        {
            "result_id": "hydraulic_ood_error_reduction",
            "result_text": "Hydraulic conductivity out-of-domain lowers mean state and mean outlet-flux proxy error while remaining residual-inferior to POD-LSPG.",
            "manuscript_location": "Results §5",
            "final_results_rows": "hydraulic-conductivity / out-of-domain",
            "uncertainty_rows": "hydraulic-conductivity / out-of-domain",
            "solver_delta_rows": "hydraulic-conductivity / out-of-domain",
            "figures": "Figure 1; Figure 3",
        },
        {
            "result_id": "dimension_check",
            "result_text": "A POD-dimension check shows that selected configurations retain lower error at larger POD dimensions, including front-layer in-domain and hydraulic-conductivity out-of-domain on QoI, but the effect is not uniform and the residual gap remains.",
            "manuscript_location": "Introduction; Results §1; Discussion",
            "final_results_rows": "front-layer in-domain; front-layer out-of-domain; Bratu-source out-of-domain; hydraulic-conductivity in-domain; hydraulic-conductivity out-of-domain",
            "uncertainty_rows": "not primary",
            "solver_delta_rows": "not primary",
            "figures": "Supplementary Table S3",
        },
        {
            "result_id": "residual_aligned_check",
            "result_text": "A residual-weight check selects front-layer out-of-domain and Bratu-source out-of-domain settings, but only front-layer out-of-domain preserves a state advantage through POD dimension 8 and neither configuration eliminates the residual gap.",
            "manuscript_location": "Results §1; Results §2; Results §3; Discussion",
            "final_results_rows": "not primary: front-layer out-of-domain; Bratu-source out-of-domain",
            "uncertainty_rows": "not primary",
            "solver_delta_rows": "not primary",
            "figures": "Supplementary Table S4",
        },
        {
            "result_id": "full_order_residual_baseline",
            "result_text": "POD-LSPG remains the lower-residual baseline across the benchmark configurations.",
            "manuscript_location": "Abstract; Results §1; Results §8",
            "final_results_rows": "all configurations",
            "uncertainty_rows": "all highlighted configurations",
            "solver_delta_rows": "not primary",
            "figures": "Figure 1; Figure 2; Figure 3",
        },
        {
            "result_id": "same_manifold_solver_effect",
            "result_text": "Solver choice on the same learned manifold changes error, residual, runtime, and failures.",
            "manuscript_location": "Introduction; Results §1",
            "final_results_rows": "solver-selection context across all configurations",
            "uncertainty_rows": "not primary",
            "solver_delta_rows": "all configurations",
            "figures": "Figure 1",
        },
    ]

    markdown_lines = [
        "# Results Traceability Map",
        "",
        "| result | manuscript location | final-results entries | uncertainty entries | same-manifold delta entries | figures |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        markdown_lines.append(
            "| {result_text} | {manuscript_location} | {final_results_rows} | {uncertainty_rows} | {solver_delta_rows} | {figures} |".format(
                **row
            )
        )

    (artifacts_dir / "RESULTS_TRACEABILITY_MAP.md").write_text("\n".join(markdown_lines) + "\n")
    _write_csv(
        artifacts_dir / "results_traceability_map.csv",
        rows,
        [
            "result_id",
            "result_text",
            "manuscript_location",
            "final_results_rows",
            "uncertainty_rows",
            "solver_delta_rows",
            "figures",
        ],
    )
    return {"rows": rows}


def main() -> None:
    build_publication_summary_artifacts()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_seedwise_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    parsed = []
    for row in rows:
        parsed.append(
            {
                **row,
                "error_gap_vs_lspg": float(row["error_gap_vs_lspg"]),
                "residual_gap_vs_lspg": float(row["residual_gap_vs_lspg"]),
                "qoi_gap_vs_lspg": float(row["qoi_gap_vs_lspg"]),
            }
        )
    return parsed


def _is_hydrologic_summary(*, summary_stem: str, solver_path: Path | None) -> bool:
    tokens = [summary_stem.lower()]
    if solver_path is not None:
        tokens.append(str(solver_path).lower())
    return any("hydrologic" in token for token in tokens)


def _build_solver_table(solver: dict) -> list[dict]:
    return list(solver["aggregated"])


def _build_bratu_resolution(preset_tuning: dict) -> dict:
    recommendations = preset_tuning["recommendations"]
    promote_r3 = all(
        row["recommended_label"].startswith("mlp:r=3:")
        and row["error_improvement"] > 0.0
        and row["residual_change"] < 0.0
        for row in recommendations
    )
    return {
        "decision": "promote_r3" if promote_r3 else "retain_r2",
        "recommendations": recommendations,
    }


def _build_objective_table(objective_weight: dict) -> list[dict]:
    recommendations_by_preset = {
        row["preset_name"]: row for row in objective_weight["recommendations"]
    }
    rows = []
    for row in objective_weight["aggregated"]:
        recommendation = recommendations_by_preset[row["preset_name"]]
        if not recommendation["promoted"]:
            continue
        if row["config_id"] != recommendation["recommended_config_id"]:
            continue
        rows.append(
            {
                "preset_name": row["preset_name"],
                "problem_name": row["problem_name"],
                "test_regime": row["test_regime"],
                "recommended_config_id": recommendation["recommended_config_id"],
                "recommended_solver": row["recommended_solver"],
                "projected_residual_penalty_weight": row["projected_residual_penalty_weight"],
                "ambient_residual_penalty_weight": row["ambient_residual_penalty_weight"],
                "recommended_mean_error_l2": row["recommended_mean_error_l2"],
                "recommended_mean_residual": row["recommended_mean_residual"],
                "reconstruction_mean_projected_residual": row["reconstruction_mean_projected_residual"],
                "recommended_mean_qoi_error": row["recommended_mean_qoi_error"],
                "recommended_error_gap_vs_lspg": row["recommended_error_gap_vs_lspg"],
                "recommended_residual_gap_vs_lspg": row["recommended_residual_gap_vs_lspg"],
            }
        )
    return rows


def _build_objective_ablation_table(objective_ablation: dict) -> list[dict]:
    return list(objective_ablation["recommendations"])


def _build_solver_win_count_table(solver: dict) -> list[dict]:
    rows = []
    for row in solver["aggregated"]:
        rows.append(
            {
                "preset_name": row["preset_name"],
                "problem_name": row["problem_name"],
                "test_regime": row["test_regime"],
                "mgp_error_wins_vs_lspg": row["mgp_error_wins_vs_lspg"],
                "mgp_qoi_wins_vs_lspg": row["mgp_qoi_wins_vs_lspg"],
                "mgp_lspg_error_wins_vs_lspg": row["mgp_lspg_error_wins_vs_lspg"],
                "mgp_lspg_qoi_wins_vs_lspg": row["mgp_lspg_qoi_wins_vs_lspg"],
                "num_runs": row["num_runs"],
            }
        )
    return rows


def _build_solver_failure_table(solver: dict) -> list[dict]:
    rows = []
    for row in solver["aggregated"]:
        rows.append(
            {
                "preset_name": row["preset_name"],
                "problem_name": row["problem_name"],
                "test_regime": row["test_regime"],
                "total_case_count": row["total_case_count"],
                "mgp_failed_case_count": row["mgp_failed_case_count"],
                "mgp_lspg_failed_case_count": row["mgp_lspg_failed_case_count"],
                "pod_failed_case_count": row["pod_failed_case_count"],
                "lspg_failed_case_count": row["lspg_failed_case_count"],
                "mgp_failed_seed_count": row["mgp_failed_seed_count"],
                "mgp_lspg_failed_seed_count": row["mgp_lspg_failed_seed_count"],
                "pod_failed_seed_count": row["pod_failed_seed_count"],
                "lspg_failed_seed_count": row["lspg_failed_seed_count"],
            }
        )
    return rows


def _build_solver_time_table(solver: dict) -> list[dict]:
    rows = []
    for row in solver["aggregated"]:
        rows.append(
            {
                "preset_name": row["preset_name"],
                "problem_name": row["problem_name"],
                "test_regime": row["test_regime"],
                "mean_full_time_sec": row["mean_full_time_sec"],
                "mean_mgp_time_sec": row["mean_mgp_time_sec"],
                "mean_mgp_lspg_time_sec": row["mean_mgp_lspg_time_sec"],
                "mean_pod_time_sec": row["mean_pod_time_sec"],
                "mean_lspg_time_sec": row["mean_lspg_time_sec"],
                "mgp_speedup_vs_full": row["mgp_speedup_vs_full"],
                "mgp_lspg_speedup_vs_full": row["mgp_lspg_speedup_vs_full"],
            }
        )
    return rows


def _build_convergence_context_table(solver: dict) -> list[dict]:
    rows = []
    for row in solver["aggregated"]:
        rows.append(
            {
                "preset_name": row["preset_name"],
                "problem_name": row["problem_name"],
                "test_regime": row["test_regime"],
                "qoi_name": row["qoi_name"],
                "num_runs": row["num_runs"],
                "total_case_count": row["total_case_count"],
                "mgp_convergence_rate": row["mgp_convergence_rate"],
                "mgp_failed_case_count": row["mgp_failed_case_count"],
                "mgp_mean_error_l2": row["mgp_mean_error_l2"],
                "mgp_mean_qoi_error": row["mgp_mean_qoi_error"],
                "mgp_mean_residual": row["mgp_mean_residual"],
                "mgp_lspg_convergence_rate": row["mgp_lspg_convergence_rate"],
                "mgp_lspg_failed_case_count": row["mgp_lspg_failed_case_count"],
                "mgp_lspg_mean_error_l2": row["mgp_lspg_mean_error_l2"],
                "mgp_lspg_mean_qoi_error": row["mgp_lspg_mean_qoi_error"],
                "mgp_lspg_mean_residual": row["mgp_lspg_mean_residual"],
                "lspg_convergence_rate": 1.0
                - (row["lspg_failed_case_count"] / max(row["total_case_count"], 1)),
                "lspg_failed_case_count": row["lspg_failed_case_count"],
                "lspg_mean_error_l2": row["lspg_mean_error_l2"],
                "lspg_mean_qoi_error": row["lspg_mean_qoi_error"],
                "lspg_mean_residual": row["lspg_mean_residual"],
            }
        )
    return rows


def _build_solver_threshold_sensitivity_table(solver: dict) -> list[dict]:
    thresholds = (0.0, 0.01, 0.02, 0.05)
    rows = []
    for row in solver["aggregated"]:
        base = {
            "mgp_error_gap_vs_lspg": row["mgp_error_gap_vs_lspg"],
            "mgp_residual_gap_vs_lspg": row["mgp_residual_gap_vs_lspg"],
            "mgp_lspg_error_gap_vs_lspg": row["mgp_lspg_error_gap_vs_lspg"],
            "mgp_lspg_residual_gap_vs_lspg": row["mgp_lspg_residual_gap_vs_lspg"],
        }
        sensitivity = {
            f"recommendation_at_{_threshold_label(threshold)}": pick_solver_recommendation(
                base, error_gap_tolerance=threshold
            )
            for threshold in thresholds
        }
        rows.append(
            {
                "preset_name": row["preset_name"],
                "problem_name": row["problem_name"],
                "test_regime": row["test_regime"],
                "baseline_recommendation": row["solver_recommendation"],
                **sensitivity,
            }
        )
    return rows


def _build_narrative(
    reference: dict, solver: dict, preset_tuning: dict, objective_weight: dict, objective_ablation: dict
) -> dict:
    promoted_objective_configs = {row["recommended_config_id"] for row in objective_weight["recommendations"] if row["promoted"]}
    ablation_modes = {row["recommended_mode"] for row in objective_ablation["recommendations"]}
    solver_preferences = {row["solver_recommendation"] for row in solver["aggregated"]}
    return {
        "architecture_rule": "pareto_unresolved"
        if any(row["status"] == "unresolved" for row in reference["decisions"])
        else "fully_resolved",
        "bratu_latent_dim_rule": _build_bratu_resolution(preset_tuning)["decision"],
        "objective_rule": "weights vary by problem family" if len(promoted_objective_configs) > 1 else "uniform weights",
        "solver_rule": "mixed solver selections"
        if len(solver_preferences) > 1
        else _solver_decision_label(next(iter(solver_preferences), "retain_manifold_galerkin")),
        "code_default_loss_note": "the training objective supports a mixed tangent-test-plus-ambient loss family",
        "confirmed_objective_note": (
            "the objective weights vary by problem family and should be reported per test problem"
            if len(promoted_objective_configs) > 1
            else "the objective weights are uniform across test problems"
        ),
        "ablation_note": (
            "objective ablations vary by problem family across ambient_only, projected_only, and projected_plus_ambient"
            if len(ablation_modes) > 1
            else f"objective ablation recommendation is {next(iter(ablation_modes))}"
        ),
    }


def _build_hydrologic_narrative(solver: dict) -> dict:
    solver_preferences = {row["solver_recommendation"] for row in solver["aggregated"]}
    return {
        "study_scope": "hydrologic-only",
        "solver_rule": "mixed solver selections"
        if len(solver_preferences) > 1
        else _solver_decision_label(next(iter(solver_preferences), "retain_manifold_galerkin")),
        "objective_provenance_note": (
            "the hydraulic-conductivity test used the fixed hydraulic-conductivity settings reported in the experiment-settings appendix rather than the earlier family-specific objective-weight analysis"
        ),
        "hydrologic_scope_note": (
            "the hydraulic-conductivity test is a separate completed deterministic run and is summarized without unrelated Bratu or three-configuration objective-tuning sections"
        ),
    }


def _build_final_summary_narrative(solver: dict) -> dict:
    solver_preferences = {row["solver_recommendation"] for row in solver["aggregated"]}
    return {
        "study_scope": "expanded configurations",
        "solver_rule": "mixed solver selections"
        if len(solver_preferences) > 1
        else _solver_decision_label(next(iter(solver_preferences), "retain_manifold_galerkin")),
        "summary_scope_note": (
            "this summary reports the expanded solver comparisons only; tuning and selection provenance are written separately"
        ),
    }


def _final_results_row(row: dict) -> dict:
    return {
        "preset_name": row["preset_name"],
        "problem_name": row["problem_name"],
        "test_regime": row["test_regime"],
        "recommended_solver": row["solver_recommendation"],
        "mgp_mean_error_l2": row["mgp_mean_error_l2"],
        "mgp_mean_qoi_error": row["mgp_mean_qoi_error"],
        "mgp_mean_residual": row["mgp_mean_residual"],
        "lspg_mean_error_l2": row["lspg_mean_error_l2"],
        "lspg_mean_qoi_error": row["lspg_mean_qoi_error"],
        "lspg_mean_residual": row["lspg_mean_residual"],
        "mgp_error_wins_vs_lspg": row["mgp_error_wins_vs_lspg"],
        "mgp_qoi_wins_vs_lspg": row["mgp_qoi_wins_vs_lspg"],
        "num_runs": row["num_runs"],
        "mgp_failed_case_count": row["mgp_failed_case_count"],
        "mgp_lspg_failed_case_count": row["mgp_lspg_failed_case_count"],
        "total_case_count": row["total_case_count"],
    }


def _format_markdown(payload: dict) -> str:
    lines = [
        "# Reported Summary",
        "",
        "## Solver Tradeoff Table",
        "",
        "Manifold Galerkin = tangent-space residual projection equation on the learned manifold. Manifold-LSPG = stationarity solve for the full-order-residual least-squares problem on the same learned manifold.",
        "",
        "| Case | Evaluation regime | architecture | solver decision | Manifold Galerkin error | Manifold Galerkin error 2.5--97.5% seed pct. | Manifold Galerkin residual | Manifold Galerkin residual 2.5--97.5% seed pct. | Manifold-LSPG error | Manifold-LSPG residual | Manifold Galerkin QoI | Manifold-LSPG QoI | Manifold Galerkin speedup | Manifold-LSPG speedup | Manifold Galerkin mean time (s) | Manifold-LSPG mean time (s) | POD-LSPG mean time (s) | Manifold Galerkin error gap vs POD-LSPG | Manifold Galerkin residual gap vs POD-LSPG | Manifold Galerkin lower-error seeds | Manifold Galerkin lower-QoI-error seeds | Manifold-LSPG failures |",
        "| --- | --- | --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["solver_table"]:
        display = _display_row(row)
        lines.append(
            "| {benchmark_label} | {regime_label} | {architecture_name} | {solver_decision_label} | {mgp_mean_error_l2:.4f} | [{mgp_error_ci95_low:.4f}, {mgp_error_ci95_high:.4f}] | {mgp_mean_residual:.4f} | [{mgp_residual_ci95_low:.4f}, {mgp_residual_ci95_high:.4f}] | {mgp_lspg_mean_error_l2:.4f} | {mgp_lspg_mean_residual:.4f} | {mgp_mean_qoi_error:.4f} | {mgp_lspg_mean_qoi_error:.4f} | {mgp_speedup_vs_full:.2f} | {mgp_lspg_speedup_vs_full:.2f} | {mean_mgp_time_sec:.4f} | {mean_mgp_lspg_time_sec:.4f} | {mean_lspg_time_sec:.4f} | {mgp_error_gap_vs_lspg:.4f} | {mgp_residual_gap_vs_lspg:.4f} | {mgp_error_wins_vs_lspg}/{num_runs} | {mgp_qoi_wins_vs_lspg}/{num_runs} | {mgp_lspg_failed_case_count}/{total_case_count} |".format(
                **display
            )
        )

    lines.extend(
        [
            "",
            "## Solver Lower-Error Seed Counts",
            "",
            "| Case | Evaluation regime | Manifold Galerkin lower-error seeds vs POD-LSPG | Manifold Galerkin lower-QoI-error seeds vs POD-LSPG | Manifold-LSPG lower-error seeds vs POD-LSPG | Manifold-LSPG lower-QoI-error seeds vs POD-LSPG |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["solver_win_count_table"]:
        display = _display_row(row)
        lines.append(
            "| {benchmark_label} | {regime_label} | {mgp_error_wins_vs_lspg}/{num_runs} | {mgp_qoi_wins_vs_lspg}/{num_runs} | {mgp_lspg_error_wins_vs_lspg}/{num_runs} | {mgp_lspg_qoi_wins_vs_lspg}/{num_runs} |".format(
                **display
            )
        )

    lines.extend(
        [
            "",
            "## Convergence Failures",
            "",
            "| Case | Evaluation regime | Manifold Galerkin failures | Manifold-LSPG failures | POD-Galerkin failures | POD-LSPG failures |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["solver_failure_table"]:
        display = _display_row(row)
        lines.append(
            "| {benchmark_label} | {regime_label} | {mgp_failed_case_count}/{total_case_count} | {mgp_lspg_failed_case_count}/{total_case_count} | {pod_failed_case_count}/{total_case_count} | {lspg_failed_case_count}/{total_case_count} |".format(
                **display
            )
        )

    lines.extend(
        [
            "",
            "## Solver Times",
            "",
            "POD-LSPG is the displayed residual baseline. Any supplementary POD entries in this timing table are included for completeness rather than as the primary baseline comparison.",
            "",
            "| Case | Evaluation regime | full time (s) | Manifold Galerkin time (s) | Manifold-LSPG time (s) | POD-Galerkin time (s) | POD-LSPG time (s) | Manifold Galerkin speedup | Manifold-LSPG speedup |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["solver_time_table"]:
        display = _display_row(row)
        lines.append(
            "| {benchmark_label} | {regime_label} | {mean_full_time_sec:.4f} | {mean_mgp_time_sec:.4f} | {mean_mgp_lspg_time_sec:.4f} | {mean_pod_time_sec:.4f} | {mean_lspg_time_sec:.4f} | {mgp_speedup_vs_full:.2f} | {mgp_lspg_speedup_vs_full:.2f} |".format(
                **display
            )
        )

    lines.extend(
        [
            "",
            "## Convergence Context",
            "",
            "| Case | Evaluation regime | Manifold Galerkin conv. | Manifold Galerkin failures | Manifold Galerkin error | Manifold Galerkin QoI | Manifold Galerkin residual | Manifold-LSPG conv. | Manifold-LSPG failures | POD-LSPG conv. | POD-LSPG failures |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["convergence_context_table"]:
        display = _display_row(row)
        lines.append(
            "| {benchmark_label} | {regime_label} | {mgp_convergence_rate:.3f} | {mgp_failed_case_count}/{total_case_count} | {mgp_mean_error_l2:.4f} | {mgp_mean_qoi_error:.4f} | {mgp_mean_residual:.4f} | {mgp_lspg_convergence_rate:.3f} | {mgp_lspg_failed_case_count}/{total_case_count} | {lspg_convergence_rate:.3f} | {lspg_failed_case_count}/{total_case_count} |".format(
                **display
            )
        )

    lines.extend(
        [
            "",
            "## Solver Threshold Sensitivity",
            "",
            "| Case | Evaluation regime | baseline | tol=0.00 | tol=0.01 | tol=0.02 | tol=0.05 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["solver_threshold_sensitivity_table"]:
        display = {
            **_display_row(row),
            "baseline_recommendation": _solver_decision_label(row["baseline_recommendation"]),
            "recommendation_at_0p00": _solver_decision_label(row["recommendation_at_0p00"]),
            "recommendation_at_0p01": _solver_decision_label(row["recommendation_at_0p01"]),
            "recommendation_at_0p02": _solver_decision_label(row["recommendation_at_0p02"]),
            "recommendation_at_0p05": _solver_decision_label(row["recommendation_at_0p05"]),
        }
        lines.append(
            "| {benchmark_label} | {regime_label} | {baseline_recommendation} | {recommendation_at_0p00} | {recommendation_at_0p01} | {recommendation_at_0p02} | {recommendation_at_0p05} |".format(
                **display
            )
        )

    if "narrative" in payload:
        lines.extend(
            [
                "",
                "## Narrative",
                "",
            ]
        )
        for key, value in payload["narrative"].items():
            label = key.replace("_", " ")
            if isinstance(value, str) and value.startswith(("retain_", "prefer_", "promote_", "family_", "mixed_", "hydrologic_", "expanded_")):
                lines.append(f"- {label}: `{value}`")
            else:
                lines.append(f"- {label}: {value}")
    return "\n".join(lines) + "\n"


def _format_provenance_markdown(payload: dict) -> str:
    lines = [
        "# Reported Summary Provenance",
        "",
        "This companion artifact records tuning and selection-study provenance for the expanded summary. It is not part of the main case table.",
        "",
        "## Bratu Latent-Dimension Resolution",
        "",
        f"- decision: `{payload['bratu_resolution']['decision']}`",
    ]
    for row in payload["bratu_resolution"]["recommendations"]:
        lines.append(
            "- regime={test_regime}, recommended={recommended_label}, baseline_error={baseline_error:.4f}, recommended_error={recommended_error:.4f}, baseline_residual={baseline_residual:.4f}, recommended_residual={recommended_residual:.4f}".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Objective Table",
            "",
            "| case | evaluation regime | configuration identifier | selected solver | tangent-test weight | residual weight | error | residual | recon tangent-test residual | QoI error | error gap vs POD-LSPG | residual gap vs POD-LSPG |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["objective_table"]:
        lines.append(
            "| {benchmark_configuration} | {evaluation_regime} | {recommended_config_id} | {selected_solver} | {projected_residual_penalty_weight:.4f} | {ambient_residual_penalty_weight:.4f} | {recommended_mean_error_l2:.4f} | {recommended_mean_residual:.4f} | {reconstruction_mean_projected_residual:.3e} | {recommended_mean_qoi_error:.4f} | {recommended_error_gap_vs_lspg:.4f} | {recommended_residual_gap_vs_lspg:.4f} |".format(
                benchmark_configuration=_benchmark_label(row["preset_name"]),
                evaluation_regime=_regime_label(row["test_regime"]),
                selected_solver=_solver_decision_label(row["recommended_solver"]),
                **row,
            )
        )

    if payload.get("objective_ablation_table"):
        lines.extend(
            [
                "",
                "## Objective Ablation Provenance",
                "",
            ]
        )
        for row in payload["objective_ablation_table"]:
            lines.append(
                "- case={benchmark_configuration}, evaluation regime={evaluation_regime}, lowest error={best_by_error}, lowest residual={best_by_residual}, lowest QoI error={best_by_qoi}, recommended mode={recommended_mode}, rationale={rationale}".format(
                    benchmark_configuration=_benchmark_label(row["preset_name"]),
                    evaluation_regime=_regime_label(row["test_regime"]),
                    **row,
                )
            )

    lines.extend(["", "## Narrative", ""])
    for key, value in payload["narrative"].items():
        label = key.replace("_", " ")
        if isinstance(value, str) and value.startswith(("retain_", "prefer_", "promote_", "family_", "mixed_", "hydrologic_")):
            lines.append(f"- {label}: `{value}`")
        else:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines) + "\n"


def _write_publication_csvs(payload: dict, *, artifacts_dir: Path, csv_prefix: str) -> None:
    error_residual_rows = []
    speedup_residual_rows = []
    seedwise_rows = []
    time_rows = []
    convergence_rows = []
    for row in payload["solver_table"]:
        common = {
            "preset_name": row["preset_name"],
            "problem_name": row["problem_name"],
            "test_regime": row["test_regime"],
            "architecture_name": row["architecture_name"],
        }
        error_residual_rows.append(
            {
                **common,
                "solver_variant": "mgp_galerkin",
                "error_gap_vs_pod_lspg": row["mgp_error_gap_vs_lspg"],
                "residual_gap_vs_pod_lspg": row["mgp_residual_gap_vs_lspg"],
                "qoi_gap_vs_pod_lspg": row["mgp_qoi_gap_vs_lspg"],
            }
        )
        error_residual_rows.append(
            {
                **common,
                "solver_variant": "mgp_lspg",
                "error_gap_vs_pod_lspg": row["mgp_lspg_error_gap_vs_lspg"],
                "residual_gap_vs_pod_lspg": row["mgp_lspg_residual_gap_vs_lspg"],
                "qoi_gap_vs_pod_lspg": row["mgp_lspg_qoi_gap_vs_lspg"],
            }
        )
        speedup_residual_rows.append(
            {
                **common,
                "solver_variant": "mgp_galerkin",
                "speedup_vs_full": row["mgp_speedup_vs_full"],
                "residual_gap_vs_pod_lspg": row["mgp_residual_gap_vs_lspg"],
            }
        )
        speedup_residual_rows.append(
            {
                **common,
                "solver_variant": "mgp_lspg",
                "speedup_vs_full": row["mgp_lspg_speedup_vs_full"],
                "residual_gap_vs_pod_lspg": row["mgp_lspg_residual_gap_vs_lspg"],
            }
        )
    for row in payload["solver_time_table"]:
        time_rows.append(row)
    for row in payload["convergence_context_table"]:
        convergence_rows.append(row)
    for row in payload["seedwise_solver_rows"]:
        seedwise_rows.append(row)
    _write_csv(
        artifacts_dir / f"{csv_prefix}_error_residual.csv",
        error_residual_rows,
        [
            "preset_name",
            "problem_name",
            "test_regime",
            "architecture_name",
            "solver_variant",
            "error_gap_vs_pod_lspg",
            "residual_gap_vs_pod_lspg",
            "qoi_gap_vs_pod_lspg",
        ],
    )
    _write_csv(
        artifacts_dir / f"{csv_prefix}_speedup_residual.csv",
        speedup_residual_rows,
        [
            "preset_name",
            "problem_name",
            "test_regime",
            "architecture_name",
            "solver_variant",
            "speedup_vs_full",
            "residual_gap_vs_pod_lspg",
        ],
    )
    _write_csv(
        artifacts_dir / f"{csv_prefix}_seedwise_solver_rows.csv",
        seedwise_rows,
        [
            "preset_name",
            "problem_name",
            "test_regime",
            "architecture_name",
            "seed",
            "qoi_name",
            "solver_variant",
            "mean_error_l2",
            "mean_residual",
            "mean_projected_residual",
            "mean_qoi_error",
            "convergence_rate",
            "speedup_vs_full",
            "mean_time_sec",
            "error_gap_vs_lspg",
            "residual_gap_vs_lspg",
            "qoi_gap_vs_lspg",
        ],
    )
    _write_csv(
        artifacts_dir / f"{csv_prefix}_solver_times.csv",
        time_rows,
        [
            "preset_name",
            "problem_name",
            "test_regime",
            "mean_full_time_sec",
            "mean_mgp_time_sec",
            "mean_mgp_lspg_time_sec",
            "mean_pod_time_sec",
            "mean_lspg_time_sec",
            "mgp_speedup_vs_full",
            "mgp_lspg_speedup_vs_full",
        ],
    )
    _write_csv(
        artifacts_dir / f"{csv_prefix}_convergence_context.csv",
        convergence_rows,
        [
            "preset_name",
            "problem_name",
            "test_regime",
            "qoi_name",
            "num_runs",
            "total_case_count",
            "mgp_convergence_rate",
            "mgp_failed_case_count",
            "mgp_mean_error_l2",
            "mgp_mean_qoi_error",
            "mgp_mean_residual",
            "mgp_lspg_convergence_rate",
            "mgp_lspg_failed_case_count",
            "mgp_lspg_mean_error_l2",
            "mgp_lspg_mean_qoi_error",
            "mgp_lspg_mean_residual",
            "lspg_convergence_rate",
            "lspg_failed_case_count",
            "lspg_mean_error_l2",
            "lspg_mean_qoi_error",
            "lspg_mean_residual",
        ],
    )


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _threshold_label(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (q / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


if __name__ == "__main__":
    main()
