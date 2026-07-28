from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from .presets import get_benchmark_preset
from .solver_tradeoff import (
    build_solver_tradeoff_benchmark_config,
    pick_solver_recommendation,
    run_solver_tradeoff_case,
)


@dataclass(frozen=True)
class ObjectiveWeightConfig:
    config_id: str
    projected_residual_penalty_weight: float
    ambient_residual_penalty_weight: float


@dataclass
class ObjectiveWeightStudyConfig:
    preset_names: tuple[str, ...]
    test_regimes: tuple[str, ...] = ("in_domain", "ood")
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    train_size: int = 6
    test_size: int = 2
    epochs: int = 40
    deterministic_publication_mode: bool = True
    publication_num_threads: int = 1
    offline_cache_dir: str = "artifacts/cache"
    use_offline_cache: bool = True
    refresh_offline_cache: bool = False
    baseline_summary_path: str = "artifacts/publication_summary.json"


def default_objective_weight_configs_for_preset(preset_name: str) -> tuple[ObjectiveWeightConfig, ...]:
    if preset_name in ("reference_front_layer", "reference_nonlinear_diffusion"):
        return (
            ObjectiveWeightConfig("ambient_0p02", 0.0, 0.02),
            ObjectiveWeightConfig("ambient_0p03", 0.0, 0.03),
            ObjectiveWeightConfig("mixed_p0p005_a0p02", 0.005, 0.02),
            ObjectiveWeightConfig("mixed_p0p01_a0p02", 0.01, 0.02),
        )
    if preset_name == "reference_bratu_source":
        return (
            ObjectiveWeightConfig("projected_0p02", 0.02, 0.0),
            ObjectiveWeightConfig("projected_0p03", 0.03, 0.0),
            ObjectiveWeightConfig("mixed_p0p02_a0p0025", 0.02, 0.0025),
            ObjectiveWeightConfig("mixed_p0p03_a0p0025", 0.03, 0.0025),
        )
    raise ValueError(f"no default objective-weight configs for preset: {preset_name}")


def run_objective_weight_case(
    preset_name: str,
    weight_config: ObjectiveWeightConfig,
    *,
    test_regime: str,
    seed: int,
    train_size: int,
    test_size: int,
    epochs: int,
    deterministic_publication_mode: bool,
    publication_num_threads: int,
    offline_cache_dir: str,
    use_offline_cache: bool,
    refresh_offline_cache: bool,
) -> dict:
    benchmark_config = build_solver_tradeoff_benchmark_config(
        preset_name,
        test_regime=test_regime,
        seed=seed,
        train_size=train_size,
        test_size=test_size,
        epochs=epochs,
        deterministic_publication_mode=deterministic_publication_mode,
        publication_num_threads=publication_num_threads,
        offline_cache_dir=offline_cache_dir,
        use_offline_cache=use_offline_cache,
        refresh_offline_cache=refresh_offline_cache,
        projected_residual_penalty_weight=weight_config.projected_residual_penalty_weight,
        ambient_residual_penalty_weight=weight_config.ambient_residual_penalty_weight,
    )
    run = run_solver_tradeoff_case(preset_name, benchmark_config)
    return {
        **run,
        "objective_weight_config": asdict(weight_config),
        "config_id": weight_config.config_id,
    }


def run_objective_weight_study(config: ObjectiveWeightStudyConfig) -> dict:
    runs = []
    for preset_name in config.preset_names:
        for weight_config in default_objective_weight_configs_for_preset(preset_name):
            for test_regime in config.test_regimes:
                for seed in config.seeds:
                    runs.append(
                        run_objective_weight_case(
                            preset_name,
                            weight_config,
                            test_regime=test_regime,
                            seed=seed,
                            train_size=config.train_size,
                            test_size=config.test_size,
                            epochs=config.epochs,
                            deterministic_publication_mode=config.deterministic_publication_mode,
                            publication_num_threads=config.publication_num_threads,
                            offline_cache_dir=config.offline_cache_dir,
                            use_offline_cache=config.use_offline_cache,
                            refresh_offline_cache=config.refresh_offline_cache,
                        )
                    )
    return build_objective_weight_payload(asdict(config), runs, config.baseline_summary_path)


def build_objective_weight_payload(config: dict, runs: list[dict], baseline_summary_path: str) -> dict:
    aggregated = aggregate_objective_weight_runs(runs)
    recommendations = pick_objective_weight_recommendations(aggregated, baseline_summary_path)
    return {
        "config": config,
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed_list": sorted({run["seed"] for run in runs}),
            "deterministic_publication_mode": any(
                run["metadata"]["deterministic_publication_mode"] for run in runs
            ),
            "publication_num_threads": max(
                run["metadata"]["publication_num_threads_requested"] for run in runs
            ),
            "baseline_summary_path": baseline_summary_path,
        },
        "runs": runs,
        "aggregated": aggregated,
        "recommendations": recommendations,
    }


def aggregate_objective_weight_runs(runs: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for run in runs:
        key = (run["preset_name"], run["test_regime"], run["config_id"])
        grouped.setdefault(key, []).append(run)

    rows = []
    for (preset_name, test_regime, config_id), group in sorted(grouped.items()):
        first = group[0]
        row = {
            "preset_name": preset_name,
            "problem_name": first["problem_name"],
            "test_regime": test_regime,
            "architecture_name": first["architecture_name"],
            "config_id": config_id,
            "num_runs": len(group),
            "qoi_name": first["qoi_name"],
            "projected_residual_penalty_weight": first["objective_weight_config"][
                "projected_residual_penalty_weight"
            ],
            "ambient_residual_penalty_weight": first["objective_weight_config"][
                "ambient_residual_penalty_weight"
            ],
            "deterministic_publication_mode": first["metadata"]["deterministic_publication_mode"],
            "publication_num_threads": first["metadata"]["publication_num_threads_requested"],
            "mgp_mean_error_l2": _mean_summary(group, "mgp_mean_error_l2"),
            "mgp_lspg_mean_error_l2": _mean_summary(group, "mgp_lspg_mean_error_l2"),
            "mgp_std_error_l2": _pstdev_summary(group, "mgp_mean_error_l2"),
            "mgp_lspg_std_error_l2": _pstdev_summary(group, "mgp_lspg_mean_error_l2"),
            "mgp_mean_residual": _mean_summary(group, "mgp_mean_residual"),
            "mgp_lspg_mean_residual": _mean_summary(group, "mgp_lspg_mean_residual"),
            "mgp_mean_projected_residual": _mean_summary(group, "mgp_mean_projected_residual"),
            "mgp_lspg_mean_projected_residual": _mean_summary(group, "mgp_lspg_mean_projected_residual"),
            "reconstruction_mean_error_l2": _mean_summary(group, "reconstruction_mean_error_l2"),
            "reconstruction_mean_residual": _mean_summary(group, "reconstruction_mean_residual"),
            "reconstruction_mean_projected_residual": _mean_summary(
                group, "reconstruction_mean_projected_residual"
            ),
            "mgp_mean_qoi_error": _mean_summary(group, "mgp_mean_qoi_error"),
            "mgp_lspg_mean_qoi_error": _mean_summary(group, "mgp_lspg_mean_qoi_error"),
            "mgp_convergence_rate": _mean_summary(group, "mgp_convergence_rate"),
            "mgp_lspg_convergence_rate": _mean_summary(group, "mgp_lspg_convergence_rate"),
            "mgp_speedup_vs_full": _mean_summary(group, "mgp_speedup_vs_full"),
            "mgp_lspg_speedup_vs_full": _mean_summary(group, "mgp_lspg_speedup_vs_full"),
            "lspg_mean_error_l2": _mean_summary(group, "lspg_mean_error_l2"),
            "lspg_mean_residual": _mean_summary(group, "lspg_mean_residual"),
            "lspg_mean_qoi_error": _mean_summary(group, "lspg_mean_qoi_error"),
            "mgp_error_gap_vs_lspg": _mean_summary(group, "mgp_mean_error_l2")
            - _mean_summary(group, "lspg_mean_error_l2"),
            "mgp_residual_gap_vs_lspg": _mean_summary(group, "mgp_mean_residual")
            - _mean_summary(group, "lspg_mean_residual"),
            "mgp_qoi_gap_vs_lspg": _mean_summary(group, "mgp_mean_qoi_error")
            - _mean_summary(group, "lspg_mean_qoi_error"),
            "mgp_lspg_error_gap_vs_lspg": _mean_summary(group, "mgp_lspg_mean_error_l2")
            - _mean_summary(group, "lspg_mean_error_l2"),
            "mgp_lspg_residual_gap_vs_lspg": _mean_summary(group, "mgp_lspg_mean_residual")
            - _mean_summary(group, "lspg_mean_residual"),
            "mgp_lspg_qoi_gap_vs_lspg": _mean_summary(group, "mgp_lspg_mean_qoi_error")
            - _mean_summary(group, "lspg_mean_qoi_error"),
        }
        row["solver_recommendation"] = pick_solver_recommendation(row)
        row.update(_recommended_solver_metrics(row))
        rows.append(row)
    return rows


def pick_objective_weight_recommendations(rows: list[dict], baseline_summary_path: str) -> list[dict]:
    baseline_rows = _load_baseline_solver_rows(baseline_summary_path)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["preset_name"], []).append(row)

    recommendations = []
    for preset_name, group in sorted(grouped.items()):
        by_config: dict[str, list[dict]] = {}
        for row in group:
            by_config.setdefault(row["config_id"], []).append(row)
        if preset_name == "reference_front_layer":
            recommendations.append(_pick_front_layer_recommendation(preset_name, by_config, baseline_rows))
        elif preset_name == "reference_bratu_source":
            recommendations.append(_pick_bratu_recommendation(preset_name, by_config, baseline_rows))
        elif preset_name == "reference_nonlinear_diffusion":
            recommendations.append(_pick_nonlinear_diffusion_recommendation(preset_name, by_config, baseline_rows))
        else:
            raise ValueError(f"no recommendation rule for preset: {preset_name}")
    return recommendations


def format_objective_weight_study_markdown(payload: dict) -> str:
    lines = [
        "# Objective Weight Study",
        "",
        "## Recommendations",
        "",
    ]
    for row in payload["recommendations"]:
        lines.append(
            "- preset={preset_name}, promoted={promoted}, recommended_config={recommended_config_id}, rationale={rationale}".format(
                **row
            )
        )

    lines.extend(["", "## Aggregated Results", ""])
    for row in payload["aggregated"]:
        format_row = {
            **row,
            "mgp_mean_projected_residual": _format_scientific(row["mgp_mean_projected_residual"]),
            "mgp_lspg_mean_projected_residual": _format_scientific(
                row["mgp_lspg_mean_projected_residual"]
            ),
            "reconstruction_mean_projected_residual": _format_scientific(
                row["reconstruction_mean_projected_residual"]
            ),
        }
        lines.append(
            "- preset={preset_name}, regime={test_regime}, config_id={config_id}, architecture={architecture_name}, runs={num_runs}, projected_weight={projected_residual_penalty_weight:.4f}, ambient_weight={ambient_residual_penalty_weight:.4f}, solver_recommendation={solver_recommendation}, recommended_mean_error_l2={recommended_mean_error_l2:.6f}, recommended_mean_residual={recommended_mean_residual:.6f}, recommended_mean_projected_residual={recommended_mean_projected_residual}, recommended_mean_qoi_error={recommended_mean_qoi_error:.6f}, recommended_error_gap_vs_lspg={recommended_error_gap_vs_lspg:.6f}, recommended_residual_gap_vs_lspg={recommended_residual_gap_vs_lspg:.6f}, reconstruction_mean_residual={reconstruction_mean_residual:.6f}, reconstruction_mean_projected_residual={reconstruction_mean_projected_residual}, recommended_speedup_vs_full={recommended_speedup_vs_full:.3f}".format(
                **format_row
            )
        )
    return "\n".join(lines) + "\n"


def _recommended_solver_metrics(row: dict) -> dict:
    if row["solver_recommendation"] == "prefer_manifold_lspg":
        return {
            "recommended_solver": "manifold_lspg",
            "recommended_mean_error_l2": row["mgp_lspg_mean_error_l2"],
            "recommended_mean_residual": row["mgp_lspg_mean_residual"],
            "recommended_mean_projected_residual": row["mgp_lspg_mean_projected_residual"],
            "recommended_mean_qoi_error": row["mgp_lspg_mean_qoi_error"],
            "recommended_error_gap_vs_lspg": row["mgp_lspg_error_gap_vs_lspg"],
            "recommended_residual_gap_vs_lspg": row["mgp_lspg_residual_gap_vs_lspg"],
            "recommended_qoi_gap_vs_lspg": row["mgp_lspg_qoi_gap_vs_lspg"],
            "recommended_speedup_vs_full": row["mgp_lspg_speedup_vs_full"],
        }
    return {
        "recommended_solver": "manifold_galerkin",
        "recommended_mean_error_l2": row["mgp_mean_error_l2"],
        "recommended_mean_residual": row["mgp_mean_residual"],
        "recommended_mean_projected_residual": row["mgp_mean_projected_residual"],
        "recommended_mean_qoi_error": row["mgp_mean_qoi_error"],
        "recommended_error_gap_vs_lspg": row["mgp_error_gap_vs_lspg"],
        "recommended_residual_gap_vs_lspg": row["mgp_residual_gap_vs_lspg"],
        "recommended_qoi_gap_vs_lspg": row["mgp_qoi_gap_vs_lspg"],
        "recommended_speedup_vs_full": row["mgp_speedup_vs_full"],
    }


def _load_baseline_solver_rows(path: str) -> dict[tuple[str, str], dict]:
    payload = json.loads(Path(path).read_text())
    rows = {}
    for row in payload["solver_table"]:
        baseline = {
            "solver_recommendation": row["solver_recommendation"],
            "recommended_mean_error_l2": row["mgp_lspg_mean_error_l2"]
            if row["solver_recommendation"] == "prefer_manifold_lspg"
            else row["mgp_mean_error_l2"],
            "recommended_mean_residual": row["mgp_lspg_mean_residual"]
            if row["solver_recommendation"] == "prefer_manifold_lspg"
            else row["mgp_mean_residual"],
            "recommended_mean_qoi_error": row["mgp_lspg_mean_qoi_error"]
            if row["solver_recommendation"] == "prefer_manifold_lspg"
            else row["mgp_mean_qoi_error"],
            "recommended_error_gap_vs_lspg": row["mgp_lspg_error_gap_vs_lspg"]
            if row["solver_recommendation"] == "prefer_manifold_lspg"
            else row["mgp_error_gap_vs_lspg"],
            "recommended_residual_gap_vs_lspg": row["mgp_lspg_residual_gap_vs_lspg"]
            if row["solver_recommendation"] == "prefer_manifold_lspg"
            else row["mgp_residual_gap_vs_lspg"],
        }
        rows[(row["preset_name"], row["test_regime"])] = baseline
    return rows


def _pick_front_layer_recommendation(
    preset_name: str, by_config: dict[str, list[dict]], baseline_rows: dict[tuple[str, str], dict]
) -> dict:
    eligible = []
    for config_id, rows in by_config.items():
        row_map = {row["test_regime"]: row for row in rows}
        in_domain = row_map["in_domain"]
        if in_domain["recommended_error_gap_vs_lspg"] <= 0.0 and in_domain["recommended_qoi_gap_vs_lspg"] <= 0.05:
            avg_residual_gap = statistics.mean(
                row["recommended_residual_gap_vs_lspg"] for row in row_map.values()
            )
            eligible.append((avg_residual_gap, config_id))
    if not eligible:
        return {
            "preset_name": preset_name,
            "promoted": False,
            "recommended_config_id": "retain_current_canonical_mixed",
            "rationale": "no candidate preserved the front-layer in-domain error win and QoI guardrail",
        }
    _, config_id = min(eligible)
    return {
        "preset_name": preset_name,
        "promoted": True,
        "recommended_config_id": config_id,
        "rationale": "minimizes average residual gap while keeping the front-layer in-domain error win and QoI guardrail",
    }


def _pick_bratu_recommendation(
    preset_name: str, by_config: dict[str, list[dict]], baseline_rows: dict[tuple[str, str], dict]
) -> dict:
    baseline_in_domain = baseline_rows[(preset_name, "in_domain")]
    eligible = []
    for config_id, rows in by_config.items():
        row_map = {row["test_regime"]: row for row in rows}
        in_domain = row_map["in_domain"]
        ood = row_map["ood"]
        if (
            ood["recommended_error_gap_vs_lspg"] <= 0.0
            and ood["recommended_qoi_gap_vs_lspg"] <= 0.0
            and in_domain["recommended_mean_residual"]
            <= baseline_in_domain["recommended_mean_residual"] * 1.2
        ):
            avg_residual_gap = statistics.mean(
                row["recommended_residual_gap_vs_lspg"] for row in row_map.values()
            )
            eligible.append((avg_residual_gap, config_id))
    if not eligible:
        return {
            "preset_name": preset_name,
            "promoted": False,
            "recommended_config_id": "retain_current_canonical_mixed",
            "rationale": "no candidate preserved the Bratu OOD error/QoI win under the in-domain residual guardrail",
        }
    _, config_id = min(eligible)
    return {
        "preset_name": preset_name,
        "promoted": True,
        "recommended_config_id": config_id,
        "rationale": "minimizes average residual gap while preserving the Bratu OOD error/QoI win and in-domain residual guardrail",
    }


def _pick_nonlinear_diffusion_recommendation(
    preset_name: str, by_config: dict[str, list[dict]], baseline_rows: dict[tuple[str, str], dict]
) -> dict:
    baseline_error_gap = statistics.mean(
        baseline_rows[(preset_name, regime)]["recommended_error_gap_vs_lspg"] for regime in ("in_domain", "ood")
    )
    baseline_residual_gap = statistics.mean(
        baseline_rows[(preset_name, regime)]["recommended_residual_gap_vs_lspg"]
        for regime in ("in_domain", "ood")
    )

    eligible = []
    for config_id, rows in by_config.items():
        avg_error_gap = statistics.mean(row["recommended_error_gap_vs_lspg"] for row in rows)
        avg_residual_gap = statistics.mean(row["recommended_residual_gap_vs_lspg"] for row in rows)
        if avg_error_gap <= baseline_error_gap * 0.9 and avg_residual_gap <= baseline_residual_gap * 0.9:
            eligible.append((avg_residual_gap, avg_error_gap, config_id))
    if not eligible:
        return {
            "preset_name": preset_name,
            "promoted": False,
            "recommended_config_id": "retain_current_canonical_mixed",
            "rationale": "no candidate improved both nonlinear-diffusion mean error gap and residual gap by at least 10%",
        }
    _, _, config_id = min(eligible)
    return {
        "preset_name": preset_name,
        "promoted": True,
        "recommended_config_id": config_id,
        "rationale": "improves both nonlinear-diffusion mean error gap and residual gap by at least 10%",
    }


def _mean_summary(group: list[dict], key: str) -> float:
    return float(sum(run["summary"][key] for run in group) / len(group))


def _pstdev_summary(group: list[dict], key: str) -> float:
    values = [run["summary"][key] for run in group]
    if len(values) == 1:
        return 0.0
    return float(statistics.pstdev(values))


def _format_scientific(value: float) -> str:
    return f"{value:.3e}"
