from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import statistics

from .benchmark import run_benchmark
from .presets import get_benchmark_preset


@dataclass(frozen=True)
class ObjectiveAblationMode:
    name: str
    projected_residual_penalty_weight: float
    ambient_residual_penalty_weight: float


@dataclass
class ObjectiveAblationStudyConfig:
    preset_names: tuple[str, ...]
    test_regimes: tuple[str, ...] = ("in_domain", "ood")
    seeds: tuple[int, ...] = (0, 1, 2)
    train_size: int | None = None
    test_size: int | None = None
    epochs: int | None = None
    deterministic_publication_mode: bool = False
    publication_num_threads: int = 1
    modes: tuple[ObjectiveAblationMode, ...] = (
        ObjectiveAblationMode(
            name="ambient_only",
            projected_residual_penalty_weight=0.0,
            ambient_residual_penalty_weight=0.02,
        ),
        ObjectiveAblationMode(
            name="projected_only",
            projected_residual_penalty_weight=0.02,
            ambient_residual_penalty_weight=0.0,
        ),
        ObjectiveAblationMode(
            name="projected_plus_ambient",
            projected_residual_penalty_weight=0.02,
            ambient_residual_penalty_weight=0.005,
        ),
    )


def run_objective_ablation_study(config: ObjectiveAblationStudyConfig) -> dict:
    runs = []
    for preset_name in config.preset_names:
        base_config = get_benchmark_preset(preset_name)
        for mode in config.modes:
            for test_regime in config.test_regimes:
                for seed in config.seeds:
                    benchmark_config = replace(
                        base_config,
                        test_regime=test_regime,
                        seed=seed,
                        train_size=config.train_size
                        if config.train_size is not None
                        else base_config.train_size,
                        test_size=config.test_size if config.test_size is not None else base_config.test_size,
                        autoencoder_epochs=config.epochs
                        if config.epochs is not None
                        else base_config.autoencoder_epochs,
                        projected_residual_penalty_weight=mode.projected_residual_penalty_weight,
                        ambient_residual_penalty_weight=mode.ambient_residual_penalty_weight,
                        deterministic_publication_mode=config.deterministic_publication_mode,
                        publication_num_threads=config.publication_num_threads,
                    )
                    result = run_benchmark(benchmark_config)
                    runs.append(
                        {
                            "preset_name": preset_name,
                            "problem_name": benchmark_config.problem_name,
                            "test_regime": test_regime,
                            "mode_name": mode.name,
                            "seed": seed,
                            "config": asdict(benchmark_config),
                            "metadata": result["metadata"],
                            "summary": result["summary"],
                            "qoi_name": result["qoi_name"],
                        }
                    )

    aggregated = aggregate_objective_ablation_runs(runs)
    recommendations = pick_objective_ablation_recommendations(aggregated)
    return {
        "config": {
            **asdict(config),
            "modes": [asdict(mode) for mode in config.modes],
        },
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed_list": list(config.seeds),
            "deterministic_publication_mode": config.deterministic_publication_mode,
            "publication_num_threads": config.publication_num_threads,
        },
        "runs": runs,
        "aggregated": aggregated,
        "recommendations": recommendations,
    }


def aggregate_objective_ablation_runs(runs: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for run in runs:
        key = (run["preset_name"], run["test_regime"], run["mode_name"])
        grouped.setdefault(key, []).append(run)

    rows = []
    for (preset_name, test_regime, mode_name), group in sorted(grouped.items()):
        first = group[0]
        rows.append(
            {
                "preset_name": preset_name,
                "problem_name": first["problem_name"],
                "test_regime": test_regime,
                "mode_name": mode_name,
                "num_runs": len(group),
                "qoi_name": first["qoi_name"],
                "projected_residual_penalty_weight": first["config"]["projected_residual_penalty_weight"],
                "ambient_residual_penalty_weight": first["config"]["ambient_residual_penalty_weight"],
                "mgp_mean_error_l2": _mean_summary(group, "mgp_mean_error_l2"),
                "mgp_std_error_l2": _pstdev_summary(group, "mgp_mean_error_l2"),
                "mgp_mean_residual": _mean_summary(group, "mgp_mean_residual"),
                "mgp_std_residual": _pstdev_summary(group, "mgp_mean_residual"),
                "mgp_mean_projected_residual": _mean_summary(group, "mgp_mean_projected_residual"),
                "reconstruction_mean_error_l2": _mean_summary(group, "reconstruction_mean_error_l2"),
                "reconstruction_mean_residual": _mean_summary(group, "reconstruction_mean_residual"),
                "reconstruction_mean_projected_residual": _mean_summary(
                    group, "reconstruction_mean_projected_residual"
                ),
                "mgp_mean_qoi_error": _mean_summary(group, "mgp_mean_qoi_error"),
                "mgp_convergence_rate": _mean_summary(group, "mgp_convergence_rate"),
                "mgp_speedup_vs_full": _mean_summary(group, "mgp_speedup_vs_full"),
                "lspg_mean_error_l2": _mean_summary(group, "lspg_mean_error_l2"),
                "lspg_mean_residual": _mean_summary(group, "lspg_mean_residual"),
                "lspg_mean_qoi_error": _mean_summary(group, "lspg_mean_qoi_error"),
                "deterministic_publication_mode": first["metadata"]["deterministic_publication_mode"],
                "publication_num_threads": first["metadata"]["publication_num_threads_requested"],
                "mgp_error_gap_vs_lspg": _mean_summary(group, "mgp_mean_error_l2")
                - _mean_summary(group, "lspg_mean_error_l2"),
                "mgp_residual_gap_vs_lspg": _mean_summary(group, "mgp_mean_residual")
                - _mean_summary(group, "lspg_mean_residual"),
                "mgp_qoi_gap_vs_lspg": _mean_summary(group, "mgp_mean_qoi_error")
                - _mean_summary(group, "lspg_mean_qoi_error"),
            }
        )
    return rows


def pick_objective_ablation_recommendations(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["preset_name"], row["test_regime"])
        grouped.setdefault(key, []).append(row)

    recommendations = []
    for (preset_name, test_regime), group in sorted(grouped.items()):
        best_error = min(group, key=lambda row: row["mgp_mean_error_l2"])
        best_residual = min(group, key=lambda row: row["mgp_mean_residual"])
        best_qoi = min(group, key=lambda row: row["mgp_mean_qoi_error"])
        recommendations.append(
            {
                "preset_name": preset_name,
                "problem_name": best_error["problem_name"],
                "test_regime": test_regime,
                "best_by_error": best_error["mode_name"],
                "best_by_residual": best_residual["mode_name"],
                "best_by_qoi": best_qoi["mode_name"],
                "recommended_mode": best_error["mode_name"],
                "rationale": "prefer lowest MGP error; inspect residual and QoI as separate axes",
            }
        )
    return recommendations


def format_objective_ablation_study_markdown(payload: dict) -> str:
    lines = [
        "# Objective Ablation Study",
        "",
        "## Recommendations",
        "",
    ]
    for row in payload["recommendations"]:
        lines.append(
            "- preset={preset_name}, problem={problem_name}, regime={test_regime}, recommended={recommended_mode}, best_error={best_by_error}, best_residual={best_by_residual}, best_qoi={best_by_qoi}, rationale={rationale}".format(
                **row
            )
        )

    lines.extend(["", "## Aggregated Results", ""])
    for row in payload["aggregated"]:
        format_row = {
            **row,
            "mgp_mean_projected_residual": _format_scientific(row["mgp_mean_projected_residual"]),
            "reconstruction_mean_projected_residual": _format_scientific(
                row["reconstruction_mean_projected_residual"]
            ),
        }
        lines.append(
            "- preset={preset_name}, problem={problem_name}, regime={test_regime}, mode={mode_name}, runs={num_runs}, publication_mode={deterministic_publication_mode}, threads={publication_num_threads}, projected_weight={projected_residual_penalty_weight:.4f}, ambient_weight={ambient_residual_penalty_weight:.4f}, mgp_mean_error_l2={mgp_mean_error_l2:.6f}, mgp_std_error_l2={mgp_std_error_l2:.6f}, mgp_mean_residual={mgp_mean_residual:.6f}, mgp_mean_projected_residual={mgp_mean_projected_residual}, reconstruction_mean_residual={reconstruction_mean_residual:.6f}, reconstruction_mean_projected_residual={reconstruction_mean_projected_residual}, mgp_mean_qoi_error={mgp_mean_qoi_error:.6f}, error_gap_vs_lspg={mgp_error_gap_vs_lspg:.6f}, residual_gap_vs_lspg={mgp_residual_gap_vs_lspg:.6f}, mgp_speedup_vs_full={mgp_speedup_vs_full:.3f}".format(
                **format_row,
            )
        )
    return "\n".join(lines) + "\n"


def _mean_summary(group: list[dict], key: str) -> float:
    return float(sum(run["summary"][key] for run in group) / len(group))


def _pstdev_summary(group: list[dict], key: str) -> float:
    values = [run["summary"][key] for run in group]
    if len(values) == 1:
        return 0.0
    return float(statistics.pstdev(values))


def _format_scientific(value: float) -> str:
    return f"{value:.3e}"
