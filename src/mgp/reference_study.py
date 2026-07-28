from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import statistics

from .benchmark import BenchmarkConfig, run_benchmark
from .presets import get_benchmark_preset


@dataclass
class ReferenceArchitectureStudyConfig:
    preset_names: tuple[str, ...]
    architecture_names: tuple[str, ...] = ("mlp", "conv1d")
    test_regimes: tuple[str, ...] = ("in_domain", "ood")
    seeds: tuple[int, ...] = (0, 1, 2)
    train_size: int | None = None
    test_size: int | None = None
    epochs: int | None = None
    deterministic_publication_mode: bool = False
    publication_num_threads: int = 1


def run_reference_architecture_study(config: ReferenceArchitectureStudyConfig) -> dict:
    runs = []
    for preset_name in config.preset_names:
        base_config = get_benchmark_preset(preset_name)
        for architecture_name in config.architecture_names:
            for test_regime in config.test_regimes:
                for seed in config.seeds:
                    benchmark_config = replace(
                        base_config,
                        architecture_name=architecture_name,
                        test_regime=test_regime,
                        seed=seed,
                        train_size=config.train_size
                        if config.train_size is not None
                        else base_config.train_size,
                        test_size=config.test_size if config.test_size is not None else base_config.test_size,
                        autoencoder_epochs=config.epochs
                        if config.epochs is not None
                        else base_config.autoencoder_epochs,
                        deterministic_publication_mode=config.deterministic_publication_mode,
                        publication_num_threads=config.publication_num_threads,
                    )
                    result = run_benchmark(benchmark_config)
                    runs.append(
                        {
                            "preset_name": preset_name,
                            "problem_name": benchmark_config.problem_name,
                            "test_regime": test_regime,
                            "architecture_name": architecture_name,
                            "seed": seed,
                            "config": asdict(benchmark_config),
                            "metadata": result["metadata"],
                            "summary": result["summary"],
                            "qoi_name": result["qoi_name"],
                        }
                    )

    aggregated = aggregate_reference_architecture_runs(runs)
    decisions = pick_reference_architecture_decisions(aggregated)
    return {
        "config": asdict(config),
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed_list": list(config.seeds),
            "deterministic_publication_mode": config.deterministic_publication_mode,
            "publication_num_threads": config.publication_num_threads,
        },
        "runs": runs,
        "aggregated": aggregated,
        "decisions": decisions,
    }


def aggregate_reference_architecture_runs(runs: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    for run in runs:
        key = (
            run["preset_name"],
            run["problem_name"],
            run["test_regime"],
            run["architecture_name"],
        )
        grouped.setdefault(key, []).append(run)

    aggregated = []
    for key, group in grouped.items():
        preset_name, problem_name, test_regime, architecture_name = key
        aggregated.append(
            {
                "preset_name": preset_name,
                "problem_name": problem_name,
                "test_regime": test_regime,
                "architecture_name": architecture_name,
                "num_runs": len(group),
                "qoi_name": group[0]["qoi_name"],
                "mgp_mean_error_l2": _mean_summary(group, "mgp_mean_error_l2"),
                "mgp_lspg_mean_error_l2": _mean_summary(group, "mgp_lspg_mean_error_l2"),
                "mgp_median_error_l2": _median_summary(group, "mgp_mean_error_l2"),
                "mgp_lspg_median_error_l2": _median_summary(group, "mgp_lspg_mean_error_l2"),
                "mgp_std_error_l2": _pstdev_summary(group, "mgp_mean_error_l2"),
                "mgp_lspg_std_error_l2": _pstdev_summary(group, "mgp_lspg_mean_error_l2"),
                "mgp_mean_residual": _mean_summary(group, "mgp_mean_residual"),
                "mgp_lspg_mean_residual": _mean_summary(group, "mgp_lspg_mean_residual"),
                "mgp_mean_projected_residual": _mean_summary(group, "mgp_mean_projected_residual"),
                "mgp_lspg_mean_projected_residual": _mean_summary(
                    group, "mgp_lspg_mean_projected_residual"
                ),
                "mgp_median_residual": _median_summary(group, "mgp_mean_residual"),
                "mgp_lspg_median_residual": _median_summary(group, "mgp_lspg_mean_residual"),
                "mgp_std_residual": _pstdev_summary(group, "mgp_mean_residual"),
                "mgp_lspg_std_residual": _pstdev_summary(group, "mgp_lspg_mean_residual"),
                "reconstruction_mean_error_l2": _mean_summary(group, "reconstruction_mean_error_l2"),
                "reconstruction_mean_residual": _mean_summary(group, "reconstruction_mean_residual"),
                "reconstruction_mean_projected_residual": _mean_summary(
                    group, "reconstruction_mean_projected_residual"
                ),
                "mgp_mean_qoi_error": _mean_summary(group, "mgp_mean_qoi_error"),
                "mgp_lspg_mean_qoi_error": _mean_summary(group, "mgp_lspg_mean_qoi_error"),
                "mgp_std_qoi_error": _pstdev_summary(group, "mgp_mean_qoi_error"),
                "mgp_lspg_std_qoi_error": _pstdev_summary(group, "mgp_lspg_mean_qoi_error"),
                "mgp_convergence_rate": _mean_summary(group, "mgp_convergence_rate"),
                "mgp_lspg_convergence_rate": _mean_summary(group, "mgp_lspg_convergence_rate"),
                "mgp_speedup_vs_full": _mean_summary(group, "mgp_speedup_vs_full"),
                "mgp_lspg_speedup_vs_full": _mean_summary(group, "mgp_lspg_speedup_vs_full"),
                "projected_residual_penalty_weight": _effective_projected_weight(group[0]["config"]),
                "ambient_residual_penalty_weight": group[0]["config"]["ambient_residual_penalty_weight"],
                "deterministic_publication_mode": group[0]["metadata"]["deterministic_publication_mode"],
                "publication_num_threads": group[0]["metadata"]["publication_num_threads_requested"],
                "pod_mean_error_l2": _mean_summary(group, "pod_mean_error_l2"),
                "pod_mean_residual": _mean_summary(group, "pod_mean_residual"),
                "pod_mean_qoi_error": _mean_summary(group, "pod_mean_qoi_error"),
                "lspg_mean_error_l2": _mean_summary(group, "lspg_mean_error_l2"),
                "lspg_mean_residual": _mean_summary(group, "lspg_mean_residual"),
                "lspg_mean_qoi_error": _mean_summary(group, "lspg_mean_qoi_error"),
                "mgp_error_gap_vs_pod": _mean_summary(group, "mgp_mean_error_l2")
                - _mean_summary(group, "pod_mean_error_l2"),
                "mgp_error_gap_vs_lspg": _mean_summary(group, "mgp_mean_error_l2")
                - _mean_summary(group, "lspg_mean_error_l2"),
                "mgp_residual_gap_vs_pod": _mean_summary(group, "mgp_mean_residual")
                - _mean_summary(group, "pod_mean_residual"),
                "mgp_residual_gap_vs_lspg": _mean_summary(group, "mgp_mean_residual")
                - _mean_summary(group, "lspg_mean_residual"),
                "mgp_lspg_error_gap_vs_pod": _mean_summary(group, "mgp_lspg_mean_error_l2")
                - _mean_summary(group, "pod_mean_error_l2"),
                "mgp_lspg_error_gap_vs_lspg": _mean_summary(group, "mgp_lspg_mean_error_l2")
                - _mean_summary(group, "lspg_mean_error_l2"),
                "mgp_lspg_residual_gap_vs_pod": _mean_summary(group, "mgp_lspg_mean_residual")
                - _mean_summary(group, "pod_mean_residual"),
                "mgp_lspg_residual_gap_vs_lspg": _mean_summary(group, "mgp_lspg_mean_residual")
                - _mean_summary(group, "lspg_mean_residual"),
                "mgp_qoi_gap_vs_pod": _mean_summary(group, "mgp_mean_qoi_error")
                - _mean_summary(group, "pod_mean_qoi_error"),
                "mgp_qoi_gap_vs_lspg": _mean_summary(group, "mgp_mean_qoi_error")
                - _mean_summary(group, "lspg_mean_qoi_error"),
                "mgp_lspg_qoi_gap_vs_pod": _mean_summary(group, "mgp_lspg_mean_qoi_error")
                - _mean_summary(group, "pod_mean_qoi_error"),
                "mgp_lspg_qoi_gap_vs_lspg": _mean_summary(group, "mgp_lspg_mean_qoi_error")
                - _mean_summary(group, "lspg_mean_qoi_error"),
            }
        )
    return sorted(
        aggregated,
        key=lambda row: (row["preset_name"], row["test_regime"], row["architecture_name"]),
    )


def pick_reference_architecture_decisions(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["preset_name"], row["test_regime"])
        groups.setdefault(key, []).append(row)

    decisions = []
    for (preset_name, test_regime), group in sorted(groups.items()):
        current_default = get_benchmark_preset(preset_name).architecture_name
        best_error = min(group, key=lambda row: row["mgp_mean_error_l2"])
        best_residual = min(group, key=lambda row: row["mgp_mean_residual"])
        best_qoi = min(group, key=lambda row: row["mgp_mean_qoi_error"])
        if best_error["architecture_name"] == best_residual["architecture_name"]:
            status = "resolved"
            recommended_architecture = best_error["architecture_name"]
            rationale = "same architecture wins both MGP error and residual"
        else:
            status = "unresolved"
            recommended_architecture = current_default
            rationale = "error and residual winners differ; keep current preset default"
        decisions.append(
            {
                "preset_name": preset_name,
                "problem_name": best_error["problem_name"],
                "test_regime": test_regime,
                "current_default_architecture": current_default,
                "best_by_error": best_error["architecture_name"],
                "best_by_residual": best_residual["architecture_name"],
                "best_by_qoi": best_qoi["architecture_name"],
                "recommended_architecture": recommended_architecture,
                "status": status,
                "rationale": rationale,
                "default_matches_recommendation": current_default == recommended_architecture,
            }
        )
    return decisions


def format_reference_architecture_study_markdown(payload: dict) -> str:
    lines = [
        "# Reference Architecture Study",
        "",
        "## Decisions",
        "",
    ]
    for decision in payload["decisions"]:
        lines.append(
            "- preset={preset_name}, problem={problem_name}, regime={test_regime}, status={status}, current_default={current_default_architecture}, recommended={recommended_architecture}, best_error={best_by_error}, best_residual={best_by_residual}, best_qoi={best_by_qoi}, rationale={rationale}".format(
                **decision
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
            "- preset={preset_name}, problem={problem_name}, regime={test_regime}, architecture={architecture_name}, runs={num_runs}, publication_mode={deterministic_publication_mode}, threads={publication_num_threads}, projected_weight={projected_residual_penalty_weight:.4f}, ambient_weight={ambient_residual_penalty_weight:.4f}, mgp_mean_error_l2={mgp_mean_error_l2:.6f}, mgp_std_error_l2={mgp_std_error_l2:.6f}, mgp_mean_residual={mgp_mean_residual:.6f}, mgp_mean_projected_residual={mgp_mean_projected_residual}, mgp_lspg_mean_error_l2={mgp_lspg_mean_error_l2:.6f}, mgp_lspg_mean_residual={mgp_lspg_mean_residual:.6f}, mgp_lspg_mean_projected_residual={mgp_lspg_mean_projected_residual}, reconstruction_mean_residual={reconstruction_mean_residual:.6f}, reconstruction_mean_projected_residual={reconstruction_mean_projected_residual}, mgp_std_residual={mgp_std_residual:.6f}, mgp_lspg_std_residual={mgp_lspg_std_residual:.6f}, mgp_mean_qoi_error={mgp_mean_qoi_error:.6f}, mgp_lspg_mean_qoi_error={mgp_lspg_mean_qoi_error:.6f}, error_gap_vs_lspg={mgp_error_gap_vs_lspg:.6f}, residual_gap_vs_lspg={mgp_residual_gap_vs_lspg:.6f}, mgp_lspg_error_gap_vs_lspg={mgp_lspg_error_gap_vs_lspg:.6f}, mgp_lspg_residual_gap_vs_lspg={mgp_lspg_residual_gap_vs_lspg:.6f}, mgp_speedup_vs_full={mgp_speedup_vs_full:.3f}, mgp_lspg_speedup_vs_full={mgp_lspg_speedup_vs_full:.3f}".format(
                **format_row
            )
        )
    return "\n".join(lines) + "\n"


def _mean_summary(group: list[dict], key: str) -> float:
    return float(sum(run["summary"][key] for run in group) / len(group))


def _median_summary(group: list[dict], key: str) -> float:
    values = [run["summary"][key] for run in group]
    return float(statistics.median(values))


def _pstdev_summary(group: list[dict], key: str) -> float:
    values = [run["summary"][key] for run in group]
    if len(values) == 1:
        return 0.0
    return float(statistics.pstdev(values))


def _effective_projected_weight(config: dict) -> float:
    projected_weight = config.get("projected_residual_penalty_weight")
    if projected_weight is not None:
        return float(projected_weight)
    return float(config["residual_penalty_weight"])


def _format_scientific(value: float) -> str:
    return f"{value:.3e}"
