from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import statistics

from .benchmark import evaluate_benchmark_from_offline_artifact, load_or_build_offline_artifact
from .presets import get_benchmark_preset


@dataclass
class SolverTradeoffStudyConfig:
    preset_names: tuple[str, ...]
    test_regimes: tuple[str, ...] = ("in_domain", "ood")
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    train_size: int | None = None
    test_size: int | None = None
    epochs: int | None = None
    deterministic_publication_mode: bool = False
    publication_num_threads: int = 1
    offline_cache_dir: str = "artifacts/cache"
    use_offline_cache: bool = True
    refresh_offline_cache: bool = False


def build_solver_tradeoff_benchmark_config(
    preset_name: str,
    *,
    test_regime: str,
    seed: int,
    train_size: int | None = None,
    test_size: int | None = None,
    epochs: int | None = None,
    deterministic_publication_mode: bool = False,
    publication_num_threads: int = 1,
    offline_cache_dir: str = "artifacts/cache",
    use_offline_cache: bool = True,
    refresh_offline_cache: bool = False,
    projected_residual_penalty_weight: float | None = None,
    ambient_residual_penalty_weight: float | None = None,
):
    base_config = get_benchmark_preset(preset_name)
    return replace(
        base_config,
        test_regime=test_regime,
        seed=seed,
        train_size=train_size if train_size is not None else base_config.train_size,
        test_size=test_size if test_size is not None else base_config.test_size,
        autoencoder_epochs=epochs if epochs is not None else base_config.autoencoder_epochs,
        deterministic_publication_mode=deterministic_publication_mode,
        publication_num_threads=publication_num_threads,
        offline_cache_dir=offline_cache_dir,
        use_offline_cache=use_offline_cache,
        refresh_offline_cache=refresh_offline_cache,
        projected_residual_penalty_weight=projected_residual_penalty_weight,
        ambient_residual_penalty_weight=ambient_residual_penalty_weight
        if ambient_residual_penalty_weight is not None
        else base_config.ambient_residual_penalty_weight,
    )


def run_solver_tradeoff_case(preset_name: str, benchmark_config) -> dict:
    artifact, cache_info = load_or_build_offline_artifact(benchmark_config)
    result = evaluate_benchmark_from_offline_artifact(artifact, benchmark_config)
    return {
        "preset_name": preset_name,
        "problem_name": benchmark_config.problem_name,
        "test_regime": benchmark_config.test_regime,
        "architecture_name": benchmark_config.architecture_name,
        "seed": benchmark_config.seed,
        "config": asdict(benchmark_config),
        "metadata": result["metadata"],
        "offline_artifact": {
            **result.get("offline_artifact", {}),
            **cache_info,
        },
        "summary": result["summary"],
        "qoi_name": result["qoi_name"],
    }


def build_solver_tradeoff_payload(config: dict, runs: list[dict]) -> dict:
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
        },
        "runs": runs,
        "seedwise_solver_rows": build_seedwise_solver_rows(runs),
        "aggregated": aggregate_solver_tradeoff_runs(runs),
    }


def run_solver_tradeoff_study(config: SolverTradeoffStudyConfig) -> dict:
    runs = []
    for preset_name in config.preset_names:
        for test_regime in config.test_regimes:
            for seed in config.seeds:
                benchmark_config = build_solver_tradeoff_benchmark_config(
                    preset_name,
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
                runs.append(run_solver_tradeoff_case(preset_name, benchmark_config))

    return build_solver_tradeoff_payload(asdict(config), runs)


def aggregate_solver_tradeoff_runs(runs: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        key = (run["preset_name"], run["test_regime"])
        grouped.setdefault(key, []).append(run)

    rows = []
    for (preset_name, test_regime), group in sorted(grouped.items()):
        first = group[0]
        rows.append(
            {
                "preset_name": preset_name,
                "problem_name": first["problem_name"],
                "test_regime": test_regime,
                "architecture_name": first["architecture_name"],
                "num_runs": len(group),
                "qoi_name": first["qoi_name"],
                "deterministic_publication_mode": first["metadata"]["deterministic_publication_mode"],
                "publication_num_threads": first["metadata"]["publication_num_threads_requested"],
                "offline_cache_dir": first["config"]["offline_cache_dir"],
                "offline_cache_hits": sum(1 for run in group if run["offline_artifact"]["cache_hit"]),
                "mgp_mean_error_l2": _mean_summary(group, "mgp_mean_error_l2"),
                "mgp_lspg_mean_error_l2": _mean_summary(group, "mgp_lspg_mean_error_l2"),
                "mgp_std_error_l2": _pstdev_summary(group, "mgp_mean_error_l2"),
                "mgp_lspg_std_error_l2": _pstdev_summary(group, "mgp_lspg_mean_error_l2"),
                "mgp_mean_residual": _mean_summary(group, "mgp_mean_residual"),
                "mgp_lspg_mean_residual": _mean_summary(group, "mgp_lspg_mean_residual"),
                "mgp_mean_projected_residual": _mean_summary(group, "mgp_mean_projected_residual"),
                "mgp_lspg_mean_projected_residual": _mean_summary(
                    group, "mgp_lspg_mean_projected_residual"
                ),
                "reconstruction_mean_error_l2": _mean_summary(group, "reconstruction_mean_error_l2"),
                "reconstruction_mean_residual": _mean_summary(group, "reconstruction_mean_residual"),
                "reconstruction_mean_projected_residual": _mean_summary(
                    group, "reconstruction_mean_projected_residual"
                ),
                "mgp_mean_qoi_error": _mean_summary(group, "mgp_mean_qoi_error"),
                "mgp_lspg_mean_qoi_error": _mean_summary(group, "mgp_lspg_mean_qoi_error"),
                "mgp_convergence_rate": _mean_summary(group, "mgp_convergence_rate"),
                "mgp_lspg_convergence_rate": _mean_summary(group, "mgp_lspg_convergence_rate"),
                "mean_full_time_sec": _mean_summary(group, "mean_full_time_sec"),
                "mean_mgp_time_sec": _mean_summary(group, "mean_mgp_time_sec"),
                "mean_mgp_lspg_time_sec": _mean_summary(group, "mean_mgp_lspg_time_sec"),
                "mean_pod_time_sec": _mean_summary(group, "mean_pod_time_sec"),
                "mean_lspg_time_sec": _mean_summary(group, "mean_lspg_time_sec"),
                "mgp_speedup_vs_full": _mean_summary(group, "mean_full_time_sec")
                / max(_mean_summary(group, "mean_mgp_time_sec"), 1e-12),
                "mgp_lspg_speedup_vs_full": _mean_summary(group, "mean_full_time_sec")
                / max(_mean_summary(group, "mean_mgp_lspg_time_sec"), 1e-12),
                "lspg_mean_error_l2": _mean_summary(group, "lspg_mean_error_l2"),
                "lspg_mean_residual": _mean_summary(group, "lspg_mean_residual"),
                "lspg_mean_qoi_error": _mean_summary(group, "lspg_mean_qoi_error"),
                "lspg_error_ci95_low": _percentile_summary(group, "lspg_mean_error_l2", 2.5),
                "lspg_error_ci95_high": _percentile_summary(group, "lspg_mean_error_l2", 97.5),
                "lspg_residual_ci95_low": _percentile_summary(group, "lspg_mean_residual", 2.5),
                "lspg_residual_ci95_high": _percentile_summary(group, "lspg_mean_residual", 97.5),
                "lspg_qoi_ci95_low": _percentile_summary(group, "lspg_mean_qoi_error", 2.5),
                "lspg_qoi_ci95_high": _percentile_summary(group, "lspg_mean_qoi_error", 97.5),
                "mgp_error_ci95_low": _percentile_summary(group, "mgp_mean_error_l2", 2.5),
                "mgp_error_ci95_high": _percentile_summary(group, "mgp_mean_error_l2", 97.5),
                "mgp_residual_ci95_low": _percentile_summary(group, "mgp_mean_residual", 2.5),
                "mgp_residual_ci95_high": _percentile_summary(group, "mgp_mean_residual", 97.5),
                "mgp_qoi_ci95_low": _percentile_summary(group, "mgp_mean_qoi_error", 2.5),
                "mgp_qoi_ci95_high": _percentile_summary(group, "mgp_mean_qoi_error", 97.5),
                "mgp_lspg_error_ci95_low": _percentile_summary(group, "mgp_lspg_mean_error_l2", 2.5),
                "mgp_lspg_error_ci95_high": _percentile_summary(group, "mgp_lspg_mean_error_l2", 97.5),
                "mgp_lspg_residual_ci95_low": _percentile_summary(
                    group, "mgp_lspg_mean_residual", 2.5
                ),
                "mgp_lspg_residual_ci95_high": _percentile_summary(
                    group, "mgp_lspg_mean_residual", 97.5
                ),
                "mgp_lspg_qoi_ci95_low": _percentile_summary(group, "mgp_lspg_mean_qoi_error", 2.5),
                "mgp_lspg_qoi_ci95_high": _percentile_summary(group, "mgp_lspg_mean_qoi_error", 97.5),
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
                "mgp_error_wins_vs_lspg": _seedwise_win_count(group, "mgp_mean_error_l2", "lspg_mean_error_l2"),
                "mgp_qoi_wins_vs_lspg": _seedwise_win_count(group, "mgp_mean_qoi_error", "lspg_mean_qoi_error"),
                "mgp_lspg_error_wins_vs_lspg": _seedwise_win_count(
                    group, "mgp_lspg_mean_error_l2", "lspg_mean_error_l2"
                ),
                "mgp_lspg_qoi_wins_vs_lspg": _seedwise_win_count(
                    group, "mgp_lspg_mean_qoi_error", "lspg_mean_qoi_error"
                ),
                "total_case_count": _total_case_count(group),
                "mgp_failed_case_count": _failed_case_count(group, "mgp_converged"),
                "mgp_lspg_failed_case_count": _failed_case_count(group, "mgp_lspg_converged"),
                "pod_failed_case_count": _failed_case_count(group, "pod_converged"),
                "lspg_failed_case_count": _failed_case_count(group, "lspg_converged"),
                "mgp_failed_seed_count": _failed_seed_count(group, "mgp_convergence_rate"),
                "mgp_lspg_failed_seed_count": _failed_seed_count(group, "mgp_lspg_convergence_rate"),
                "pod_failed_seed_count": _failed_seed_count(group, "pod_convergence_rate"),
                "lspg_failed_seed_count": _failed_seed_count(group, "lspg_convergence_rate"),
                "solver_recommendation": pick_solver_recommendation(
                    {
                        "mgp_error_gap_vs_lspg": _mean_summary(group, "mgp_mean_error_l2")
                        - _mean_summary(group, "lspg_mean_error_l2"),
                        "mgp_residual_gap_vs_lspg": _mean_summary(group, "mgp_mean_residual")
                        - _mean_summary(group, "lspg_mean_residual"),
                        "mgp_lspg_error_gap_vs_lspg": _mean_summary(group, "mgp_lspg_mean_error_l2")
                        - _mean_summary(group, "lspg_mean_error_l2"),
                        "mgp_lspg_residual_gap_vs_lspg": _mean_summary(group, "mgp_lspg_mean_residual")
                        - _mean_summary(group, "lspg_mean_residual"),
                    }
                ),
            }
        )
    return rows


def build_seedwise_solver_rows(runs: list[dict]) -> list[dict]:
    rows = []
    for run in sorted(runs, key=lambda item: (item["preset_name"], item["test_regime"], item["seed"])):
        summary = run["summary"]
        common = {
            "preset_name": run["preset_name"],
            "problem_name": run["problem_name"],
            "test_regime": run["test_regime"],
            "architecture_name": run["architecture_name"],
            "seed": run["seed"],
            "qoi_name": run["qoi_name"],
        }
        rows.append(
            {
                **common,
                "solver_variant": "manifold_galerkin",
                "mean_error_l2": summary["mgp_mean_error_l2"],
                "mean_residual": summary["mgp_mean_residual"],
                "mean_projected_residual": summary["mgp_mean_projected_residual"],
                "mean_qoi_error": summary["mgp_mean_qoi_error"],
                "convergence_rate": summary["mgp_convergence_rate"],
                "speedup_vs_full": summary["mgp_speedup_vs_full"],
                "mean_time_sec": summary["mean_mgp_time_sec"],
                "error_gap_vs_lspg": summary["mgp_mean_error_l2"] - summary["lspg_mean_error_l2"],
                "residual_gap_vs_lspg": summary["mgp_mean_residual"] - summary["lspg_mean_residual"],
                "qoi_gap_vs_lspg": summary["mgp_mean_qoi_error"] - summary["lspg_mean_qoi_error"],
            }
        )
        rows.append(
            {
                **common,
                "solver_variant": "manifold_lspg",
                "mean_error_l2": summary["mgp_lspg_mean_error_l2"],
                "mean_residual": summary["mgp_lspg_mean_residual"],
                "mean_projected_residual": summary["mgp_lspg_mean_projected_residual"],
                "mean_qoi_error": summary["mgp_lspg_mean_qoi_error"],
                "convergence_rate": summary["mgp_lspg_convergence_rate"],
                "speedup_vs_full": summary["mgp_lspg_speedup_vs_full"],
                "mean_time_sec": summary["mean_mgp_lspg_time_sec"],
                "error_gap_vs_lspg": summary["mgp_lspg_mean_error_l2"] - summary["lspg_mean_error_l2"],
                "residual_gap_vs_lspg": summary["mgp_lspg_mean_residual"] - summary["lspg_mean_residual"],
                "qoi_gap_vs_lspg": summary["mgp_lspg_mean_qoi_error"] - summary["lspg_mean_qoi_error"],
            }
        )
        rows.append(
            {
                **common,
                "solver_variant": "pod_lspg",
                "mean_error_l2": summary["lspg_mean_error_l2"],
                "mean_residual": summary["lspg_mean_residual"],
                "mean_projected_residual": 0.0,
                "mean_qoi_error": summary["lspg_mean_qoi_error"],
                "convergence_rate": summary["lspg_convergence_rate"],
                "speedup_vs_full": summary["lspg_speedup_vs_full"],
                "mean_time_sec": summary["mean_lspg_time_sec"],
                "error_gap_vs_lspg": 0.0,
                "residual_gap_vs_lspg": 0.0,
                "qoi_gap_vs_lspg": 0.0,
            }
        )
    return rows


def pick_solver_recommendation(row: dict, *, error_gap_tolerance: float = 0.02) -> str:
    residual_gap_improved = row["mgp_lspg_residual_gap_vs_lspg"] < row["mgp_residual_gap_vs_lspg"]
    error_gap_materially_worse = (
        row["mgp_lspg_error_gap_vs_lspg"] - row["mgp_error_gap_vs_lspg"]
    ) > (error_gap_tolerance + 1e-7)
    if residual_gap_improved and not error_gap_materially_worse:
        return "prefer_manifold_lspg"
    return "retain_manifold_galerkin"


def format_solver_tradeoff_study_markdown(payload: dict) -> str:
    lines = [
        "# Solver Tradeoff Study",
        "",
        "## Aggregated Results",
        "",
    ]
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
            "- preset={preset_name}, problem={problem_name}, regime={test_regime}, architecture={architecture_name}, runs={num_runs}, publication_mode={deterministic_publication_mode}, threads={publication_num_threads}, cache_hits={offline_cache_hits}, solver_recommendation={solver_recommendation}, mgp_mean_error_l2={mgp_mean_error_l2:.6f}, mgp_error_ci95=[{mgp_error_ci95_low:.6f}, {mgp_error_ci95_high:.6f}], mgp_mean_residual={mgp_mean_residual:.6f}, mgp_residual_ci95=[{mgp_residual_ci95_low:.6f}, {mgp_residual_ci95_high:.6f}], mgp_mean_projected_residual={mgp_mean_projected_residual}, mgp_lspg_mean_error_l2={mgp_lspg_mean_error_l2:.6f}, mgp_lspg_error_ci95=[{mgp_lspg_error_ci95_low:.6f}, {mgp_lspg_error_ci95_high:.6f}], mgp_lspg_mean_residual={mgp_lspg_mean_residual:.6f}, mgp_lspg_residual_ci95=[{mgp_lspg_residual_ci95_low:.6f}, {mgp_lspg_residual_ci95_high:.6f}], mgp_lspg_mean_projected_residual={mgp_lspg_mean_projected_residual}, reconstruction_mean_residual={reconstruction_mean_residual:.6f}, reconstruction_mean_projected_residual={reconstruction_mean_projected_residual}, mgp_mean_qoi_error={mgp_mean_qoi_error:.6f}, mgp_qoi_ci95=[{mgp_qoi_ci95_low:.6f}, {mgp_qoi_ci95_high:.6f}], mgp_lspg_mean_qoi_error={mgp_lspg_mean_qoi_error:.6f}, mgp_lspg_qoi_ci95=[{mgp_lspg_qoi_ci95_low:.6f}, {mgp_lspg_qoi_ci95_high:.6f}], mgp_error_gap_vs_lspg={mgp_error_gap_vs_lspg:.6f}, mgp_residual_gap_vs_lspg={mgp_residual_gap_vs_lspg:.6f}, mgp_lspg_error_gap_vs_lspg={mgp_lspg_error_gap_vs_lspg:.6f}, mgp_lspg_residual_gap_vs_lspg={mgp_lspg_residual_gap_vs_lspg:.6f}, mgp_error_wins_vs_lspg={mgp_error_wins_vs_lspg}, mgp_qoi_wins_vs_lspg={mgp_qoi_wins_vs_lspg}, mgp_lspg_error_wins_vs_lspg={mgp_lspg_error_wins_vs_lspg}, mgp_lspg_qoi_wins_vs_lspg={mgp_lspg_qoi_wins_vs_lspg}, mgp_failed_case_count={mgp_failed_case_count}, mgp_lspg_failed_case_count={mgp_lspg_failed_case_count}, lspg_failed_case_count={lspg_failed_case_count}, mgp_speedup_vs_full={mgp_speedup_vs_full:.3f}, mgp_lspg_speedup_vs_full={mgp_lspg_speedup_vs_full:.3f}, mean_full_time_sec={mean_full_time_sec:.6f}, mean_mgp_time_sec={mean_mgp_time_sec:.6f}, mean_mgp_lspg_time_sec={mean_mgp_lspg_time_sec:.6f}, mean_lspg_time_sec={mean_lspg_time_sec:.6f}".format(
                **format_row
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


def _percentile_summary(group: list[dict], key: str, percentile: float) -> float:
    values = [run["summary"][key] for run in group]
    return float(_percentile(values, percentile))


def _seedwise_win_count(group: list[dict], left_key: str, right_key: str) -> int:
    return sum(1 for run in group if run["summary"][left_key] < run["summary"][right_key])


def _failed_case_count(group: list[dict], converged_key: str) -> int:
    if all("cases" in run for run in group):
        return sum(1 for run in group for case in run["cases"] if not case[converged_key])
    summary_key = converged_key.replace("_converged", "_convergence_rate")
    total_failures = 0
    for run in group:
        test_size = int(run["config"]["test_size"])
        success_count = int(round(run["summary"][summary_key] * test_size))
        total_failures += max(test_size - success_count, 0)
    return total_failures


def _failed_seed_count(group: list[dict], convergence_rate_key: str) -> int:
    return sum(1 for run in group if run["summary"][convergence_rate_key] < 0.999999)


def _total_case_count(group: list[dict]) -> int:
    if all("cases" in run for run in group):
        return sum(len(run["cases"]) for run in group)
    return sum(int(run["config"]["test_size"]) for run in group)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    values = sorted(float(value) for value in values)
    position = (len(values) - 1) * (percentile / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _format_scientific(value: float) -> str:
    return f"{value:.3e}"
