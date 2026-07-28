from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone

from .benchmark import BenchmarkConfig, evaluate_benchmark_from_offline_artifact, load_or_build_offline_artifact
from .presets import get_benchmark_preset


@dataclass
class DimensionRobustnessStudyConfig:
    preset_names: tuple[str, ...]
    test_regimes: tuple[str, ...] = ("in_domain", "ood")
    latent_dims: tuple[int, ...] = (2,)
    pod_dims: tuple[int, ...] = (2, 4, 8)
    seeds: tuple[int, ...] = (0, 1, 2)
    train_size: int | None = None
    test_size: int | None = None
    epochs: int | None = None
    deterministic_publication_mode: bool = False
    publication_num_threads: int = 1
    offline_cache_dir: str = "artifacts/cache"
    use_offline_cache: bool = True
    refresh_offline_cache: bool = False
    projected_residual_penalty_weight: float | None = None
    ambient_residual_penalty_weight: float | None = None


def _build_case_audit(result: dict) -> list[dict]:
    audits = []
    for case_index, case in enumerate(result.get("cases", [])):
        audits.append(
            {
                "case_index": case_index,
                "mu": list(case["mu"]),
                "full_qoi": case["full_qoi"],
                "mgp_qoi": case["mgp_qoi"],
                "mgp_qoi_error": case["mgp_qoi_error"],
                "mgp_error_l2": case["mgp_error_l2"],
                "mgp_residual": case["mgp_residual"],
                "mgp_converged": case["mgp_converged"],
                "mgp_lspg_qoi": case["mgp_lspg_qoi"],
                "mgp_lspg_qoi_error": case["mgp_lspg_qoi_error"],
                "mgp_lspg_error_l2": case["mgp_lspg_error_l2"],
                "mgp_lspg_residual": case["mgp_lspg_residual"],
                "mgp_lspg_converged": case["mgp_lspg_converged"],
                "pod_qoi": case["pod_qoi"],
                "pod_qoi_error": case["pod_qoi_error"],
                "pod_error_l2": case["pod_error_l2"],
                "pod_residual": case["pod_residual"],
                "pod_converged": case["pod_converged"],
                "lspg_qoi": case["lspg_qoi"],
                "lspg_qoi_error": case["lspg_qoi_error"],
                "lspg_error_l2": case["lspg_error_l2"],
                "lspg_residual": case["lspg_residual"],
                "lspg_converged": case["lspg_converged"],
            }
        )
    return audits


def build_dimension_robustness_benchmark_config(
    preset_name: str,
    *,
    test_regime: str,
    latent_dim: int,
    pod_dim: int,
    seed: int,
    train_size: int | None = None,
    test_size: int | None = None,
    epochs: int | None = None,
    architecture_name: str | None = None,
    hidden_dims: tuple[int, ...] | None = None,
    decoder_hidden_dims: tuple[int, ...] | None = None,
    deterministic_publication_mode: bool = False,
    publication_num_threads: int = 1,
    offline_cache_dir: str = "artifacts/cache",
    use_offline_cache: bool = True,
    refresh_offline_cache: bool = False,
    projected_residual_penalty_weight: float | None = None,
    ambient_residual_penalty_weight: float | None = None,
) -> BenchmarkConfig:
    base_config = get_benchmark_preset(preset_name)
    return replace(
        base_config,
        test_regime=test_regime,
        latent_dim=latent_dim,
        pod_dim=pod_dim,
        seed=seed,
        train_size=train_size if train_size is not None else base_config.train_size,
        test_size=test_size if test_size is not None else base_config.test_size,
        autoencoder_epochs=epochs if epochs is not None else base_config.autoencoder_epochs,
        architecture_name=architecture_name if architecture_name is not None else base_config.architecture_name,
        hidden_dims=hidden_dims if hidden_dims is not None else base_config.hidden_dims,
        decoder_hidden_dims=(
            decoder_hidden_dims if decoder_hidden_dims is not None else base_config.decoder_hidden_dims
        ),
        deterministic_publication_mode=deterministic_publication_mode,
        publication_num_threads=publication_num_threads,
        offline_cache_dir=offline_cache_dir,
        use_offline_cache=use_offline_cache,
        refresh_offline_cache=refresh_offline_cache,
        projected_residual_penalty_weight=projected_residual_penalty_weight,
        ambient_residual_penalty_weight=(
            ambient_residual_penalty_weight
            if ambient_residual_penalty_weight is not None
            else base_config.ambient_residual_penalty_weight
        ),
    )


def run_dimension_robustness_case(
    preset_name: str,
    *,
    test_regime: str,
    latent_dim: int,
    pod_dims: tuple[int, ...],
    seed: int,
    train_size: int | None = None,
    test_size: int | None = None,
    epochs: int | None = None,
    architecture_name: str | None = None,
    hidden_dims: tuple[int, ...] | None = None,
    decoder_hidden_dims: tuple[int, ...] | None = None,
    deterministic_publication_mode: bool = False,
    publication_num_threads: int = 1,
    offline_cache_dir: str = "artifacts/cache",
    use_offline_cache: bool = True,
    refresh_offline_cache: bool = False,
    projected_residual_penalty_weight: float | None = None,
    ambient_residual_penalty_weight: float | None = None,
) -> list[dict]:
    if not pod_dims:
        raise ValueError("pod_dims must not be empty")

    benchmark_config = build_dimension_robustness_benchmark_config(
        preset_name,
        test_regime=test_regime,
        latent_dim=latent_dim,
        pod_dim=pod_dims[0],
        seed=seed,
        train_size=train_size,
        test_size=test_size,
        epochs=epochs,
        architecture_name=architecture_name,
        hidden_dims=hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        deterministic_publication_mode=deterministic_publication_mode,
        publication_num_threads=publication_num_threads,
        offline_cache_dir=offline_cache_dir,
        use_offline_cache=use_offline_cache,
        refresh_offline_cache=refresh_offline_cache,
        projected_residual_penalty_weight=projected_residual_penalty_weight,
        ambient_residual_penalty_weight=ambient_residual_penalty_weight,
    )
    artifact, cache_info = load_or_build_offline_artifact(benchmark_config)

    runs = []
    for pod_dim in pod_dims:
        current_config = replace(benchmark_config, pod_dim=pod_dim, refresh_offline_cache=False)
        result = evaluate_benchmark_from_offline_artifact(artifact, current_config)
        runs.append(
            {
                "preset_name": preset_name,
                "problem_name": current_config.problem_name,
                "test_regime": test_regime,
                "latent_dim": latent_dim,
                "pod_dim": pod_dim,
                "seed": seed,
                "config": asdict(current_config),
                "metadata": result["metadata"],
                "offline_artifact": {
                    **result.get("offline_artifact", {}),
                    **cache_info,
                },
                "summary": result["summary"],
                "qoi_name": result["qoi_name"],
                "case_audit": _build_case_audit(result),
            }
        )
    return runs


def run_dimension_robustness_study(config: DimensionRobustnessStudyConfig) -> dict:
    runs = []
    for preset_name in config.preset_names:
        for test_regime in config.test_regimes:
            for latent_dim in config.latent_dims:
                for seed in config.seeds:
                    runs.extend(
                        run_dimension_robustness_case(
                            preset_name,
                            test_regime=test_regime,
                            latent_dim=latent_dim,
                            pod_dims=config.pod_dims,
                            seed=seed,
                            train_size=config.train_size,
                            test_size=config.test_size,
                            epochs=config.epochs,
                            architecture_name=None,
                            hidden_dims=None,
                            decoder_hidden_dims=None,
                            deterministic_publication_mode=config.deterministic_publication_mode,
                            publication_num_threads=config.publication_num_threads,
                            offline_cache_dir=config.offline_cache_dir,
                            use_offline_cache=config.use_offline_cache,
                            refresh_offline_cache=config.refresh_offline_cache,
                            projected_residual_penalty_weight=config.projected_residual_penalty_weight,
                            ambient_residual_penalty_weight=config.ambient_residual_penalty_weight,
                        )
                    )

    return build_dimension_robustness_payload(asdict(config), runs)


def build_dimension_robustness_payload(config: dict, runs: list[dict]) -> dict:
    aggregated = aggregate_dimension_robustness_runs(runs)
    decisions = summarize_dimension_robustness(aggregated)
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
        "aggregated": aggregated,
        "decisions": decisions,
    }


def aggregate_dimension_robustness_runs(runs: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int, int], list[dict]] = {}
    for run in runs:
        key = (
            run["preset_name"],
            run["test_regime"],
            run["latent_dim"],
            run["pod_dim"],
        )
        grouped.setdefault(key, []).append(run)

    rows = []
    for key, group in sorted(grouped.items()):
        preset_name, test_regime, latent_dim, pod_dim = key
        first = group[0]
        rows.append(
            {
                "preset_name": preset_name,
                "problem_name": first["problem_name"],
                "test_regime": test_regime,
                "latent_dim": latent_dim,
                "pod_dim": pod_dim,
                "num_runs": len(group),
                "qoi_name": first["qoi_name"],
                "architecture_name": first["config"]["architecture_name"],
                "hidden_dims": list(first["config"]["hidden_dims"]),
                "decoder_hidden_dims": list(first["config"]["decoder_hidden_dims"]),
                "activation_name": first["config"]["activation_name"],
                "projected_residual_penalty_weight": first["config"]["projected_residual_penalty_weight"],
                "ambient_residual_penalty_weight": first["config"]["ambient_residual_penalty_weight"],
                "offline_cache_hits": sum(1 for run in group if run["offline_artifact"]["cache_hit"]),
                "mgp_mean_error_l2": _mean_summary(group, "mgp_mean_error_l2"),
                "lspg_mean_error_l2": _mean_summary(group, "lspg_mean_error_l2"),
                "pod_mean_error_l2": _mean_summary(group, "pod_mean_error_l2"),
                "mgp_mean_qoi_error": _mean_summary(group, "mgp_mean_qoi_error"),
                "lspg_mean_qoi_error": _mean_summary(group, "lspg_mean_qoi_error"),
                "pod_mean_qoi_error": _mean_summary(group, "pod_mean_qoi_error"),
                "mgp_mean_residual": _mean_summary(group, "mgp_mean_residual"),
                "lspg_mean_residual": _mean_summary(group, "lspg_mean_residual"),
                "pod_mean_residual": _mean_summary(group, "pod_mean_residual"),
                "mgp_convergence_rate": _mean_summary(group, "mgp_convergence_rate"),
                "lspg_convergence_rate": _mean_summary(group, "lspg_convergence_rate"),
                "pod_convergence_rate": _mean_summary(group, "pod_convergence_rate"),
                "mgp_speedup_vs_full": _mean_summary(group, "mgp_speedup_vs_full"),
                "lspg_speedup_vs_full": _mean_summary(group, "lspg_speedup_vs_full"),
                "pod_speedup_vs_full": _mean_summary(group, "pod_speedup_vs_full"),
                "mgp_error_gap_vs_lspg": _mean_summary(group, "mgp_mean_error_l2")
                - _mean_summary(group, "lspg_mean_error_l2"),
                "mgp_qoi_gap_vs_lspg": _mean_summary(group, "mgp_mean_qoi_error")
                - _mean_summary(group, "lspg_mean_qoi_error"),
                "mgp_residual_gap_vs_lspg": _mean_summary(group, "mgp_mean_residual")
                - _mean_summary(group, "lspg_mean_residual"),
                "mgp_error_gap_vs_pod": _mean_summary(group, "mgp_mean_error_l2")
                - _mean_summary(group, "pod_mean_error_l2"),
                "mgp_qoi_gap_vs_pod": _mean_summary(group, "mgp_mean_qoi_error")
                - _mean_summary(group, "pod_mean_qoi_error"),
                "mgp_residual_gap_vs_pod": _mean_summary(group, "mgp_mean_residual")
                - _mean_summary(group, "pod_mean_residual"),
            }
        )
    return rows


def summarize_dimension_robustness(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for row in rows:
        key = (row["preset_name"], row["test_regime"], row["latent_dim"])
        groups.setdefault(key, []).append(row)

    decisions = []
    for (preset_name, test_regime, latent_dim), group in sorted(groups.items()):
        best_lspg_state = min(group, key=lambda row: row["lspg_mean_error_l2"])
        best_lspg_qoi = min(group, key=lambda row: row["lspg_mean_qoi_error"])
        best_lspg_residual = min(group, key=lambda row: row["lspg_mean_residual"])
        state_advantage_pod_dims = sorted(
            row["pod_dim"] for row in group if row["mgp_error_gap_vs_lspg"] < 0.0
        )
        qoi_advantage_pod_dims = sorted(
            row["pod_dim"] for row in group if row["mgp_qoi_gap_vs_lspg"] < 0.0
        )
        residual_advantage_pod_dims = sorted(
            row["pod_dim"] for row in group if row["mgp_residual_gap_vs_lspg"] < 0.0
        )
        decisions.append(
            {
                "preset_name": preset_name,
                "problem_name": best_lspg_state["problem_name"],
                "test_regime": test_regime,
                "latent_dim": latent_dim,
                "mgp_architecture_name": best_lspg_state["architecture_name"],
                "best_lspg_state_pod_dim": best_lspg_state["pod_dim"],
                "best_lspg_state_error": best_lspg_state["lspg_mean_error_l2"],
                "mgp_state_error_at_best_lspg_state_dim": best_lspg_state["mgp_mean_error_l2"],
                "mgp_error_gap_vs_best_lspg": best_lspg_state["mgp_error_gap_vs_lspg"],
                "best_lspg_qoi_pod_dim": best_lspg_qoi["pod_dim"],
                "mgp_qoi_gap_vs_best_lspg": best_lspg_qoi["mgp_qoi_gap_vs_lspg"],
                "best_lspg_residual_pod_dim": best_lspg_residual["pod_dim"],
                "mgp_residual_gap_vs_best_lspg": best_lspg_residual["mgp_residual_gap_vs_lspg"],
                "state_advantage_pod_dims": state_advantage_pod_dims,
                "qoi_advantage_pod_dims": qoi_advantage_pod_dims,
                "residual_advantage_pod_dims": residual_advantage_pod_dims,
                "state_advantage_status": _advantage_status(state_advantage_pod_dims, group),
                "qoi_advantage_status": _advantage_status(qoi_advantage_pod_dims, group),
                "residual_advantage_status": _advantage_status(residual_advantage_pod_dims, group),
            }
        )
    return decisions


def format_dimension_robustness_markdown(payload: dict) -> str:
    lines = [
        "# Dimension Robustness Study",
        "",
        "This study holds each benchmark-family preset fixed while varying manifold latent dimension and POD/POD-LSPG reduced dimension independently.",
        "",
        "## Summary decisions",
        "",
    ]
    for decision in payload["decisions"]:
        lines.append(
            "- preset={preset_name}, problem={problem_name}, regime={test_regime}, latent_dim={latent_dim}, "
            "state_advantage={state_advantage_status} at pod_dims={state_advantage_pod_dims}, "
            "qoi_advantage={qoi_advantage_status} at pod_dims={qoi_advantage_pod_dims}, "
            "residual_advantage={residual_advantage_status} at pod_dims={residual_advantage_pod_dims}, "
            "best_lspg_state_pod_dim={best_lspg_state_pod_dim}, mgp_error_gap_vs_best_lspg={mgp_error_gap_vs_best_lspg:.6f}, "
            "best_lspg_residual_pod_dim={best_lspg_residual_pod_dim}, mgp_residual_gap_vs_best_lspg={mgp_residual_gap_vs_best_lspg:.6f}".format(
                preset_name=decision["preset_name"],
                problem_name=decision["problem_name"],
                test_regime=decision["test_regime"],
                latent_dim=decision["latent_dim"],
                state_advantage_status=decision["state_advantage_status"],
                state_advantage_pod_dims=_format_dim_list(decision["state_advantage_pod_dims"]),
                qoi_advantage_status=decision["qoi_advantage_status"],
                qoi_advantage_pod_dims=_format_dim_list(decision["qoi_advantage_pod_dims"]),
                residual_advantage_status=decision["residual_advantage_status"],
                residual_advantage_pod_dims=_format_dim_list(decision["residual_advantage_pod_dims"]),
                best_lspg_state_pod_dim=decision["best_lspg_state_pod_dim"],
                mgp_error_gap_vs_best_lspg=decision["mgp_error_gap_vs_best_lspg"],
                best_lspg_residual_pod_dim=decision["best_lspg_residual_pod_dim"],
                mgp_residual_gap_vs_best_lspg=decision["mgp_residual_gap_vs_best_lspg"],
            )
        )

    lines.extend(["", "## Aggregated rows", ""])
    for row in payload["aggregated"]:
        lines.append(
            "- preset={preset_name}, regime={test_regime}, latent_dim={latent_dim}, pod_dim={pod_dim}, "
            "mgp_error={mgp_mean_error_l2:.6f}, lspg_error={lspg_mean_error_l2:.6f}, "
            "mgp_qoi={mgp_mean_qoi_error:.6f}, lspg_qoi={lspg_mean_qoi_error:.6f}, "
            "mgp_residual={mgp_mean_residual:.6f}, lspg_residual={lspg_mean_residual:.6f}, "
            "gap_vs_lspg(error={mgp_error_gap_vs_lspg:.6f}, qoi={mgp_qoi_gap_vs_lspg:.6f}, residual={mgp_residual_gap_vs_lspg:.6f}), "
            "offline_cache_hits={offline_cache_hits}".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def _mean_summary(runs: list[dict], key: str) -> float:
    return float(sum(run["summary"][key] for run in runs) / len(runs))


def _advantage_status(winning_dims: list[int], group: list[dict]) -> str:
    if not winning_dims:
        return "none"
    if len(winning_dims) == len(group):
        return "all"
    return "partial"


def _format_dim_list(values: list[int]) -> str:
    return "none" if not values else ",".join(str(value) for value in values)
