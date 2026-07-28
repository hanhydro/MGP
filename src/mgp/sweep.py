from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math

from .benchmark import BenchmarkConfig, run_benchmark


@dataclass
class SweepConfig:
    problem_names: tuple[str, ...] = ("nonlinear_diffusion",)
    test_regimes: tuple[str, ...] = ("in_domain",)
    latent_dims: tuple[int, ...] = (2, 3, 4, 5)
    pod_dims: tuple[int, ...] | None = None
    seeds: tuple[int, ...] = (0, 1, 2)
    train_size: int = 24
    test_size: int = 8
    epochs: int = 250
    architecture_names: tuple[str, ...] = ("mlp",)
    hidden_dims_options: tuple[tuple[int, ...], ...] = ((48, 48),)
    decoder_hidden_dims_options: tuple[tuple[int, ...], ...] = ((),)
    activation_names: tuple[str, ...] = ("tanh",)
    smoothness_weights: tuple[float, ...] = (0.0,)
    residual_penalty_weights: tuple[float, ...] = (0.02,)
    projected_residual_penalty_weights: tuple[float, ...] | None = None
    ambient_residual_penalty_weights: tuple[float, ...] = (0.005,)
    residual_penalty_schedules: tuple[str, ...] = ("constant",)
    residual_penalty_metrics: tuple[str, ...] = ("l2_norm",)
    training_objective_modes: tuple[str, ...] = ("standard",)
    online_residual_sample_counts: tuple[int, ...] = (0,)
    online_residual_sample_scales: tuple[float, ...] = (0.05,)
    warm_start_strategies: tuple[str, ...] = ("nearest_train",)
    deterministic_publication_mode: bool = False
    publication_num_threads: int = 1


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def parse_hidden_dims_options(raw: str) -> tuple[tuple[int, ...], ...]:
    options = []
    for block in raw.split(","):
        block = block.strip()
        if not block:
            continue
        if block.lower() == "auto":
            options.append(())
            continue
        options.append(tuple(int(part.strip()) for part in block.split("x") if part.strip()))
    return tuple(options)


def parse_float_tuple(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def parse_str_tuple(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _objective_option_is_valid(
    training_objective_mode: str,
    online_residual_sample_count: int,
    residual_penalty_schedule: str,
) -> bool:
    online_mode = training_objective_mode in {"staged_online_residual", "staged_online_projected"}
    if online_mode != (online_residual_sample_count > 0):
        return False
    if training_objective_mode.startswith("staged") and residual_penalty_schedule not in {
        "staged_recon_to_residual",
        "delayed_linear_ramp",
    }:
        return False
    return True


def run_sweep(config: SweepConfig) -> dict:
    projected_residual_penalty_weights = (
        config.projected_residual_penalty_weights
        if config.projected_residual_penalty_weights is not None
        else config.residual_penalty_weights
    )
    pod_dims = config.pod_dims if config.pod_dims is not None else config.latent_dims
    runs = []
    for problem_name in config.problem_names:
        for test_regime in config.test_regimes:
            for latent_dim in config.latent_dims:
                pod_dim_options = pod_dims if config.pod_dims is not None else (latent_dim,)
                for pod_dim in pod_dim_options:
                    for seed in config.seeds:
                        for architecture_name in config.architecture_names:
                            for hidden_dims in config.hidden_dims_options:
                                for decoder_hidden_dims in config.decoder_hidden_dims_options:
                                    for activation_name in config.activation_names:
                                        for smoothness_weight in config.smoothness_weights:
                                            for projected_residual_penalty_weight in projected_residual_penalty_weights:
                                                for ambient_residual_penalty_weight in config.ambient_residual_penalty_weights:
                                                    for residual_penalty_schedule in config.residual_penalty_schedules:
                                                        for residual_penalty_metric in config.residual_penalty_metrics:
                                                            for training_objective_mode in config.training_objective_modes:
                                                                for online_residual_sample_count in config.online_residual_sample_counts:
                                                                    if not _objective_option_is_valid(
                                                                        training_objective_mode,
                                                                        online_residual_sample_count,
                                                                        residual_penalty_schedule,
                                                                    ):
                                                                        continue
                                                                    for online_residual_sample_scale in config.online_residual_sample_scales:
                                                                        for warm_start_strategy in config.warm_start_strategies:
                                                                            benchmark_config = BenchmarkConfig(
                                                                                train_size=config.train_size,
                                                                                test_size=config.test_size,
                                                                                problem_name=problem_name,
                                                                                test_regime=test_regime,
                                                                                latent_dim=latent_dim,
                                                                                pod_dim=pod_dim,
                                                                                architecture_name=architecture_name,
                                                                                hidden_dims=hidden_dims,
                                                                                decoder_hidden_dims=decoder_hidden_dims,
                                                                                activation_name=activation_name,
                                                                                autoencoder_epochs=config.epochs,
                                                                                smoothness_weight=smoothness_weight,
                                                                                residual_penalty_weight=projected_residual_penalty_weight,
                                                                                projected_residual_penalty_weight=projected_residual_penalty_weight,
                                                                                ambient_residual_penalty_weight=ambient_residual_penalty_weight,
                                                                                residual_penalty_schedule=residual_penalty_schedule,
                                                                                residual_penalty_metric=residual_penalty_metric,
                                                                                training_objective_mode=training_objective_mode,
                                                                                online_residual_sample_count=online_residual_sample_count,
                                                                                online_residual_sample_scale=online_residual_sample_scale,
                                                                                warm_start_strategy=warm_start_strategy,
                                                                                seed=seed,
                                                                                deterministic_publication_mode=config.deterministic_publication_mode,
                                                                                publication_num_threads=config.publication_num_threads,
                                                                            )
                                                                            result = run_benchmark(benchmark_config)
                                                                            runs.append(
                                                                                {
                                                                                    "problem_name": problem_name,
                                                                                    "test_regime": test_regime,
                                                                                    "latent_dim": latent_dim,
                                                                                    "pod_dim": pod_dim,
                                                                                    "seed": seed,
                                                                                    "architecture_name": architecture_name,
                                                                                    "hidden_dims": list(hidden_dims),
                                                                                    "decoder_hidden_dims": list(decoder_hidden_dims),
                                                                                    "activation_name": activation_name,
                                                                                    "smoothness_weight": smoothness_weight,
                                                                                    "residual_penalty_weight": projected_residual_penalty_weight,
                                                                                    "projected_residual_penalty_weight": projected_residual_penalty_weight,
                                                                                    "ambient_residual_penalty_weight": ambient_residual_penalty_weight,
                                                                                    "residual_penalty_schedule": residual_penalty_schedule,
                                                                                    "residual_penalty_metric": residual_penalty_metric,
                                                                                    "training_objective_mode": training_objective_mode,
                                                                                    "online_residual_sample_count": online_residual_sample_count,
                                                                                    "online_residual_sample_scale": online_residual_sample_scale,
                                                                                    "warm_start_strategy": warm_start_strategy,
                                                                                    "config": asdict(benchmark_config),
                                                                                    "metadata": result["metadata"],
                                                                                    "summary": result["summary"],
                                                                                }
                                                                            )

    aggregated = []
    for problem_name in config.problem_names:
        for test_regime in config.test_regimes:
            for latent_dim in config.latent_dims:
                pod_dim_options = pod_dims if config.pod_dims is not None else (latent_dim,)
                for pod_dim in pod_dim_options:
                    for architecture_name in config.architecture_names:
                        for hidden_dims in config.hidden_dims_options:
                            for decoder_hidden_dims in config.decoder_hidden_dims_options:
                                for activation_name in config.activation_names:
                                    for smoothness_weight in config.smoothness_weights:
                                        for projected_residual_penalty_weight in projected_residual_penalty_weights:
                                            for ambient_residual_penalty_weight in config.ambient_residual_penalty_weights:
                                                for residual_penalty_schedule in config.residual_penalty_schedules:
                                                    for residual_penalty_metric in config.residual_penalty_metrics:
                                                        for training_objective_mode in config.training_objective_modes:
                                                            for online_residual_sample_count in config.online_residual_sample_counts:
                                                                if not _objective_option_is_valid(
                                                                    training_objective_mode,
                                                                    online_residual_sample_count,
                                                                    residual_penalty_schedule,
                                                                ):
                                                                    continue
                                                                for online_residual_sample_scale in config.online_residual_sample_scales:
                                                                    for warm_start_strategy in config.warm_start_strategies:
                                                                        combo_runs = [
                                                                            run
                                                                            for run in runs
                                                                            if run["problem_name"] == problem_name
                                                                            and run["test_regime"] == test_regime
                                                                            and run["latent_dim"] == latent_dim
                                                                            and run["pod_dim"] == pod_dim
                                                                            and run["architecture_name"] == architecture_name
                                                                            and tuple(run["hidden_dims"]) == tuple(hidden_dims)
                                                                            and tuple(run["decoder_hidden_dims"]) == tuple(decoder_hidden_dims)
                                                                            and run["activation_name"] == activation_name
                                                                            and run["smoothness_weight"] == smoothness_weight
                                                                            and run["projected_residual_penalty_weight"]
                                                                            == projected_residual_penalty_weight
                                                                            and run["ambient_residual_penalty_weight"]
                                                                            == ambient_residual_penalty_weight
                                                                            and run["residual_penalty_schedule"] == residual_penalty_schedule
                                                                            and run["residual_penalty_metric"] == residual_penalty_metric
                                                                            and run["training_objective_mode"] == training_objective_mode
                                                                            and run["online_residual_sample_count"] == online_residual_sample_count
                                                                            and run["online_residual_sample_scale"] == online_residual_sample_scale
                                                                            and run["warm_start_strategy"] == warm_start_strategy
                                                                        ]
                                                                        aggregated.append(
                                                                            {
                                                                                "problem_name": problem_name,
                                                                                "test_regime": test_regime,
                                                                                "latent_dim": latent_dim,
                                                                                "pod_dim": pod_dim,
                                                                                "architecture_name": architecture_name,
                                                                                "hidden_dims": list(hidden_dims),
                                                                                "decoder_hidden_dims": list(decoder_hidden_dims),
                                                                                "activation_name": activation_name,
                                                                                "smoothness_weight": smoothness_weight,
                                                                                "residual_penalty_weight": projected_residual_penalty_weight,
                                                                                "projected_residual_penalty_weight": projected_residual_penalty_weight,
                                                                                "ambient_residual_penalty_weight": ambient_residual_penalty_weight,
                                                                                "residual_penalty_schedule": residual_penalty_schedule,
                                                                                "residual_penalty_metric": residual_penalty_metric,
                                                                                "training_objective_mode": training_objective_mode,
                                                                                "online_residual_sample_count": online_residual_sample_count,
                                                                                "online_residual_sample_scale": online_residual_sample_scale,
                                                                                "warm_start_strategy": warm_start_strategy,
                                                                    "num_runs": len(combo_runs),
                                                                    "mgp_mean_error_l2": _mean(combo_runs, "mgp_mean_error_l2"),
                                                                    "pod_mean_error_l2": _mean(combo_runs, "pod_mean_error_l2"),
                                                                    "lspg_mean_error_l2": _mean(combo_runs, "lspg_mean_error_l2"),
                                                                    "mgp_mean_residual": _mean(combo_runs, "mgp_mean_residual"),
                                                                    "mgp_mean_projected_residual": _mean(
                                                                        combo_runs, "mgp_mean_projected_residual"
                                                                    ),
                                                                    "reconstruction_mean_error_l2": _mean(
                                                                        combo_runs, "reconstruction_mean_error_l2"
                                                                    ),
                                                                    "reconstruction_mean_residual": _mean(
                                                                        combo_runs, "reconstruction_mean_residual"
                                                                    ),
                                                                    "reconstruction_mean_projected_residual": _mean(
                                                                        combo_runs,
                                                                        "reconstruction_mean_projected_residual",
                                                                    ),
                                                                    "pod_mean_residual": _mean(combo_runs, "pod_mean_residual"),
                                                                    "lspg_mean_residual": _mean(combo_runs, "lspg_mean_residual"),
                                                                    "mgp_mean_qoi_error": _mean(combo_runs, "mgp_mean_qoi_error"),
                                                                    "pod_mean_qoi_error": _mean(combo_runs, "pod_mean_qoi_error"),
                                                                    "lspg_mean_qoi_error": _mean(combo_runs, "lspg_mean_qoi_error"),
                                                                    "mgp_convergence_rate": _mean(combo_runs, "mgp_convergence_rate"),
                                                                    "pod_convergence_rate": _mean(combo_runs, "pod_convergence_rate"),
                                                                    "lspg_convergence_rate": _mean(combo_runs, "lspg_convergence_rate"),
                                                                    "mgp_mean_latent_distance": _mean(
                                                                        combo_runs, "mgp_mean_latent_distance"
                                                                    ),
                                                                    "mgp_mean_decoder_jacobian_fro": _mean(
                                                                        combo_runs, "mgp_mean_decoder_jacobian_fro"
                                                                    ),
                                                                    "mgp_mean_decoder_jacobian_condition": _mean(
                                                                        combo_runs, "mgp_mean_decoder_jacobian_condition"
                                                                    ),
                                                                    "deterministic_publication_mode": (
                                                                        combo_runs[0]["metadata"][
                                                                            "deterministic_publication_mode"
                                                                        ]
                                                                        if combo_runs
                                                                        else config.deterministic_publication_mode
                                                                    ),
                                                                    "publication_num_threads": (
                                                                        combo_runs[0]["metadata"][
                                                                            "publication_num_threads_requested"
                                                                        ]
                                                                        if combo_runs
                                                                        else config.publication_num_threads
                                                                    ),
                                                                    "mgp_speedup_vs_full": _mean(
                                                                        combo_runs, "mgp_speedup_vs_full"
                                                                    ),
                                                                    "pod_speedup_vs_full": _mean(
                                                                        combo_runs, "pod_speedup_vs_full"
                                                                    ),
                                                                    "lspg_speedup_vs_full": _mean(
                                                                        combo_runs, "lspg_speedup_vs_full"
                                                                    ),
                                                                    "mgp_error_gap_vs_pod": _mean(
                                                                        combo_runs, "mgp_mean_error_l2"
                                                                    )
                                                                    - _mean(combo_runs, "pod_mean_error_l2"),
                                                                    "mgp_error_gap_vs_lspg": _mean(
                                                                        combo_runs, "mgp_mean_error_l2"
                                                                    )
                                                                    - _mean(combo_runs, "lspg_mean_error_l2"),
                                                                    "mgp_residual_gap_vs_pod": _mean(
                                                                        combo_runs, "mgp_mean_residual"
                                                                    )
                                                                    - _mean(combo_runs, "pod_mean_residual"),
                                                                    "mgp_residual_gap_vs_lspg": _mean(
                                                                        combo_runs, "mgp_mean_residual"
                                                                    )
                                                                    - _mean(combo_runs, "lspg_mean_residual"),
                                                                }
                                                            )

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
    }


def rank_aggregated_rows(rows: list[dict], *, top_k: int = 3) -> dict[str, list[dict]]:
    valid_rows = [
        row
        for row in rows
        if not math.isnan(row["mgp_mean_error_l2"]) and not math.isnan(row["mgp_mean_residual"])
    ]
    by_error = sorted(valid_rows, key=lambda row: row["mgp_mean_error_l2"])[:top_k]
    by_residual = sorted(valid_rows, key=lambda row: row["mgp_mean_residual"])[:top_k]
    by_tradeoff = sorted(
        valid_rows,
        key=lambda row: row["mgp_mean_error_l2"] * row["mgp_mean_residual"],
    )[:top_k]
    by_competitive_error = sorted(
        valid_rows,
        key=lambda row: (row["mgp_error_gap_vs_pod"], row["mgp_error_gap_vs_lspg"]),
    )[:top_k]
    by_competitive_residual = sorted(
        valid_rows,
        key=lambda row: (row["mgp_residual_gap_vs_pod"], row["mgp_residual_gap_vs_lspg"]),
    )[:top_k]
    return {
        "best_by_error": by_error,
        "best_by_residual": by_residual,
        "best_by_tradeoff": by_tradeoff,
        "best_by_competitive_error_gap": by_competitive_error,
        "best_by_competitive_residual_gap": by_competitive_residual,
    }


def _mean(runs: list[dict], key: str) -> float:
    if not runs:
        return float("nan")
    return float(sum(run["summary"][key] for run in runs) / len(runs))
