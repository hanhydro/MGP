from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import statistics

from .benchmark import BenchmarkConfig, run_benchmark
from .presets import get_benchmark_preset


@dataclass(frozen=True)
class PresetTuningCandidate:
    architecture_name: str
    latent_dim: int
    hidden_dims: tuple[int, ...]
    decoder_hidden_dims: tuple[int, ...]

    @property
    def label(self) -> str:
        hidden = "x".join(str(value) for value in self.hidden_dims)
        decoder = "auto" if not self.decoder_hidden_dims else "x".join(
            str(value) for value in self.decoder_hidden_dims
        )
        return (
            f"{self.architecture_name}:r={self.latent_dim}:enc={hidden}:dec={decoder}"
        )


@dataclass
class PresetTuningStudyConfig:
    preset_name: str
    candidates: tuple[PresetTuningCandidate, ...]
    test_regimes: tuple[str, ...] = ("in_domain", "ood")
    seeds: tuple[int, ...] = (0, 1, 2)
    train_size: int | None = None
    test_size: int | None = None
    epochs: int | None = None
    projected_residual_penalty_weight: float | None = None
    ambient_residual_penalty_weight: float | None = None
    residual_tolerance_fraction: float = 0.1
    deterministic_publication_mode: bool = False
    publication_num_threads: int = 1


def run_preset_tuning_study(config: PresetTuningStudyConfig) -> dict:
    base_config = get_benchmark_preset(config.preset_name)
    baseline_runs = []
    candidate_runs = []

    for test_regime in config.test_regimes:
        for seed in config.seeds:
            baseline_config = _resolve_benchmark_config(
                base_config,
                test_regime=test_regime,
                seed=seed,
                train_size=config.train_size,
                test_size=config.test_size,
                epochs=config.epochs,
                projected_residual_penalty_weight=config.projected_residual_penalty_weight,
                ambient_residual_penalty_weight=config.ambient_residual_penalty_weight,
                deterministic_publication_mode=config.deterministic_publication_mode,
                publication_num_threads=config.publication_num_threads,
            )
            baseline_result = run_benchmark(baseline_config)
            baseline_runs.append(
                {
                    "preset_name": config.preset_name,
                    "test_regime": test_regime,
                    "seed": seed,
                    "candidate_label": "baseline",
                    "config": asdict(baseline_config),
                    "metadata": baseline_result["metadata"],
                    "summary": baseline_result["summary"],
                }
            )

            for candidate in config.candidates:
                if _candidate_matches_base(candidate, base_config):
                    continue
                candidate_config = _resolve_benchmark_config(
                    base_config,
                    test_regime=test_regime,
                    seed=seed,
                    train_size=config.train_size,
                    test_size=config.test_size,
                    epochs=config.epochs,
                    architecture_name=candidate.architecture_name,
                    latent_dim=candidate.latent_dim,
                    pod_dim=candidate.latent_dim,
                    hidden_dims=candidate.hidden_dims,
                    decoder_hidden_dims=candidate.decoder_hidden_dims,
                    projected_residual_penalty_weight=config.projected_residual_penalty_weight,
                    ambient_residual_penalty_weight=config.ambient_residual_penalty_weight,
                    deterministic_publication_mode=config.deterministic_publication_mode,
                    publication_num_threads=config.publication_num_threads,
                )
                candidate_result = run_benchmark(candidate_config)
                candidate_runs.append(
                    {
                        "preset_name": config.preset_name,
                        "test_regime": test_regime,
                        "seed": seed,
                        "candidate_label": candidate.label,
                        "config": asdict(candidate_config),
                        "metadata": candidate_result["metadata"],
                        "summary": candidate_result["summary"],
                    }
                )

    baseline_rows = aggregate_preset_tuning_runs(baseline_runs, include_candidate_metadata=False)
    candidate_rows = aggregate_preset_tuning_runs(candidate_runs, include_candidate_metadata=True)
    recommendations = pick_preset_tuning_recommendations(
        baseline_rows,
        candidate_rows,
        residual_tolerance_fraction=config.residual_tolerance_fraction,
    )
    return {
        "config": asdict(config),
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed_list": list(config.seeds),
            "deterministic_publication_mode": config.deterministic_publication_mode,
            "publication_num_threads": config.publication_num_threads,
        },
        "baseline": baseline_rows,
        "candidates": candidate_rows,
        "recommendations": recommendations,
    }


def aggregate_preset_tuning_runs(runs: list[dict], *, include_candidate_metadata: bool) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        key = (run["test_regime"], run["candidate_label"])
        grouped.setdefault(key, []).append(run)

    rows = []
    for (test_regime, candidate_label), group in sorted(grouped.items()):
        first_config = group[0]["config"]
        row = {
            "preset_name": group[0]["preset_name"],
            "problem_name": first_config["problem_name"],
            "test_regime": test_regime,
            "candidate_label": candidate_label,
            "num_runs": len(group),
            "mgp_mean_error_l2": _mean_summary(group, "mgp_mean_error_l2"),
            "mgp_mean_residual": _mean_summary(group, "mgp_mean_residual"),
            "mgp_mean_projected_residual": _mean_summary(group, "mgp_mean_projected_residual"),
            "reconstruction_mean_error_l2": _mean_summary(group, "reconstruction_mean_error_l2"),
            "reconstruction_mean_residual": _mean_summary(group, "reconstruction_mean_residual"),
            "reconstruction_mean_projected_residual": _mean_summary(
                group, "reconstruction_mean_projected_residual"
            ),
            "mgp_mean_qoi_error": _mean_summary(group, "mgp_mean_qoi_error"),
            "mgp_convergence_rate": _mean_summary(group, "mgp_convergence_rate"),
            "mgp_speedup_vs_full": _mean_summary(group, "mgp_speedup_vs_full"),
            "mgp_std_error_l2": _pstdev_summary(group, "mgp_mean_error_l2"),
            "mgp_std_residual": _pstdev_summary(group, "mgp_mean_residual"),
        }
        if include_candidate_metadata:
            row.update(
                {
                    "architecture_name": first_config["architecture_name"],
                    "latent_dim": first_config["latent_dim"],
                    "hidden_dims": list(first_config["hidden_dims"]),
                    "decoder_hidden_dims": list(first_config["decoder_hidden_dims"]),
                    "activation_name": first_config["activation_name"],
                    "projected_residual_penalty_weight": first_config["projected_residual_penalty_weight"],
                    "ambient_residual_penalty_weight": first_config["ambient_residual_penalty_weight"],
                    "deterministic_publication_mode": group[0]["metadata"]["deterministic_publication_mode"],
                    "publication_num_threads": group[0]["metadata"]["publication_num_threads_requested"],
                }
            )
        else:
            row.update(
                {
                    "architecture_name": first_config["architecture_name"],
                    "latent_dim": first_config["latent_dim"],
                    "hidden_dims": list(first_config["hidden_dims"]),
                    "decoder_hidden_dims": list(first_config["decoder_hidden_dims"]),
                    "activation_name": first_config["activation_name"],
                    "projected_residual_penalty_weight": first_config["projected_residual_penalty_weight"],
                    "ambient_residual_penalty_weight": first_config["ambient_residual_penalty_weight"],
                    "deterministic_publication_mode": group[0]["metadata"]["deterministic_publication_mode"],
                    "publication_num_threads": group[0]["metadata"]["publication_num_threads_requested"],
                }
            )
        rows.append(row)
    return rows


def pick_preset_tuning_recommendations(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    *,
    residual_tolerance_fraction: float,
) -> list[dict]:
    baseline_by_regime = {row["test_regime"]: row for row in baseline_rows}
    recommendations = []

    for test_regime, baseline_row in sorted(baseline_by_regime.items()):
        residual_limit = baseline_row["mgp_mean_residual"] * (1.0 + residual_tolerance_fraction)
        regime_candidates = [row for row in candidate_rows if row["test_regime"] == test_regime]
        feasible = [
            row
            for row in regime_candidates
            if row["mgp_mean_residual"] <= residual_limit
        ]
        if feasible:
            recommended = min(feasible, key=lambda row: row["mgp_mean_error_l2"])
            status = "candidate_selected"
        else:
            recommended = baseline_row
            status = "baseline_retained"

        recommendations.append(
            {
                "preset_name": baseline_row["preset_name"],
                "problem_name": baseline_row["problem_name"],
                "test_regime": test_regime,
                "baseline_label": baseline_row["candidate_label"],
                "baseline_architecture": baseline_row["architecture_name"],
                "baseline_error": baseline_row["mgp_mean_error_l2"],
                "baseline_residual": baseline_row["mgp_mean_residual"],
                "residual_limit": residual_limit,
                "status": status,
                "recommended_label": recommended["candidate_label"],
                "recommended_architecture": recommended["architecture_name"],
                "recommended_error": recommended["mgp_mean_error_l2"],
                "recommended_residual": recommended["mgp_mean_residual"],
                "recommended_projected_residual": recommended.get("mgp_mean_projected_residual", 0.0),
                "recommended_reconstruction_projected_residual": recommended.get(
                    "reconstruction_mean_projected_residual", 0.0
                ),
                "recommended_qoi_error": recommended["mgp_mean_qoi_error"],
                "error_improvement": baseline_row["mgp_mean_error_l2"] - recommended["mgp_mean_error_l2"],
                "residual_change": recommended["mgp_mean_residual"] - baseline_row["mgp_mean_residual"],
                "num_feasible_candidates": len(feasible),
            }
        )
    return recommendations


def format_preset_tuning_study_markdown(payload: dict) -> str:
    lines = [
        "# Preset Tuning Study",
        "",
        "## Recommendations",
        "",
    ]
    for row in payload["recommendations"]:
        format_row = {
            **row,
            "recommended_projected_residual": _format_scientific(row["recommended_projected_residual"]),
            "recommended_reconstruction_projected_residual": _format_scientific(
                row["recommended_reconstruction_projected_residual"]
            ),
        }
        lines.append(
            "- preset={preset_name}, problem={problem_name}, regime={test_regime}, status={status}, recommended={recommended_label}, baseline_error={baseline_error:.6f}, recommended_error={recommended_error:.6f}, baseline_residual={baseline_residual:.6f}, recommended_residual={recommended_residual:.6f}, recommended_projected_residual={recommended_projected_residual}, recommended_reconstruction_projected_residual={recommended_reconstruction_projected_residual}, residual_limit={residual_limit:.6f}, feasible_candidates={num_feasible_candidates}".format(
                **format_row,
            )
        )

    lines.extend(["", "## Baseline", ""])
    for row in payload["baseline"]:
        format_row = {
            **row,
            "mgp_mean_projected_residual": _format_scientific(row["mgp_mean_projected_residual"]),
            "reconstruction_mean_projected_residual": _format_scientific(
                row["reconstruction_mean_projected_residual"]
            ),
        }
        lines.append(
            "- regime={test_regime}, architecture={architecture_name}, latent_dim={latent_dim}, hidden_dims={hidden_dims}, decoder_hidden_dims={decoder_hidden_dims}, publication_mode={deterministic_publication_mode}, threads={publication_num_threads}, projected_weight={projected_residual_penalty_weight:.4f}, ambient_weight={ambient_residual_penalty_weight:.4f}, mgp_mean_error_l2={mgp_mean_error_l2:.6f}, mgp_mean_residual={mgp_mean_residual:.6f}, mgp_mean_projected_residual={mgp_mean_projected_residual}, reconstruction_mean_residual={reconstruction_mean_residual:.6f}, reconstruction_mean_projected_residual={reconstruction_mean_projected_residual}, mgp_mean_qoi_error={mgp_mean_qoi_error:.6f}".format(
                test_regime=format_row["test_regime"],
                architecture_name=format_row["architecture_name"],
                latent_dim=format_row["latent_dim"],
                hidden_dims="x".join(str(value) for value in format_row["hidden_dims"]),
                decoder_hidden_dims=(
                    "auto"
                    if not format_row["decoder_hidden_dims"]
                    else "x".join(str(value) for value in format_row["decoder_hidden_dims"])
                ),
                projected_residual_penalty_weight=format_row["projected_residual_penalty_weight"],
                ambient_residual_penalty_weight=format_row["ambient_residual_penalty_weight"],
                deterministic_publication_mode=format_row["deterministic_publication_mode"],
                publication_num_threads=format_row["publication_num_threads"],
                mgp_mean_error_l2=format_row["mgp_mean_error_l2"],
                mgp_mean_residual=format_row["mgp_mean_residual"],
                mgp_mean_projected_residual=format_row["mgp_mean_projected_residual"],
                reconstruction_mean_residual=format_row["reconstruction_mean_residual"],
                reconstruction_mean_projected_residual=format_row["reconstruction_mean_projected_residual"],
                mgp_mean_qoi_error=format_row["mgp_mean_qoi_error"],
            )
        )

    lines.extend(["", "## Candidates", ""])
    for row in payload["candidates"]:
        format_row = {
            **row,
            "mgp_mean_projected_residual": _format_scientific(row["mgp_mean_projected_residual"]),
            "reconstruction_mean_projected_residual": _format_scientific(
                row["reconstruction_mean_projected_residual"]
            ),
        }
        lines.append(
            "- regime={test_regime}, label={candidate_label}, architecture={architecture_name}, latent_dim={latent_dim}, hidden_dims={hidden_dims}, decoder_hidden_dims={decoder_hidden_dims}, publication_mode={deterministic_publication_mode}, threads={publication_num_threads}, projected_weight={projected_residual_penalty_weight:.4f}, ambient_weight={ambient_residual_penalty_weight:.4f}, mgp_mean_error_l2={mgp_mean_error_l2:.6f}, mgp_mean_residual={mgp_mean_residual:.6f}, mgp_mean_projected_residual={mgp_mean_projected_residual}, reconstruction_mean_residual={reconstruction_mean_residual:.6f}, reconstruction_mean_projected_residual={reconstruction_mean_projected_residual}, mgp_mean_qoi_error={mgp_mean_qoi_error:.6f}".format(
                test_regime=format_row["test_regime"],
                candidate_label=format_row["candidate_label"],
                architecture_name=format_row["architecture_name"],
                latent_dim=format_row["latent_dim"],
                hidden_dims="x".join(str(value) for value in format_row["hidden_dims"]),
                decoder_hidden_dims=(
                    "auto"
                    if not format_row["decoder_hidden_dims"]
                    else "x".join(str(value) for value in format_row["decoder_hidden_dims"])
                ),
                projected_residual_penalty_weight=format_row["projected_residual_penalty_weight"],
                ambient_residual_penalty_weight=format_row["ambient_residual_penalty_weight"],
                deterministic_publication_mode=format_row["deterministic_publication_mode"],
                publication_num_threads=format_row["publication_num_threads"],
                mgp_mean_error_l2=format_row["mgp_mean_error_l2"],
                mgp_mean_residual=format_row["mgp_mean_residual"],
                mgp_mean_projected_residual=format_row["mgp_mean_projected_residual"],
                reconstruction_mean_residual=format_row["reconstruction_mean_residual"],
                reconstruction_mean_projected_residual=format_row["reconstruction_mean_projected_residual"],
                mgp_mean_qoi_error=format_row["mgp_mean_qoi_error"],
            )
        )
    return "\n".join(lines) + "\n"


def _resolve_benchmark_config(
    base_config: BenchmarkConfig,
    *,
    test_regime: str,
    seed: int,
    train_size: int | None,
    test_size: int | None,
    epochs: int | None,
    architecture_name: str | None = None,
    latent_dim: int | None = None,
    pod_dim: int | None = None,
    hidden_dims: tuple[int, ...] | None = None,
    decoder_hidden_dims: tuple[int, ...] | None = None,
    projected_residual_penalty_weight: float | None = None,
    ambient_residual_penalty_weight: float | None = None,
    deterministic_publication_mode: bool | None = None,
    publication_num_threads: int | None = None,
) -> BenchmarkConfig:
    return replace(
        base_config,
        test_regime=test_regime,
        seed=seed,
        train_size=train_size if train_size is not None else base_config.train_size,
        test_size=test_size if test_size is not None else base_config.test_size,
        autoencoder_epochs=epochs if epochs is not None else base_config.autoencoder_epochs,
        architecture_name=architecture_name if architecture_name is not None else base_config.architecture_name,
        latent_dim=latent_dim if latent_dim is not None else base_config.latent_dim,
        pod_dim=pod_dim if pod_dim is not None else base_config.pod_dim,
        hidden_dims=hidden_dims if hidden_dims is not None else base_config.hidden_dims,
        decoder_hidden_dims=(
            decoder_hidden_dims if decoder_hidden_dims is not None else base_config.decoder_hidden_dims
        ),
        projected_residual_penalty_weight=(
            projected_residual_penalty_weight
            if projected_residual_penalty_weight is not None
            else base_config.projected_residual_penalty_weight
        ),
        ambient_residual_penalty_weight=(
            ambient_residual_penalty_weight
            if ambient_residual_penalty_weight is not None
            else base_config.ambient_residual_penalty_weight
        ),
        deterministic_publication_mode=(
            deterministic_publication_mode
            if deterministic_publication_mode is not None
            else base_config.deterministic_publication_mode
        ),
        publication_num_threads=(
            publication_num_threads
            if publication_num_threads is not None
            else base_config.publication_num_threads
        ),
    )


def _candidate_matches_base(candidate: PresetTuningCandidate, base_config: BenchmarkConfig) -> bool:
    return (
        candidate.architecture_name == base_config.architecture_name
        and candidate.latent_dim == base_config.latent_dim
        and candidate.hidden_dims == base_config.hidden_dims
        and candidate.decoder_hidden_dims == base_config.decoder_hidden_dims
    )


def _mean_summary(group: list[dict], key: str) -> float:
    return float(sum(run["summary"][key] for run in group) / len(group))


def _pstdev_summary(group: list[dict], key: str) -> float:
    values = [run["summary"][key] for run in group]
    if len(values) == 1:
        return 0.0
    return float(statistics.pstdev(values))


def _format_scientific(value: float) -> str:
    return f"{value:.3e}"
