from __future__ import annotations

from dataclasses import replace

from .benchmark import BenchmarkConfig


_BENCHMARK_PRESETS = {
    "reference_nonlinear_diffusion": BenchmarkConfig(
        problem_name="nonlinear_diffusion",
        test_regime="in_domain",
        latent_dim=2,
        pod_dim=2,
        architecture_name="conv1d",
        hidden_dims=(24, 24, 24),
        decoder_hidden_dims=(48, 48),
        activation_name="silu",
        autoencoder_epochs=250,
        residual_penalty_weight=0.02,
        residual_penalty_schedule="constant",
        residual_penalty_metric="l2_norm",
        warm_start_strategy="mean_train",
    ),
    "reference_front_layer": BenchmarkConfig(
        problem_name="front_layer",
        test_regime="in_domain",
        latent_dim=2,
        pod_dim=2,
        architecture_name="conv1d",
        hidden_dims=(24, 24, 24),
        decoder_hidden_dims=(48, 48),
        activation_name="silu",
        autoencoder_epochs=250,
        residual_penalty_weight=0.02,
        residual_penalty_schedule="constant",
        residual_penalty_metric="l2_norm",
        warm_start_strategy="mean_train",
    ),
    "reference_bratu_source": BenchmarkConfig(
        problem_name="bratu_source",
        test_regime="in_domain",
        latent_dim=2,
        pod_dim=2,
        architecture_name="mlp",
        hidden_dims=(24, 24, 24),
        decoder_hidden_dims=(48, 48),
        activation_name="silu",
        autoencoder_epochs=250,
        residual_penalty_weight=0.02,
        residual_penalty_schedule="constant",
        residual_penalty_metric="l2_norm",
        warm_start_strategy="mean_train",
    ),
    "reference_hydrologic_conductivity": BenchmarkConfig(
        problem_name="hydrologic_conductivity",
        test_regime="in_domain",
        latent_dim=2,
        pod_dim=2,
        architecture_name="conv1d",
        hidden_dims=(24, 24, 24),
        decoder_hidden_dims=(48, 48),
        activation_name="silu",
        autoencoder_epochs=250,
        residual_penalty_weight=0.02,
        residual_penalty_schedule="constant",
        residual_penalty_metric="l2_norm",
        warm_start_strategy="mean_train",
    ),
}


def list_benchmark_presets() -> tuple[str, ...]:
    return tuple(_BENCHMARK_PRESETS.keys())


def get_benchmark_preset(name: str) -> BenchmarkConfig:
    try:
        return replace(_BENCHMARK_PRESETS[name])
    except KeyError as exc:
        raise ValueError(f"unknown benchmark preset: {name}") from exc
