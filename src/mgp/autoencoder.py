from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def resolve_activation(name: str) -> type[nn.Module]:
    activations = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "elu": nn.ELU,
    }
    try:
        return activations[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown activation: {name}") from exc


def _build_mlp(layer_dims: Iterable[int], activation: type[nn.Module]) -> nn.Sequential:
    dims = list(layer_dims)
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def _build_conv_stack(
    channel_dims: Iterable[int],
    activation: type[nn.Module],
    *,
    final_activation: bool,
) -> nn.Sequential:
    dims = list(channel_dims)
    layers: list[nn.Module] = []
    for index, (in_channels, out_channels) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2))
        is_last = index == len(dims) - 2
        if not is_last or final_activation:
            layers.append(activation())
    return nn.Sequential(*layers)


def _as_batch(values: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if values.ndim == 1:
        return values.unsqueeze(0), True
    return values, False


def _coordinate_channel(input_dim: int) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, input_dim, dtype=torch.float64).reshape(1, 1, input_dim)


@dataclass
class SnapshotScaler:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, snapshots: np.ndarray) -> "SnapshotScaler":
        mean = torch.tensor(np.mean(snapshots, axis=0), dtype=torch.float64)
        std = torch.tensor(np.std(snapshots, axis=0), dtype=torch.float64)
        std = torch.where(std > 1e-10, std, torch.ones_like(std))
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        mean = self.mean.detach().cpu().numpy()
        std = self.std.detach().cpu().numpy()
        return (values - mean) / std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        mean = self.mean.detach().cpu().numpy()
        std = self.std.detach().cpu().numpy()
        return values * std + mean

    def transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(values.device)
        std = self.std.to(values.device)
        return (values - mean) / std

    def inverse_transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(values.device)
        std = self.std.to(values.device)
        return values * std + mean


class SnapshotAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: tuple[int, ...] = (64, 64),
        decoder_hidden_dims: tuple[int, ...] = (),
        activation: type[nn.Module] = nn.Tanh,
        architecture_name: str = "mlp",
    ) -> None:
        super().__init__()
        if latent_dim >= input_dim:
            raise ValueError("latent_dim must be smaller than input_dim for reduction")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.architecture_name = architecture_name

        if architecture_name == "mlp":
            encoder_dims = (input_dim, *hidden_dims, latent_dim)
            resolved_decoder_hidden_dims = decoder_hidden_dims or tuple(reversed(hidden_dims))
            decoder_dims = (latent_dim, *resolved_decoder_hidden_dims, input_dim)
            self.encoder = _build_mlp(encoder_dims, activation)
            self.decoder = _build_mlp(decoder_dims, activation)
        elif architecture_name in {"conv1d", "coordconv1d"}:
            encoder_channels = hidden_dims or (16, 32)
            resolved_decoder_channels = decoder_hidden_dims or tuple(reversed(encoder_channels))
            if not resolved_decoder_channels:
                raise ValueError("conv1d architecture requires decoder channels")
            encoder_input_channels = 2 if architecture_name == "coordconv1d" else 1
            self.encoder_conv = _build_conv_stack(
                (encoder_input_channels, *encoder_channels),
                activation,
                final_activation=True,
            )
            self.encoder_projection = nn.Linear(encoder_channels[-1] * input_dim, latent_dim)
            self.decoder_start_channels = resolved_decoder_channels[0]
            self.decoder_projection = nn.Linear(latent_dim, self.decoder_start_channels * input_dim)
            self.decoder_conv = _build_conv_stack(
                (*resolved_decoder_channels, 1),
                activation,
                final_activation=False,
            )
            if architecture_name == "coordconv1d":
                self.register_buffer("encoder_coordinate_channel", _coordinate_channel(input_dim))
        else:
            raise ValueError(f"unknown architecture_name: {architecture_name}")
        self.double()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch, squeeze = _as_batch(x)
        if self.architecture_name == "mlp":
            encoded = self.encoder(batch)
        else:
            state_channel = batch.unsqueeze(1)
            if self.architecture_name == "coordconv1d":
                coordinate_channel = self.encoder_coordinate_channel.to(batch.device).expand(batch.shape[0], -1, -1)
                encoder_input = torch.cat((state_channel, coordinate_channel), dim=1)
            else:
                encoder_input = state_channel
            features = self.encoder_conv(encoder_input)
            encoded = self.encoder_projection(features.flatten(start_dim=1))
        return encoded.squeeze(0) if squeeze else encoded

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch, squeeze = _as_batch(z)
        if self.architecture_name == "mlp":
            decoded = self.decoder(batch)
        else:
            features = self.decoder_projection(batch).reshape(
                batch.shape[0], self.decoder_start_channels, self.input_dim
            )
            decoded = self.decoder_conv(features).squeeze(1)
        return decoded.squeeze(0) if squeeze else decoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def decoder_jacobian(self, z: torch.Tensor, *, create_graph: bool = False) -> torch.Tensor:
        return torch.autograd.functional.jacobian(self.decode, z, create_graph=create_graph)


def _encoder_smoothness_penalty(model: SnapshotAutoencoder, batch: torch.Tensor) -> torch.Tensor:
    penalties = []
    for sample in batch:
        sample_var = sample.detach().clone().requires_grad_(True)
        jacobian = torch.autograd.functional.jacobian(model.encode, sample_var)
        penalties.append((jacobian**2).mean())
    return torch.stack(penalties).mean()


def residual_penalty_scale(schedule: str, epoch_index: int, num_epochs: int) -> float:
    if schedule == "constant":
        return 1.0
    if schedule == "linear_ramp":
        return float(epoch_index + 1) / float(max(num_epochs, 1))
    if schedule in {"staged_recon_to_residual", "delayed_linear_ramp"}:
        progress = float(epoch_index + 1) / float(max(num_epochs, 1))
        if progress <= 0.4:
            return 0.0
        if progress >= 0.8:
            return 1.0
        return (progress - 0.4) / 0.4
    raise ValueError(f"unknown residual_penalty_schedule: {schedule}")


def _validate_training_objective(
    training_objective_mode: str,
    residual_penalty_schedule: str,
    online_residual_sample_count: int,
    online_residual_sample_scale: float,
    online_residual_sample_source: str,
) -> None:
    objective_modes = {
        "standard",
        "staged_curriculum",
        "staged_online_residual",
        "staged_online_projected",
    }
    if training_objective_mode not in objective_modes:
        raise ValueError(f"unknown training_objective_mode: {training_objective_mode}")
    if training_objective_mode.startswith("staged") and residual_penalty_schedule not in {
        "staged_recon_to_residual",
        "delayed_linear_ramp",
    }:
        raise ValueError(
            "staged training objective requires residual_penalty_schedule="
            "'staged_recon_to_residual'"
        )
    if online_residual_sample_count < 0:
        raise ValueError("online_residual_sample_count must be non-negative")
    online_mode = training_objective_mode in {"staged_online_residual", "staged_online_projected"}
    if online_residual_sample_count > 0 and not online_mode:
        raise ValueError("online residual samples require an online training_objective_mode")
    if online_mode and online_residual_sample_count == 0:
        raise ValueError("online training_objective_mode requires online_residual_sample_count > 0")
    if online_residual_sample_scale < 0.0:
        raise ValueError("online_residual_sample_scale must be non-negative")
    if online_residual_sample_source != "encoded_training_latent_perturbation":
        raise ValueError(f"unknown online_residual_sample_source: {online_residual_sample_source}")


def _online_residual_noise_bank(
    sample_count: int,
    latent_dim: int,
    *,
    snapshots_count: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randn(
        snapshots_count,
        sample_count,
        latent_dim,
        generator=generator,
        dtype=torch.float64,
    )


def train_autoencoder(
    model: SnapshotAutoencoder,
    snapshots: np.ndarray,
    *,
    epochs: int = 600,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    smoothness_weight: float = 0.0,
    residual_penalty_weight: float | None = None,
    projected_residual_penalty_weight: float = 0.0,
    ambient_residual_penalty_weight: float = 0.0,
    residual_penalty_schedule: str = "constant",
    training_objective_mode: str = "standard",
    online_residual_sample_count: int = 0,
    online_residual_sample_scale: float = 0.05,
    online_residual_sample_seed_offset: int = 10007,
    online_residual_sample_source: str = "encoded_training_latent_perturbation",
    parameters: np.ndarray | None = None,
    physics_penalty_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, torch.Tensor]] | None = None,
    seed: int = 0,
) -> dict[str, list[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    _validate_training_objective(
        training_objective_mode,
        residual_penalty_schedule,
        online_residual_sample_count,
        online_residual_sample_scale,
        online_residual_sample_source,
    )

    if residual_penalty_weight is not None:
        projected_residual_penalty_weight = residual_penalty_weight
    total_physics_penalty_weight = projected_residual_penalty_weight + ambient_residual_penalty_weight
    if training_objective_mode in {"staged_online_residual", "staged_online_projected"}:
        if total_physics_penalty_weight <= 0.0:
            raise ValueError("online training_objective_mode requires a positive residual penalty weight")
    if total_physics_penalty_weight > 0.0 and (parameters is None or physics_penalty_fn is None):
        raise ValueError("residual penalty requires both parameters and residual_penalty_fn")

    snapshot_tensor = torch.tensor(snapshots, dtype=torch.float64)
    online_samples_enabled = online_residual_sample_count > 0 and total_physics_penalty_weight > 0.0
    if parameters is None:
        parameter_tensor = None
        dataset = TensorDataset(snapshot_tensor)
    else:
        parameter_tensor = torch.tensor(parameters, dtype=torch.float64)
        if online_samples_enabled:
            index_tensor = torch.arange(len(snapshots), dtype=torch.long)
            dataset = TensorDataset(snapshot_tensor, parameter_tensor, index_tensor)
        else:
            dataset = TensorDataset(snapshot_tensor, parameter_tensor)
    if online_samples_enabled:
        online_noise = _online_residual_noise_bank(
            online_residual_sample_count,
            model.latent_dim,
            snapshots_count=len(snapshots),
            seed=seed + online_residual_sample_seed_offset,
        )
    else:
        online_noise = None
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {
        "loss": [],
        "reconstruction": [],
        "smoothness": [],
        "residual": [],
        "projected_residual": [],
        "ambient_residual": [],
        "residual_scale": [],
    }

    for epoch_index in range(epochs):
        residual_scale = residual_penalty_scale(residual_penalty_schedule, epoch_index, epochs)
        if online_samples_enabled:
            with torch.no_grad():
                epoch_latents = model.encode(snapshot_tensor)
                latent_std = torch.std(epoch_latents, dim=0, unbiased=False)
                latent_step = torch.clamp(latent_std, min=1e-3) * online_residual_sample_scale
        total_loss = 0.0
        total_recon = 0.0
        total_smooth = 0.0
        total_residual = 0.0
        total_projected_residual = 0.0
        total_ambient_residual = 0.0
        seen = 0
        for items in loader:
            batch = items[0]
            parameter_batch = items[1] if len(items) > 1 else None
            batch_indices = items[2] if len(items) > 2 else None
            optimizer.zero_grad()
            latent = model.encode(batch)
            reconstructed = model.decode(latent)
            recon_loss = torch.mean((reconstructed - batch) ** 2)
            smooth_loss = torch.tensor(0.0, dtype=torch.float64)
            projected_residual_loss = torch.tensor(0.0, dtype=torch.float64)
            ambient_residual_loss = torch.tensor(0.0, dtype=torch.float64)
            if smoothness_weight > 0.0:
                smooth_loss = _encoder_smoothness_penalty(model, batch)
            if total_physics_penalty_weight > 0.0:
                residual_states = reconstructed
                residual_latents = latent
                residual_parameters = parameter_batch
                if online_samples_enabled:
                    batch_noise = online_noise[batch_indices].to(batch.device)
                    latent_step_batch = latent_step.to(batch.device)
                    online_latents = (
                        latent.detach().unsqueeze(1)
                        + batch_noise * latent_step_batch.reshape(1, 1, -1)
                    ).reshape(-1, model.latent_dim)
                    online_reconstructed = model.decode(online_latents)
                    residual_states = torch.cat((reconstructed, online_reconstructed), dim=0)
                    residual_latents = torch.cat((latent, online_latents), dim=0)
                    residual_parameters = torch.cat(
                        (
                            parameter_batch,
                            parameter_batch.repeat_interleave(online_residual_sample_count, dim=0),
                        ),
                        dim=0,
                    )
                penalty_terms = physics_penalty_fn(residual_states, residual_latents, residual_parameters)
                projected_residual_loss = penalty_terms.get(
                    "projected",
                    torch.tensor(0.0, dtype=torch.float64, device=batch.device),
                )
                ambient_residual_loss = penalty_terms.get(
                    "ambient",
                    torch.tensor(0.0, dtype=torch.float64, device=batch.device),
                )
            residual_loss = (
                projected_residual_penalty_weight * projected_residual_loss
                + ambient_residual_penalty_weight * ambient_residual_loss
            )
            loss = (
                recon_loss
                + smoothness_weight * smooth_loss
                + residual_scale * residual_loss
            )
            loss.backward()
            optimizer.step()

            batch_size_actual = batch.shape[0]
            total_loss += loss.item() * batch_size_actual
            total_recon += recon_loss.item() * batch_size_actual
            total_smooth += smooth_loss.item() * batch_size_actual
            total_residual += residual_loss.item() * batch_size_actual
            total_projected_residual += projected_residual_loss.item() * batch_size_actual
            total_ambient_residual += ambient_residual_loss.item() * batch_size_actual
            seen += batch_size_actual

        history["loss"].append(total_loss / seen)
        history["reconstruction"].append(total_recon / seen)
        history["smoothness"].append(total_smooth / seen)
        history["residual"].append(total_residual / seen)
        history["projected_residual"].append(total_projected_residual / seen)
        history["ambient_residual"].append(total_ambient_residual / seen)
        history["residual_scale"].append(residual_scale)

    return history
