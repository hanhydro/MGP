from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from mgp.autoencoder import residual_penalty_scale


def train_autoencoder_variant(
    model,
    snapshots: np.ndarray,
    *,
    epochs: int,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    projected_residual_penalty_weight: float = 0.0,
    ambient_residual_penalty_weight: float = 0.0,
    residual_penalty_schedule: str = "constant",
    parameters: np.ndarray | None = None,
    physics_penalty_fn=None,
    seed: int = 0,
    lr_schedule: str | None = None,
    lr_min: float = 1e-5,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    total_physics_penalty_weight = projected_residual_penalty_weight + ambient_residual_penalty_weight
    snapshot_tensor = torch.tensor(snapshots, dtype=torch.float64)
    parameter_tensor = torch.tensor(parameters, dtype=torch.float64)
    dataset = TensorDataset(snapshot_tensor, parameter_tensor)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"loss": [], "reconstruction": [], "residual": [], "projected_residual": [], "ambient_residual": [], "learning_rate": []}
    best = {"loss": math.inf, "epoch": -1, "state_dict": None}

    for epoch_index in range(epochs):
        if lr_schedule == "cosine":
            lr = lr_min + 0.5 * (learning_rate - lr_min) * (1.0 + math.cos(math.pi * epoch_index / max(1, epochs - 1)))
            for group in optimizer.param_groups:
                group["lr"] = lr
        elif lr_schedule is not None:
            raise ValueError(f"unknown lr_schedule: {lr_schedule}")
        residual_scale = residual_penalty_scale(residual_penalty_schedule, epoch_index, epochs)
        total_loss = total_recon = total_residual = total_projected = total_ambient = 0.0
        seen = 0
        for batch, parameter_batch in loader:
            optimizer.zero_grad()
            latent = model.encode(batch)
            reconstructed = model.decode(latent)
            recon_loss = torch.mean((reconstructed - batch) ** 2)
            projected_loss = torch.tensor(0.0, dtype=torch.float64)
            ambient_loss = torch.tensor(0.0, dtype=torch.float64)
            if total_physics_penalty_weight > 0.0:
                terms = physics_penalty_fn(reconstructed, latent, parameter_batch)
                projected_loss = terms.get("projected", torch.tensor(0.0, dtype=torch.float64))
                ambient_loss = terms.get("ambient", torch.tensor(0.0, dtype=torch.float64))
            residual_loss = projected_residual_penalty_weight * projected_loss + ambient_residual_penalty_weight * ambient_loss
            loss = recon_loss + 0.0 * torch.tensor(0.0, dtype=torch.float64) + residual_scale * residual_loss
            loss.backward()
            optimizer.step()
            n = batch.shape[0]
            total_loss += loss.item() * n
            total_recon += recon_loss.item() * n
            total_residual += residual_loss.item() * n
            total_projected += projected_loss.item() * n
            total_ambient += ambient_loss.item() * n
            seen += n
        epoch_loss = total_loss / seen
        history["loss"].append(epoch_loss)
        history["reconstruction"].append(total_recon / seen)
        history["residual"].append(total_residual / seen)
        history["projected_residual"].append(total_projected / seen)
        history["ambient_residual"].append(total_ambient / seen)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        if epoch_loss < best["loss"]:
            best = {"loss": epoch_loss, "epoch": epoch_index + 1, "state_dict": copy.deepcopy(model.state_dict())}
    return history, best
