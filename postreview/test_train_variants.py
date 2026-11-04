from __future__ import annotations

import inspect
import io
import os
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.environ.get("MGP_SRC", str(Path(__file__).resolve().parent / "src")))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mgp.autoencoder import SnapshotAutoencoder, resolve_activation, train_autoencoder
from mgp.benchmark import BenchmarkConfig, _effective_projected_weight, _instantiate_scaler, _physics_penalty_terms
from mgp.problems import create_problem
from train_variants import train_autoencoder_variant


def build(artifact):
    config = BenchmarkConfig(**artifact["config"])
    problem = create_problem(config.problem_name)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = SnapshotAutoencoder(input_dim=problem.interior_dim, latent_dim=config.latent_dim, architecture_name=config.architecture_name,
                                hidden_dims=tuple(config.hidden_dims), decoder_hidden_dims=tuple(config.decoder_hidden_dims),
                                activation=resolve_activation(config.activation_name))
    return config, problem, model


def main():
    archive = Path(sys.argv[1]); epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with tarfile.open(archive, "r:gz") as tar:
        m = next(x for x in tar.getmembers() if x.name.endswith("offline_artifact.pt"))
        artifact = torch.load(io.BytesIO(tar.extractfile(m).read()), map_location="cpu", weights_only=False)
    scaler = _instantiate_scaler(artifact)
    snaps = np.asarray(artifact["train_snapshots"], dtype=np.float64)
    params = np.asarray(artifact["train_parameters"], dtype=np.float64)
    normalized = scaler.transform(snaps)
    out = []
    for which in ("frozen", "variant"):
        config, problem, model = build(artifact)
        fast = {"compute_projected_graph": _effective_projected_weight(config) != 0.0} if "compute_projected_graph" in inspect.signature(_physics_penalty_terms).parameters else {}
        pen = lambda r, z, p, model=model, problem=problem, config=config, fast=fast: _physics_penalty_terms(r, z, p, problem, scaler, model, config.residual_penalty_metric, **fast)
        t0 = time.perf_counter()
        if which == "frozen":
            hist = train_autoencoder(model, normalized, epochs=epochs, learning_rate=config.autoencoder_lr,
                                     projected_residual_penalty_weight=_effective_projected_weight(config),
                                     ambient_residual_penalty_weight=config.ambient_residual_penalty_weight,
                                     residual_penalty_schedule=config.residual_penalty_schedule, parameters=params,
                                     physics_penalty_fn=pen, seed=config.seed)
        else:
            hist, best = train_autoencoder_variant(model, normalized, epochs=epochs, learning_rate=config.autoencoder_lr,
                                                   projected_residual_penalty_weight=_effective_projected_weight(config),
                                                   ambient_residual_penalty_weight=config.ambient_residual_penalty_weight,
                                                   residual_penalty_schedule=config.residual_penalty_schedule, parameters=params,
                                                   physics_penalty_fn=pen, seed=config.seed)
        out.append((model.state_dict(), hist["loss"], time.perf_counter() - t0))
    same = all(torch.equal(out[0][0][k], out[1][0][k]) for k in out[0][0])
    print(f"epochs={epochs} bitwise_identical_weights={same} identical_loss_history={out[0][1] == out[1][1]} "
          f"time_frozen={out[0][2]:.1f}s time_variant={out[1][2]:.1f}s per_epoch={out[0][2]/epochs:.2f}s")
    if epochs == artifact["config"]["autoencoder_epochs"]:
        dw = max(float((out[0][0][k] - artifact["autoencoder_state_dict"][k]).abs().max()) for k in out[0][0])
        print(f"max |w - archived| = {dw:.3e}")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
