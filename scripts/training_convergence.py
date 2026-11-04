#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import math
import tarfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parents[1] / "supplementary_data"
BLOCKS = {
    "block1": ROOT / "eval/prospective_weight_validation_2026-07-17/first_look_input/sealed",
    "block2": ROOT / "eval/prospective_2d_scale_warmstart_2026-07-19/first_look_input/sealed",
}
FAMILY_LABEL = {
    "bratu_source": "Bratu source",
    "front_layer": "Front layer",
    "hydrologic_conductivity": "Hydraulic conductivity",
    "nonlinear_diffusion": "Nonlinear diffusion",
    "two_d_burgers_large": "2D Burgers-type (N=2304)",
}


def load_artifact(tar_path: Path) -> dict:
    with tarfile.open(tar_path, "r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("offline_artifact.pt"))
        data = tar.extractfile(member).read()
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


def windowed_decrease(x: np.ndarray) -> float:
    k = max(2, len(x) // 10)
    prev, last = x[-2 * k:-k].mean(), x[-k:].mean()
    return (prev - last) / prev


def model_row(block: str, art: dict) -> dict:
    cfg = art["config"]
    history = art["autoencoder_history"]
    loss = np.asarray(art["autoencoder_history"]["loss"], dtype=float)
    recon = np.asarray(art["autoencoder_history"]["reconstruction"], dtype=float)
    n = len(loss)
    k = max(2, n // 10)
    prev, last = loss[-2 * k:-k].mean(), loss[-k:].mean()
    batches = math.ceil(cfg["train_size"] / min(16, cfg["train_size"]))
    return {
        "block": block,
        "family": art["problem_name"],
        "seed": cfg["seed"],
        "regime": cfg["test_regime"],
        "epochs": n,
        "adam_updates": n * batches,
        "initial_loss": loss[0],
        "final_loss": loss[-1],
        "min_loss": loss.min(),
        "best_epoch": int(np.argmin(loss)) + 1,
        "best_epoch_fraction": int(np.argmin(loss)) / (n - 1),
        "final_over_min_loss": loss[-1] / loss.min(),
        "windowed_decrease": (prev - last) / prev,
        "final_reconstruction_loss": recon[-1],
        "train_reconstruction_error": art["autoencoder_reconstruction_error"],
        "windowed_decrease_reconstruction": windowed_decrease(recon),
        "windowed_decrease_projected_residual": windowed_decrease(np.asarray(history["projected_residual"], dtype=float)),
        "windowed_decrease_ambient_residual": windowed_decrease(np.asarray(history["ambient_residual"], dtype=float)),
        "loss_curve": loss,
    }


def main() -> None:
    rows: list[dict] = []
    in_domain_weights: dict = {}
    ood_weights: dict = {}
    for block, folder in BLOCKS.items():
        for tar_path in sorted(folder.glob("*.tar.gz")):
            art = load_artifact(tar_path)
            key = (block, art["problem_name"], art["config"]["seed"])
            if "__in_domain__" in tar_path.name:
                rows.append(model_row(block, art))
                in_domain_weights[key] = art["autoencoder_state_dict"]
            else:
                ood_weights[key] = art["autoencoder_state_dict"]
    OUT.mkdir(parents=True, exist_ok=True)

    with open(OUT / "training_decoder_pair_differences.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["block", "family", "seed", "max_abs_weight_difference", "bitwise_identical"])
        for key, sd in sorted(in_domain_weights.items()):
            other = ood_weights[key]
            diff = max(float((sd[k] - other[k]).abs().max()) for k in sd)
            w.writerow([*key, diff, diff == 0.0])
    for block in BLOCKS:
        diffs = [max(float((sd[k] - ood_weights[key][k]).abs().max()) for k in sd)
                 for key, sd in in_domain_weights.items() if key[0] == block]
        print(f"{block}: {sum(d == 0.0 for d in diffs)}/{len(diffs)} decoder pairs bitwise identical; max |dw| = {max(diffs):.2e}")

    model_fields = [f for f in rows[0] if f != "loss_curve"]
    with open(OUT / "training_convergence_models.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=model_fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in model_fields})

    families = sorted({r["family"] for r in rows}, key=lambda f: list(FAMILY_LABEL).index(f))
    summary = []
    for fam in families:
        g = [r for r in rows if r["family"] == fam]
        wd = np.array([r["windowed_decrease"] for r in g])
        fm = np.array([r["final_over_min_loss"] for r in g])
        tr = np.array([r["train_reconstruction_error"] for r in g])
        summary.append({
            "family": FAMILY_LABEL[fam],
            "n_models": len(g),
            "epochs": g[0]["epochs"],
            "adam_updates": g[0]["adam_updates"],
            "windowed_decrease_median": np.median(wd),
            "windowed_decrease_min": wd.min(),
            "windowed_decrease_max": wd.max(),
            "fraction_windowed_decrease_gt_5pct": np.mean(wd > 0.05),
            "best_epoch_fraction_median": np.median([r["best_epoch_fraction"] for r in g]),
            "final_over_min_loss_median": np.median(fm),
            "n_final_over_min_gt_1p5": int(np.sum(fm > 1.5)),
            "train_reconstruction_error_median": np.median(tr),
            "train_reconstruction_error_min": tr.min(),
            "train_reconstruction_error_max": tr.max(),
            "projected_residual_windowed_decrease_median": np.median([r["windowed_decrease_projected_residual"] for r in g]),
            "ambient_residual_windowed_decrease_median": np.median([r["windowed_decrease_ambient_residual"] for r in g]),
            "fraction_projected_residual_rising": np.mean([r["windowed_decrease_projected_residual"] < 0 for r in g]),
            "fraction_ambient_residual_rising": np.mean([r["windowed_decrease_ambient_residual"] < 0 for r in g]),
        })
    with open(OUT / "training_convergence_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    with open(OUT / "training_curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "epoch", "loss_p10", "loss_median", "loss_p90"])
        for fam in families:
            curves = np.stack([r["loss_curve"] for r in rows if r["family"] == fam])
            for e in range(curves.shape[1]):
                col = curves[:, e]
                w.writerow([FAMILY_LABEL[fam], e + 1, np.percentile(col, 10), np.median(col), np.percentile(col, 90)])

    print("LaTeX rows (family & models & epochs (updates) & median windowed decrease [min, max] & share > 5% & best-epoch fraction & final/min loss median (n > 1.5) & train recon err median [min, max])")
    for s in summary:
        print(
            f"{s['family']} & {s['n_models']} & {s['epochs']} ({s['adam_updates']}) & "
            f"{100*s['windowed_decrease_median']:.1f}\\% [{100*s['windowed_decrease_min']:.0f}, {100*s['windowed_decrease_max']:.0f}] & "
            f"{100*s['fraction_windowed_decrease_gt_5pct']:.0f}\\% & {s['best_epoch_fraction_median']:.2f} & "
            f"{s['final_over_min_loss_median']:.2f} ({s['n_final_over_min_gt_1p5']}) & "
            f"{100*s['train_reconstruction_error_median']:.1f}\\% [{100*s['train_reconstruction_error_min']:.1f}, {100*s['train_reconstruction_error_max']:.1f}] \\\\"
        )


if __name__ == "__main__":
    main()
