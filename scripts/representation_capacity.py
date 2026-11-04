#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mgp.benchmark import _instantiate_autoencoder, _instantiate_scaler

SEALED = ROOT / "eval/prospective_weight_validation_2026-07-17/first_look_input/sealed"
OUT = Path(__file__).resolve().parents[1] / "supplementary_data"
ARMS = ["mg", "newton_m_lspg", "lm_m_lspg", "pod_g", "pod_lspg", "precond_m_lspg", "precond_pod_lspg"]
FAMILY = {
    "bratu_source": "Bratu source",
    "front_layer": "Front layer",
    "hydrologic_conductivity": "Hydraulic conductivity",
    "nonlinear_diffusion": "Nonlinear diffusion",
}
torch.set_num_threads(1)


def read_archive(tar_path: Path):
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()

        def read(suffix: str) -> bytes:
            member = next(m for m in members if m.name.endswith(suffix))
            return tar.extractfile(member).read()

        result = json.loads(read(".result.json"))
        artifact = torch.load(io.BytesIO(read(".offline_artifact.pt")), map_location="cpu", weights_only=False)
        with np.load(io.BytesIO(read(".offline_matrices.npz"))) as npz:
            matrices = {k: npz[k] for k in npz.files}
    return result, artifact, matrices


def relerr(u: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(u - ref) / max(np.linalg.norm(ref), 1e-12))


def main() -> None:
    rows = []
    for tar_path in sorted(SEALED.glob("*.tar.gz")):
        result, artifact, mats = read_archive(tar_path)
        cfg = result["config"]
        ae = _instantiate_autoencoder(artifact)
        scaler = _instantiate_scaler(artifact)
        V, u_c = mats["pod_basis"], mats["pod_mean"]
        u_mean = np.asarray(artifact["train_snapshots"], dtype=float).mean(axis=0)

        def decode(z: torch.Tensor) -> torch.Tensor:
            return scaler.inverse_transform_tensor(ae.decode(z))

        def encode(u: np.ndarray) -> torch.Tensor:
            with torch.no_grad():
                return ae.encode(scaler.transform_tensor(torch.tensor(u, dtype=torch.float64)))

        with torch.no_grad():
            u_lstart = decode(encode(u_mean)).numpy()

        for case in result["cases"]:
            u_ref = np.asarray(case["full_order"]["state"], dtype=float)
            d = u_ref - u_c
            e_proj = float(np.linalg.norm(d - V @ (V.T @ d)) / np.linalg.norm(u_ref))
            z0 = encode(u_ref)
            with torch.no_grad():
                e_ae = relerr(decode(z0).numpy(), u_ref)
            z = z0.clone().requires_grad_(True)
            target = torch.tensor(u_ref, dtype=torch.float64)
            opt = torch.optim.LBFGS([z], max_iter=60, tolerance_grad=1e-12, tolerance_change=1e-14,
                                    line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                loss = torch.sum((decode(z) - target) ** 2)
                loss.backward()
                return loss

            opt.step(closure)
            with torch.no_grad():
                e_pol = min(e_ae, relerr(decode(z).numpy(), u_ref))
            row = {
                "family": cfg["problem_name"], "regime": cfg["test_regime"], "seed": cfg["seed"],
                "sample": case["sample"], "e_proj": e_proj, "e_ae": e_ae, "e_pol": e_pol,
                "e_start": relerr(u_mean, u_ref), "e_lstart": relerr(u_lstart, u_ref),
            }
            for arm in ARMS:
                row[arm] = case["arms"][arm]["relative_state_error"]
            rows.append(row)
        print(tar_path.name, flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "representation_capacity_cases.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cells = []
    for fam in FAMILY:
        for reg in ("in_domain", "ood"):
            g = [r for r in rows if r["family"] == fam and r["regime"] == reg]
            arr = lambda k: np.array([r[k] for r in g])
            cell = {"family": FAMILY[fam], "regime": "in" if reg == "in_domain" else "OOD", "n_cases": len(g)}
            for k in ["e_proj", "e_ae", "e_pol", "e_start", "e_lstart", *ARMS]:
                cell[f"mean_{k}"] = arr(k).mean()
            cell["frac_pol_below_proj"] = np.mean(arr("e_pol") < arr("e_proj"))
            cell["frac_podg_below_pol"] = np.mean(arr("pod_g") < arr("e_pol"))
            cell["frac_mg_below_proj"] = np.mean(arr("mg") < arr("e_proj"))
            for arm in ARMS:
                cell[f"frac_{arm}_above_one"] = np.mean(arr(arm) > 1.0)
                start = arr("e_lstart") if arm in ("mg", "newton_m_lspg", "lm_m_lspg", "precond_m_lspg") else arr("e_start")
                cell[f"frac_{arm}_above_start"] = np.mean(arr(arm) > start)
            cells.append(cell)
    with open(OUT / "representation_capacity_cells.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0]))
        w.writeheader()
        w.writerows(cells)

    print("\nLaTeX rows: family & regime & e_proj & e_pol & POD-G & MG & pre-M & pol<proj & POD-G<pol & start(aff/learned) & Newton>1 & Newton>start & LM>1 & LM>start")
    for c in cells:
        print(f"{c['family']} & {c['regime']} & {c['mean_e_proj']:.3f} & {c['mean_e_pol']:.3f} & {c['mean_pod_g']:.3f} & {c['mean_mg']:.3f} & {c['mean_precond_m_lspg']:.3f} & "
              f"{100*c['frac_pol_below_proj']:.0f}\\% & {100*c['frac_podg_below_pol']:.0f}\\% & {c['mean_e_start']:.2f}/{c['mean_e_lstart']:.2f} & "
              f"{100*c['frac_newton_m_lspg_above_one']:.0f}\\% & {100*c['frac_newton_m_lspg_above_start']:.0f}\\% & "
              f"{100*c['frac_lm_m_lspg_above_one']:.0f}\\% & {100*c['frac_lm_m_lspg_above_start']:.0f}\\% \\\\")


if __name__ == "__main__":
    main()
