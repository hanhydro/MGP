#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrain_variant import read_archive
from run_variant import relative_l2
from mgp.benchmark import _instantiate_autoencoder, _instantiate_scaler
from mgp.manifold_rom import ManifoldGalerkinROM, ManifoldPreconditionedLSPGROM
from mgp.problems import create_problem

NAME = re.compile(r"(?P<case>pv1__reference_(?P<family>\w+?)__in_domain__seed_(?P<seed>\d+))\.(?P<tag>ep\d+_init\d+)\.decoder\.pt$")
RETRAIN_JSON = Path(__file__).resolve().parent / "results" / "retrain_block1"


def budget_label(tag: str, seed: int) -> str:
    epochs, init = tag.split("_")
    offset = int(init.removeprefix("init")) - seed
    return f"{epochs}_dataseed" if offset == 0 else f"{epochs}_init{offset}"


def sealed_path(sealed: Path, case_id: str) -> Path:
    return sealed / f"{case_id}.attempt1.result.tar.gz"


def solver_kwargs(config: dict) -> dict:
    return {
        "solver_tol": config["nonlinear_solver_tol"],
        "solver_max_iter": config["nonlinear_solver_max_iter"],
        "solver_line_search_steps": config["nonlinear_solver_line_search_steps"],
        "solver_min_step_scale": config["nonlinear_solver_min_step_scale"],
    }


def solve_cases(rom, latent0, problem, result, seed: int) -> list[dict]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    rows = []
    for case in result["cases"]:
        mu = np.asarray(case["mu"], dtype=np.float64)
        reference = np.asarray(case["full_order"]["state"], dtype=np.float64)
        t0 = time.perf_counter()
        res = rom.solve(mu, initial=latent0)
        coords = res.solution.detach().cpu().numpy()
        state = rom.decode(coords)
        finite = bool(np.all(np.isfinite(coords)) and np.all(np.isfinite(state)))
        row = {
            "sample": case["sample"],
            "status": "returned" if finite else "nonfinite_return",
            "converged": bool(res.converged),
            "iterations": int(res.iterations),
            "time_sec": float(time.perf_counter() - t0),
            "archived": {k: case["arms"][k]["relative_state_error"] for k in case["arms"]},
        }
        if finite:
            residual = problem.residual_torch(torch.as_tensor(state, dtype=torch.float64), mu)
            row.update(relative_state_error=relative_l2(reference, state), ambient_residual_norm=float(torch.linalg.norm(residual).item()))
        rows.append(row)
    return rows


def build(artifact_like: dict, train_snapshots: np.ndarray, config: dict, b0: np.ndarray | None):
    problem = create_problem(config["problem_name"])
    autoencoder = _instantiate_autoencoder(artifact_like)
    scaler = _instantiate_scaler(artifact_like)
    kwargs = solver_kwargs(config)
    mg = ManifoldGalerkinROM(autoencoder, scaler, problem, **kwargs)
    latent0 = mg.encode(train_snapshots.mean(axis=0))
    if b0 is None:
        return problem, mg, latent0
    pre_m = ManifoldPreconditionedLSPGROM(autoencoder, scaler, problem, preconditioner=torch.as_tensor(b0, dtype=torch.float64), **kwargs)
    return problem, pre_m, latent0


def run_decoder(task: tuple[str, str, str, bool]) -> dict:
    decoder_path, sealed_dir, out_dir, mg_gate = Path(task[0]), Path(task[1]), Path(task[2]), task[3]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    m = NAME.search(decoder_path.name)
    case_id, tag = m["case"], m["tag"]
    decoder = torch.load(decoder_path, map_location="cpu", weights_only=False)
    artifact_like = {"config": decoder["config"], "autoencoder_state_dict": decoder["autoencoder_state_dict"], "scaler": decoder["scaler"]}
    payload = {"case_id": case_id, "tag": tag, "family": m["family"], "seed": int(m["seed"]), "regimes": {}}
    for regime, cid in (("in_domain", case_id), ("ood", case_id.replace("__in_domain__", "__ood__"))):
        result, artifact, mats = read_archive(sealed_path(sealed_dir, cid))
        if artifact["config"]["seed"] != decoder["config"]["seed"] or artifact["problem_name"] != decoder["config"]["problem_name"]:
            raise ValueError(f"{cid} does not match {decoder_path.name}")
        train_snapshots = np.asarray(artifact["train_snapshots"], dtype=np.float64)
        problem, rom, latent0 = build(artifact_like, train_snapshots, result["config"], None if mg_gate else mats["b0"])
        payload["regimes"][regime] = solve_cases(rom, latent0, problem, result, int(result["config"]["seed"]))
    out = out_dir / f"{case_id}.{tag}.{'mg_gate' if mg_gate else 'precond'}.json"
    out.write_text(json.dumps(payload, sort_keys=True, allow_nan=False))
    return {"file": out.name}


def run_archived_gate(task: tuple[str, str]) -> dict:
    archive, out_dir = Path(task[0]), Path(task[1])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result, artifact, mats = read_archive(archive)
    train_snapshots = np.asarray(artifact["train_snapshots"], dtype=np.float64)
    problem, rom, latent0 = build(artifact, train_snapshots, result["config"], mats["b0"])
    rows = solve_cases(rom, latent0, problem, result, int(result["config"]["seed"]))
    diffs = [abs(r["relative_state_error"] - r["archived"]["precond_m_lspg"]) for r in rows]
    payload = {"archive": archive.name, "max_abs_difference": max(diffs), "cases": rows}
    (out_dir / archive.name.replace(".attempt1.result.tar.gz", ".archived_gate.json")).write_text(json.dumps(payload, sort_keys=True, allow_nan=False))
    return {"archive": archive.name, "max_abs_difference": max(diffs)}


def write_csv(out_dir: Path) -> Path:
    path = out_dir / "precond_retrained_cases.csv"
    fields = ["family", "regime", "seed", "sample", "budget", "relative_state_error", "converged", "iterations", "ambient_residual_norm", "e_pol", "e_proj", "pod_g", "precond_pod_lspg", "mg"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for f in sorted(out_dir.glob("*.precond.json")):
            payload = json.loads(f.read_text())
            retrain = json.loads((RETRAIN_JSON / f"{payload['case_id']}.{payload['tag']}.retrain.json").read_text())
            extra = {(c["regime"], c["sample"]): c for c in retrain["cases"]}
            for regime, rows in payload["regimes"].items():
                for r in rows:
                    e = extra[(regime, r["sample"])]
                    writer.writerow({
                        "family": payload["family"], "regime": regime, "seed": payload["seed"], "sample": r["sample"],
                        "budget": budget_label(payload["tag"], payload["seed"]),
                        "relative_state_error": r.get("relative_state_error"), "converged": r["converged"], "iterations": r["iterations"],
                        "ambient_residual_norm": r.get("ambient_residual_norm"), "e_pol": e["e_pol"], "e_proj": e["e_proj"],
                        "pod_g": r["archived"]["pod_g"], "precond_pod_lspg": r["archived"]["precond_pod_lspg"],
                        "mg": e["variants"]["mg:25"].get("relative_state_error"),
                    })
    return path


def seed_ratio(frame, a: str, b: str) -> tuple[float, int]:
    cells = frame.groupby(["seed", "family", "regime"])[[a, b]].mean()
    d = np.log((cells[a] + 1e-12) / (cells[b] + 1e-12)).groupby("seed").mean()
    return float(np.exp(d.mean())), int((d < 0).sum())


def summarize(cases_csv: Path) -> Path:
    import pandas as pd

    frame = pd.read_csv(cases_csv).rename(columns={"relative_state_error": "pre_m"})
    rows = []
    for budget, g in frame.groupby("budget"):
        groups = [("all", "all", g)] + [(f, r, h) for (f, r), h in g.groupby(["family", "regime"])]
        for family, regime, h in groups:
            r_podg, n_podg = seed_ratio(h, "pre_m", "pod_g")
            r_prep, n_prep = seed_ratio(h, "pre_m", "precond_pod_lspg")
            r_mg, n_mg = seed_ratio(h, "pre_m", "mg")
            rel = (h["pre_m"] / h["e_pol"] - 1).abs()
            rows.append({
                "budget": budget, "family": family, "regime": regime, "n_cases": len(h),
                "mean_pre_m": h["pre_m"].mean(), "mean_e_pol": h["e_pol"].mean(), "mean_pod_g": h["pod_g"].mean(), "mean_mg": h["mg"].mean(),
                "ratio_pre_m_pod_g": r_podg, "seeds_pre_m_below_pod_g": n_podg,
                "ratio_pre_m_pre_p": r_prep, "seeds_pre_m_below_pre_p": n_prep,
                "ratio_pre_m_mg": r_mg, "seeds_pre_m_below_mg": n_mg,
                "median_rel_diff_e_pol": rel.median(), "max_rel_diff_e_pol": rel.max(),
                "share_converged": h["converged"].mean(),
            })
    path = cases_csv.with_name("precond_retrained_summary.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoders", type=Path, default=None)
    parser.add_argument("--sealed", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gate", choices=["archived", "mg"], default=None)
    parser.add_argument("--limit", type=int, default=None, help="number of decoders or archives (smoke tests only)")
    parser.add_argument("--csv-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.csv_only:
        cases = write_csv(args.out)
        print("wrote", cases, "\nwrote", summarize(cases))
        return
    if args.sealed is None or (args.gate != "archived" and args.decoders is None):
        parser.error("--sealed and --decoders are required unless --csv-only")
    if args.gate == "archived":
        tasks = [(str(p), str(args.out)) for p in sorted(args.sealed.glob("pv1__*.attempt1.result.tar.gz"))][: args.limit]
        worker = run_archived_gate
    else:
        decoders = sorted(p for p in args.decoders.glob("pv1__*.decoder.pt") if NAME.search(p.name))[: args.limit]
        done = {p.name for p in args.out.glob("*.json")}
        suffix = "mg_gate" if args.gate == "mg" else "precond"
        tasks = [(str(p), str(args.sealed), str(args.out), args.gate == "mg") for p in decoders if p.name.replace(".decoder.pt", f".{suffix}.json") not in done]
        worker = run_decoder
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, info in enumerate(pool.map(worker, tasks), 1):
            print(i, len(tasks), info, flush=True)
    if args.gate is None:
        cases = write_csv(args.out)
        print("wrote", cases, "\nwrote", summarize(cases))


if __name__ == "__main__":
    main()
