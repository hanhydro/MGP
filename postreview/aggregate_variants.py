#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CONTRASTS = [
    ("MG-200 / GN-25-on", "mg:200", "gn:25:on"),
    ("MG(archived) / GN-25-on", "archived:mg", "gn:25:on"),
    ("GN-25-on / Newton-25-on", "gn:25:on", "newton:25:on"),
    ("Newton-25-off / Newton-25-on", "newton:25:off", "newton:25:on"),
    ("Newton-200-on / Newton-25-on", "newton:200:on", "newton:25:on"),
    ("GN-200-off / Newton-25-on", "gn:200:off", "newton:25:on"),
    ("MG-200 / GN-200-off", "mg:200", "gn:200:off"),
    ("LM-200 / GN-200-off", "lm:200", "gn:200:off"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-errors", type=Path, default=Path(__file__).resolve().parents[1] / "supplementary_data/representation_capacity_cases.csv")
    args = parser.parse_args()

    start = {}
    if args.start_errors.exists():
        for r in csv.DictReader(open(args.start_errors)):
            start[(r["family"], r["regime"], int(r["seed"]), int(r["sample"]))] = float(r["e_lstart"])

    rows = []
    for path in sorted(args.results.glob("*.variants.json")):
        d = json.loads(path.read_text())
        for case in d["cases"]:
            key = (d["problem"], d["regime"], d["seed"], case["sample"])
            for spec, v in case["variants"].items():
                rows.append({
                    "family": d["problem"], "regime": d["regime"], "seed": d["seed"], "sample": case["sample"], "variant": spec,
                    "status": v.get("status"), "converged": v.get("converged"), "stagnated": v.get("stagnated"),
                    "iterations": v.get("iterations"), "n_fallback": v.get("n_fallback"), "n_nondescent": v.get("n_nondescent"), "n_trace": v.get("n_trace"),
                    "relative_state_error": v.get("relative_state_error"), "ambient_residual_norm": v.get("ambient_residual_norm"),
                    "time_sec": v.get("time_sec"), "e_lstart": start.get(key),
                })
            for arm, err in case["archived"].items():
                rows.append({"family": d["problem"], "regime": d["regime"], "seed": d["seed"], "sample": case["sample"], "variant": "archived:" + arm,
                             "status": "archived", "relative_state_error": err, "e_lstart": start.get(key)})
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "solver_variants_cases.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["family"], r["regime"], r["variant"])].append(r)
    cells = []
    for (fam, reg, spec), g in sorted(groups.items()):
        err = np.array([r["relative_state_error"] for r in g if r["relative_state_error"] is not None], dtype=float)
        pairs = np.array([(r["relative_state_error"], r["e_lstart"]) for r in g if r["relative_state_error"] is not None and r["e_lstart"] is not None], dtype=float).reshape(-1, 2)
        cells.append({
            "family": fam, "regime": reg, "variant": spec, "n": len(g), "n_finite": len(err),
            "mean_error": err.mean() if len(err) else None,
            "share_stopping_unsuccessful": np.mean([not r["converged"] for r in g if r.get("converged") is not None]) if spec[:8] != "archived" else None,
            "share_stagnated": np.mean([bool(r["stagnated"]) for r in g if r.get("stagnated") is not None]) if spec[:8] != "archived" else None,
            "share_iterations_with_fallback": (sum(r["n_fallback"] or 0 for r in g) / max(1, sum(r["n_trace"] or 0 for r in g))) if spec[:8] != "archived" else None,
            "share_iterations_nondescent": (sum(r["n_nondescent"] or 0 for r in g) / max(1, sum(r["n_trace"] or 0 for r in g))) if spec[:8] != "archived" else None,
            "share_error_above_one": np.mean(err > 1.0) if len(err) else None,
            "share_error_above_start": np.mean(pairs[:, 0] > pairs[:, 1] * (1 + 1e-12)) if len(pairs) else None,
            "mean_time_sec": np.mean([r["time_sec"] for r in g if r.get("time_sec") is not None]) if spec[:8] != "archived" else None,
        })
    with open(args.out / "solver_variants_cells.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0]))
        w.writeheader()
        w.writerows(cells)

    seedcell = defaultdict(list)
    for r in rows:
        if r["relative_state_error"] is not None:
            seedcell[(r["variant"], r["family"], r["regime"], r["seed"])].append(r["relative_state_error"])
    seedmean = {k: float(np.mean(v)) for k, v in seedcell.items()}
    out = []
    for label, a, b in CONTRASTS:
        per_seed_suite = defaultdict(list)
        cellkeys = sorted({(k[1], k[2]) for k in seedmean if k[0] == a})
        for fam, reg in cellkeys:
            seeds = sorted({k[3] for k in seedmean if k[0] == a and k[1] == fam and k[2] == reg})
            logs = []
            for s in seeds:
                if (b, fam, reg, s) in seedmean:
                    lr = np.log((seedmean[(a, fam, reg, s)] + 1e-12) / (seedmean[(b, fam, reg, s)] + 1e-12))
                    logs.append(lr)
                    per_seed_suite[s].append(lr)
            if logs:
                logs = np.array(logs)
                out.append({"contrast": label, "family": fam, "regime": reg, "n_seeds": len(logs), "favorable_seeds": int(np.sum(logs < 0)), "geometric_ratio": float(np.exp(logs.mean()))})
        if per_seed_suite:
            D = np.array([np.mean(v) for v in per_seed_suite.values()])
            out.append({"contrast": label, "family": "ALL", "regime": "suite", "n_seeds": len(D), "favorable_seeds": int(np.sum(D < 0)), "geometric_ratio": float(np.exp(D.mean()))})
    with open(args.out / "solver_variants_seed_ratios.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["contrast", "family", "regime", "n_seeds", "favorable_seeds", "geometric_ratio"])
        w.writeheader()
        w.writerows(out)
    mg_pairs = defaultdict(dict)
    for r in rows:
        if r["variant"] in ("mg:200", "archived:mg") and r["relative_state_error"] is not None:
            mg_pairs[(r["family"], r["regime"], r["seed"], r["sample"])][r["variant"]] = r["relative_state_error"]
    diffs = [abs(v["mg:200"] - v["archived:mg"]) for v in mg_pairs.values() if len(v) == 2]
    if diffs:
        print(f"MG gate: {sum(d <= 1e-9 for d in diffs)}/{len(diffs)} cases with |mg:200 - archived MG| <= 1e-9 (differences expected only where archived MG hit the 25-iteration cap)")
    newton_gate = [r for r in rows if r["variant"] == "newton:25:on"]
    for r in out:
        if r["family"] == "ALL":
            print(f"{r['contrast']:34s} suite: {r['favorable_seeds']}/{r['n_seeds']} favorable, ratio {r['geometric_ratio']:.3f}")


if __name__ == "__main__":
    main()
