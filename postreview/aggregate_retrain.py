#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

LEARNED = {"mg:25": "mg", "newton:25:on": "newton_m_lspg", "gn:25:on": "gn_m_lspg", "lm:25": "lm_m_lspg"}
CONTRASTS = [("POD-G / MG", "pod_g", "mg"), ("MG / Newton-M-LSPG", "mg", "newton_m_lspg"), ("MG / GN-M-LSPG", "mg", "gn_m_lspg"), ("POD-LSPG / MG", "pod_lspg", "mg")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cases, training = [], []
    baseline_done = set()
    for path in sorted(args.results.glob("*.retrain.json")):
        d = json.loads(path.read_text())
        tag = f"ep{d['epochs']}_init{d['init_seed']}"
        budget = "ep%d_%s" % (d["epochs"], "dataseed" if d["init_seed"] == d["seed"] else "init%d" % (d["init_seed"] - d["seed"]))
        training.append({"family": d["problem"], "seed": d["seed"], "epochs": d["epochs"], "init_seed": d["init_seed"], "budget": budget, **d["training"]})
        for c in d["cases"]:
            key = (d["problem"], c["regime"], d["seed"], c["sample"])
            for spec, arm in LEARNED.items():
                v = c["variants"].get(spec, {})
                cases.append({"family": d["problem"], "regime": c["regime"], "seed": d["seed"], "sample": c["sample"], "budget": budget, "arm": arm,
                              "relative_state_error": v.get("relative_state_error"), "converged": v.get("converged"), "e_pol": c["e_pol"], "e_ae": c["e_ae"], "e_proj": c["e_proj"]})
            if key not in baseline_done:
                baseline_done.add(key)
                for arm, err in c["archived"].items():
                    cases.append({"family": d["problem"], "regime": c["regime"], "seed": d["seed"], "sample": c["sample"], "budget": "ep60_dataseed", "arm": arm,
                                  "relative_state_error": err, "converged": None, "e_pol": None, "e_ae": None, "e_proj": c["e_proj"]})
            for arm in ("pod_g", "pod_lspg"):
                cases.append({"family": d["problem"], "regime": c["regime"], "seed": d["seed"], "sample": c["sample"], "budget": budget, "arm": arm,
                              "relative_state_error": c["archived"][arm], "converged": None, "e_pol": None, "e_ae": None, "e_proj": c["e_proj"]})
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("retrain_cases.csv", cases), ("retrain_training.csv", training)):
        with open(args.out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    groups = defaultdict(list)
    for r in cases:
        groups[(r["family"], r["regime"], r["budget"], r["arm"])].append(r)
    cells = []
    for (fam, reg, budget, arm), g in sorted(groups.items()):
        err = np.array([r["relative_state_error"] for r in g if r["relative_state_error"] is not None], dtype=float)
        pol = [(r["e_pol"], r["e_proj"]) for r in g if r["e_pol"] is not None]
        cells.append({"family": fam, "regime": reg, "budget": budget, "arm": arm, "n": len(err), "mean_error": err.mean() if len(err) else None,
                      "share_stopping_unsuccessful": np.mean([not r["converged"] for r in g if r["converged"] is not None]) if any(r["converged"] is not None for r in g) else None,
                      "mean_e_pol": np.mean([p for p, _ in pol]) if pol else None, "mean_e_proj": np.mean([q for _, q in pol]) if pol else None,
                      "share_pol_below_proj": np.mean([p < q for p, q in pol]) if pol else None})
    with open(args.out / "retrain_cells.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0]))
        w.writeheader()
        w.writerows(cells)

    seedmean = defaultdict(list)
    for r in cases:
        if r["relative_state_error"] is not None:
            seedmean[(r["budget"], r["arm"], r["family"], r["regime"], r["seed"])].append(r["relative_state_error"])
    seedmean = {k: float(np.mean(v)) for k, v in seedmean.items()}
    budgets = sorted({k[0] for k in seedmean})
    out = []
    for budget in budgets:
        for label, a, b in CONTRASTS:
            suite = defaultdict(list)
            cellkeys = sorted({(k[2], k[3]) for k in seedmean if k[0] == budget and k[1] == a})
            for fam, reg in cellkeys:
                logs = []
                for s in sorted({k[4] for k in seedmean if k[0] == budget and k[1] == a and k[2] == fam and k[3] == reg}):
                    if (budget, b, fam, reg, s) in seedmean:
                        lr = np.log((seedmean[(budget, a, fam, reg, s)] + 1e-12) / (seedmean[(budget, b, fam, reg, s)] + 1e-12))
                        logs.append(lr)
                        suite[s].append(lr)
                if logs:
                    out.append({"budget": budget, "contrast": label, "family": fam, "regime": reg, "n_seeds": len(logs), "favorable_seeds": int(np.sum(np.array(logs) < 0)), "geometric_ratio": float(np.exp(np.mean(logs)))})
            if suite:
                D = np.array([np.mean(v) for v in suite.values()])
                out.append({"budget": budget, "contrast": label, "family": "ALL", "regime": "suite", "n_seeds": len(D), "favorable_seeds": int(np.sum(D < 0)), "geometric_ratio": float(np.exp(D.mean()))})
    with open(args.out / "retrain_seed_ratios.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["budget", "contrast", "family", "regime", "n_seeds", "favorable_seeds", "geometric_ratio"])
        w.writeheader()
        w.writerows(out)
    for r in out:
        if r["family"] == "ALL":
            print(f"{r['budget']:18s} {r['contrast']:20s} suite: {r['favorable_seeds']}/{r['n_seeds']} favorable, ratio {r['geometric_ratio']:.3f}")


if __name__ == "__main__":
    main()
