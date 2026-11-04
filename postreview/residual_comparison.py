#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

REL = 0.01


def classify(e_ls: float, e_mg: float, r_ls: float, r_mg: float) -> str:
    if e_ls <= e_mg * (1 + REL):
        return "least_squares_error_not_larger"
    if r_ls < r_mg * (1 - REL):
        return "larger_error_smaller_residual"
    if r_ls > r_mg * (1 + REL):
        return "larger_error_larger_residual"
    return "larger_error_similar_residual"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--least-squares", default="gn:200:off")
    parser.add_argument("--galerkin", default="mg:200")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.results.glob("*.variants.json")):
        d = json.loads(path.read_text())
        for case in d["cases"]:
            ls, mg = case["variants"][args.least_squares], case["variants"][args.galerkin]
            rows.append({
                "family": d["problem"], "regime": d["regime"], "seed": d["seed"], "sample": case["sample"],
                "error_least_squares": ls["relative_state_error"], "error_mg": mg["relative_state_error"],
                "residual_least_squares": ls["ambient_residual_norm"], "residual_mg": mg["ambient_residual_norm"],
                "least_squares_converged": ls["converged"], "mg_converged": mg["converged"],
                "class": classify(ls["relative_state_error"], mg["relative_state_error"], ls["ambient_residual_norm"], mg["ambient_residual_norm"]),
            })
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "solver_variants_residual_comparison_cases.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    classes = ["least_squares_error_not_larger", "larger_error_smaller_residual", "larger_error_larger_residual", "larger_error_similar_residual"]
    cells = sorted({(r["family"], r["regime"]) for r in rows}) + [("ALL", "suite")]
    summary = []
    for fam, reg in cells:
        g = [r for r in rows if fam == "ALL" or (r["family"], r["regime"]) == (fam, reg)]
        c = Counter(r["class"] for r in g)
        ratio = np.array([r["residual_least_squares"] / r["residual_mg"] for r in g])
        summary.append({"family": fam, "regime": reg, "n": len(g), **{k: c.get(k, 0) for k in classes},
                        "median_residual_ratio_least_squares_over_mg": float(np.median(ratio))})
    with open(args.out / "solver_variants_residual_comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(f"{s['family'][:24]:25s}{s['regime']:10s} n={s['n']:4d} " + " ".join(f"{k}={s[k]}" for k in classes[1:]) + f" median r_LS/r_MG={s['median_residual_ratio_least_squares_over_mg']:.3f}")


if __name__ == "__main__":
    main()
