#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

FAMILIES = [("bratu_source", "Bratu source"), ("front_layer", "Front layer"),
            ("hydrologic_conductivity", "Hydraulic conductivity"), ("nonlinear_diffusion", "Nonlinear diffusion")]
REGIMES = [("in_domain", "in"), ("ood", "OOD")]
SOLVER_CONTRASTS = [
    ("GN-25-on / Newton-25-on", r"Gauss--Newton / Newton (25 iterations, fallback on)"),
    ("Newton-25-off / Newton-25-on", r"Newton, fallback off / on (25 iterations)"),
    ("Newton-200-on / Newton-25-on", r"Newton, 200 / 25 iterations (fallback on)"),
    ("GN-200-off / Newton-25-on", r"Gauss--Newton (200, off) / Newton (25, on)"),
    ("LM-200 / GN-200-off", r"LM-M-LSPG (200) / Gauss--Newton (200, off)"),
    ("MG-200 / GN-200-off", r"MG (200) / Gauss--Newton (200, off)"),
]
BUDGETS = ["ep60_dataseed", "ep240_dataseed", "ep960_dataseed", "ep960_init100000", "ep960_init200000"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    d = args.data

    print("% Table: solver variants, suite seed ratios")
    sr = {(r["contrast"], r["family"], r["regime"]): r for r in csv.DictReader(open(d / "solver_variants_seed_ratios.csv"))}
    for key, label in SOLVER_CONTRASTS:
        r = sr[(key, "ALL", "suite")]
        cells = [float(x["geometric_ratio"]) for (c, f, _), x in sr.items() if c == key and f != "ALL"]
        assert len(cells) == 8, (key, len(cells))
        print(f"{label} & {float(r['geometric_ratio']):.3f} & {r['favorable_seeds']}/{r['n_seeds']} & {min(cells):.3f}--{max(cells):.3f} \\\\")

    print("% Table: residual comparison by cell")
    rc = {(r["family"], r["regime"]): r for r in csv.DictReader(open(d / "solver_variants_residual_comparison.csv"))}
    for fam, flab in FAMILIES + [("ALL", "All cells")]:
        for reg, rlab in (REGIMES if fam != "ALL" else [("suite", "")]):
            r = rc[(fam, reg)]
            larger = sum(int(r[k]) for k in ("larger_error_smaller_residual", "larger_error_larger_residual", "larger_error_similar_residual"))
            print(f"{flab} & {rlab} & {larger}/{r['n']} & {r['larger_error_smaller_residual']} & {r['larger_error_larger_residual']} & "
                  f"{float(r['median_residual_ratio_least_squares_over_mg']):.2f} \\\\")

    print("% Table: P1 (POD-G / MG) by cell and decoder, and share e_pol < e_proj")
    p1 = {(r["budget"], r["family"], r["regime"]): r for r in csv.DictReader(open(d / "retrain_seed_ratios.csv")) if r["contrast"] == "POD-G / MG"}
    share = {}
    for r in csv.DictReader(open(d / "representation_capacity_cases.csv")):
        share.setdefault(("ep60_dataseed", r["family"], r["regime"]), []).append(float(r["e_pol"]) < float(r["e_proj"]))
    share = {k: float(np.mean(v)) for k, v in share.items()}
    for r in csv.DictReader(open(d / "retrain_cells.csv")):
        if r["arm"] == "mg" and r["share_pol_below_proj"]:
            share[(r["budget"], r["family"], r["regime"])] = float(r["share_pol_below_proj"])
    for fam, flab in FAMILIES + [("ALL", "All cells")]:
        for reg, rlab in (REGIMES if fam != "ALL" else [("suite", "")]):
            parts = [f"{float(p1[(b, fam, reg)]['geometric_ratio']):.2f} ({p1[(b, fam, reg)]['favorable_seeds']})" for b in BUDGETS]
            sh = [f"{100 * share[(b, fam, reg)]:.0f}" for b in ("ep60_dataseed", "ep240_dataseed", "ep960_dataseed")] if fam != "ALL" else ["", "", ""]
            print(f"{flab} & {rlab} & " + " & ".join(parts) + " & " + "/".join(sh) + " \\\\")

    print("% Suite: MG / Newton-M-LSPG and MG / GN-M-LSPG by decoder")
    rs = {(r["budget"], r["contrast"], r["family"]): r for r in csv.DictReader(open(d / "retrain_seed_ratios.csv"))}
    for b in BUDGETS:
        out = []
        for c in ("MG / Newton-M-LSPG", "MG / GN-M-LSPG"):
            r = rs.get((b, c, "ALL"))
            out.append(f"{float(r['geometric_ratio']):.3f} ({r['favorable_seeds']})" if r else "--")
        print(b, " & ".join(out))

    print("% Training: reconstruction error and final/min loss at 960 epochs")
    tr = defaultdict(list)
    for r in csv.DictReader(open(d / "retrain_training.csv")):
        tr[(r["family"], r["budget"])].append(r)
    for fam, flab in FAMILIES:
        rec = {b: np.mean([float(x["train_reconstruction_error"]) for x in tr[(fam, b)]]) for b in BUDGETS[1:]}
        arch = np.mean([float(x["archived_train_reconstruction_error"]) for x in tr[(fam, BUDGETS[1])]])
        over = [sum(float(x["final_loss"]) / float(x["min_loss"]) > 1.1 for x in tr[(fam, b)]) for b in BUDGETS[2:]]
        mx = max(float(x["final_loss"]) / float(x["min_loss"]) for b in BUDGETS[2:] for x in tr[(fam, b)])
        print(f"{flab}: recon 60={arch:.3f} 240={rec[BUDGETS[1]]:.3f} 960={min(rec[b] for b in BUDGETS[2:]):.3f}-{max(rec[b] for b in BUDGETS[2:]):.3f}; final>1.1*min at 960: {over}; max final/min {mx:.1f}")


if __name__ == "__main__":
    main()
