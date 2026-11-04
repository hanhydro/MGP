#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEALED = ROOT / "eval/prospective_weight_validation_2026-07-17/first_look_input/sealed"
DATA = Path(__file__).resolve().parents[1] / "supplementary_data"
ARMS = ["mg", "newton_m_lspg", "lm_m_lspg", "pod_g", "pod_lspg", "precond_m_lspg", "precond_pod_lspg"]
LEARNED = {"mg", "newton_m_lspg", "lm_m_lspg", "precond_m_lspg"}


def main() -> None:
    converged = {}
    for tar_path in sorted(SEALED.glob("*.tar.gz")):
        with tarfile.open(tar_path, "r:gz") as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith(".result.json"))
            result = json.loads(tar.extractfile(member).read())
        cfg = result["config"]
        for case in result["cases"]:
            for arm in ARMS:
                converged[(cfg["problem_name"], cfg["test_regime"], cfg["seed"], case["sample"], arm)] = bool(case["arms"][arm]["converged"])

    groups = defaultdict(list)
    for r in csv.DictReader(open(DATA / "representation_capacity_cases.csv")):
        for arm in ARMS:
            key = (r["family"], r["regime"], int(r["seed"]), int(r["sample"]), arm)
            start = float(r["e_lstart"] if arm in LEARNED else r["e_start"])
            err = float(r[arm])
            status = "successful" if converged[key] else "unsuccessful"
            for cell in ((r["family"], r["regime"]), ("ALL", "ALL")):
                groups[(cell[0], cell[1], arm, status)].append((err, start))
    rows = []
    for (fam, reg, arm, status), g in sorted(groups.items()):
        e = np.array([x for x, _ in g]); s = np.array([y for _, y in g])
        rows.append({"family": fam, "regime": reg, "arm": arm, "stopping": status, "n": len(g),
                     "mean_error": e.mean(), "share_above_one": np.mean(e > 1.0), "share_above_start": np.mean(e > s)})
    with open(DATA / "stopping_status_split.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("ALL cells pooled:")
    for r in rows:
        if r["family"] == "ALL" and r["arm"] in ("mg", "newton_m_lspg", "lm_m_lspg", "precond_m_lspg"):
            print(f"  {r['arm']:16s} {r['stopping']:12s} n={r['n']:5d} mean err {r['mean_error']:.3f}  >1 {100*r['share_above_one']:5.1f}%  >start {100*r['share_above_start']:5.1f}%")
    print("per cell (Newton and LM):")
    for r in rows:
        if r["family"] != "ALL" and r["arm"] in ("newton_m_lspg", "lm_m_lspg"):
            print(f"  {r['family'][:14]:14s} {r['regime']:9s} {r['arm']:14s} {r['stopping']:12s} n={r['n']:4d} err {r['mean_error']:.3f} >1 {100*r['share_above_one']:5.1f}% >start {100*r['share_above_start']:5.1f}%")


if __name__ == "__main__":
    main()
