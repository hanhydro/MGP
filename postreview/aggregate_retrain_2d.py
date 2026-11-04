#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DIRECTIONS = [
    ("D1", "in_domain", "mg", "pod_g"),
    ("D2", "in_domain", "mg", "newton_m_lspg"),
    ("D4", "ood", "pod_g", "mg"),
    ("D5", "ood", "newton_m_lspg", "mg"),
]
ARM = {"mg:25": "mg", "newton:25:on": "newton_m_lspg"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    training, cases = [], []
    archived_rows = {}
    for path in sorted(args.results.glob("*.retrain2d.json")):
        d = json.loads(path.read_text())
        tag = f"{d['schedule']}{d['epochs']}"
        training.append({"seed": d["seed"], "schedule": tag, **d["training"]})
        for c in d["cases"]:
            key = (d["seed"], c["regime"], c["sample"])
            for arm, err in c["archived"].items():
                archived_rows[key + (arm,)] = err
            for ckpt, e in c["decoders"].items():
                row = {"seed": d["seed"], "decoder": f"{tag}_{ckpt}", "regime": c["regime"], "sample": c["sample"],
                       "e_pol": e["e_pol"], "e_ae": e["e_ae"], "e_proj": c["e_proj"],
                       "reference_qoi_abs_difference": c["reference_qoi_abs_difference"],
                       "pod_g": c["archived"]["pod_g"], "pod_lspg": c["archived"]["pod_lspg"]}
                for spec, a in e["arms"].items():
                    row[ARM.get(spec, spec)] = a.get("relative_state_error")
                    row[ARM.get(spec, spec) + "_converged"] = a.get("converged")
                cases.append(row)
    archived = defaultdict(dict)
    for (seed, regime, sample, arm), err in archived_rows.items():
        archived[(seed, regime, sample)][arm] = err
    for (seed, regime, sample), arms in archived.items():
        cases.append({"seed": seed, "decoder": "archived", "regime": regime, "sample": sample, "e_pol": None, "e_ae": None,
                      "e_proj": None, "reference_qoi_abs_difference": None, "pod_g": arms["pod_g"], "pod_lspg": arms["pod_lspg"],
                      "mg": arms["mg"], "mg_converged": None, "newton_m_lspg": arms["newton_m_lspg"], "newton_m_lspg_converged": None})

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "retrain2d_training.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(training[0]))
        w.writeheader()
        w.writerows(training)
    fields = sorted({k for r in cases for k in r}, key=lambda k: (["seed", "decoder", "regime", "sample"].index(k) if k in ("seed", "decoder", "regime", "sample") else 10, k))
    with open(args.out / "retrain2d_cases.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(cases)

    means = defaultdict(list)
    for r in cases:
        for arm in ("mg", "newton_m_lspg", "pod_g", "pod_lspg"):
            if r.get(arm) is not None:
                means[(r["decoder"], r["seed"], r["regime"], arm)].append(r[arm])
    means = {k: float(np.mean(v)) for k, v in means.items()}
    decoders = sorted({k[0] for k in means})
    out = []
    for dec in decoders:
        for did, regime, a, b in DIRECTIONS:
            logs = []
            for seed in sorted({k[1] for k in means if k[0] == dec}):
                ka, kb = (dec, seed, regime, a), (dec, seed, regime, b)
                if ka in means and kb in means:
                    logs.append(np.log((means[ka] + 1e-12) / (means[kb] + 1e-12)))
            if logs:
                logs = np.array(logs)
                out.append({"decoder": dec, "direction": did, "regime": regime, "contrast": f"{a} / {b}", "n_seeds": len(logs),
                            "favorable_seeds": int(np.sum(logs < 0)), "geometric_ratio": float(np.exp(logs.mean()))})
    with open(args.out / "retrain2d_seed_ratios.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(f"{r['decoder']:18s} {r['direction']} {r['contrast']:22s} {r['favorable_seeds']}/{r['n_seeds']} favorable, ratio {r['geometric_ratio']:.3f}")


if __name__ == "__main__":
    main()
