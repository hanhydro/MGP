#!/usr/bin/env python3
"""Derive the retrospective acceptance-revision statistics and design audits.

The submission package uses a compact, claim-bearing input distilled from the tracked
retrospective shards (seeds 0--9).  By default this script reads that compact input and
rewrites the machine-readable and Markdown summaries.  An internal checkout that has the
full tracked shards can also refresh the compact input first::

    python MGP_revision/code/scripts/derive_acceptance_revision_statistics.py \
        --refresh-from-shards artifacts/cache

Only the fixed retrospective shard directories named in ``SOURCE_DIRS`` are ever read.
In particular, this script does not discover or read prospective-validation output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
DEFAULT_ARTIFACT_DIR = PACKAGE_ROOT / "artifacts"
DEFAULT_INPUT = DEFAULT_ARTIFACT_DIR / "acceptance_revision_retrospective_inputs.json"
DEFAULT_JSON = DEFAULT_ARTIFACT_DIR / "acceptance_statistics.json"
DEFAULT_MARKDOWN = DEFAULT_ARTIFACT_DIR / "acceptance_statistics.md"

SEEDS = tuple(range(10))
SAMPLES_PER_SEED = 12
RTOLS = ("1e-10", "1e-12", "1e-14")
THRESHOLDS = (1e-3, 1e-4, 1e-5)
SOURCE_DIRS = (
    "lm_solver_shards",
    "lm_solver_2d_shards",
    "precond_lspg_shards",
    "precond_lspg_2d_shards",
    "precond_pod_lspg_shards",
    "unified_rank_shards",
)

CASES = (
    ("bratu_source", "Bratu-source", "reference_bratu_source"),
    ("front_layer", "Front-layer", "reference_front_layer"),
    ("hydrologic_conductivity", "Hydraulic K", "reference_hydrologic_conductivity"),
    ("nonlinear_diffusion", "Nonlinear diffusion", "reference_nonlinear_diffusion"),
)
REGIMES = ("in_domain", "ood")
METRICS = (("state", "rel_err"), ("qoi", "qoi_err"))
ALL_CELLS = tuple(
    (problem, display, preset, regime, "1d")
    for problem, display, preset in CASES
    for regime in REGIMES
) + (
    ("two_d_burgers", "2D Burgers", "two_d_burgers", "in_domain", "2d"),
    ("two_d_burgers", "2D Burgers", "two_d_burgers", "ood", "2d"),
)
ARM_LABELS = {
    "mg": "MG",
    "newton_m": "Newton-M-LSPG",
    "lm_m": "LM-M-LSPG",
    "pod_g": "POD-G",
    "pod_l": "POD-LSPG",
    "pre_m": "pre-M",
    "pre_p": "pre-P",
}
CONTROLLED_CONTRASTS = (
    ("trial_galerkin", "pod_g", "mg", "trial set under Galerkin"),
    (
        "trial_minimum_residual",
        "pod_l",
        "newton_m",
        "trial set under the Newton minimum-residual implementation",
    ),
    ("equation_manifold", "mg", "newton_m", "equation on learned trial set"),
    ("equation_affine", "pod_g", "pod_l", "equation on affine trial set"),
    ("metric_manifold", "pre_m", "lm_m", "metric on learned trial set, LM matched"),
    ("metric_affine", "pre_p", "pod_l", "metric on affine trial set, Newton matched"),
    (
        "algorithm_manifold",
        "lm_m",
        "newton_m",
        "algorithm on learned least-squares equation",
    ),
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_source_path(path: Path, cache_dir: Path) -> str:
    repo_root = cache_dir.parent.parent
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _require_rows(rows: list[dict[str, Any]], *, path: Path, label: str) -> None:
    if len(rows) != SAMPLES_PER_SEED:
        raise ValueError(
            f"{path}: expected {SAMPLES_PER_SEED} {label} rows, found {len(rows)}"
        )


def build_compact_input(cache_dir: Path) -> dict[str, Any]:
    """Distill only the retrospective acceptance-review values from raw shards."""
    cache_dir = cache_dir.resolve()
    for dirname in SOURCE_DIRS:
        if not (cache_dir / dirname).is_dir():
            raise FileNotFoundError(
                f"missing retrospective shard directory: {cache_dir / dirname}"
            )

    sources: dict[str, str] = {}
    m1_seed_means: list[dict[str, Any]] = []
    for problem, display, preset, regime, study_kind in ALL_CELLS:
        for seed in SEEDS:
            path = (
                cache_dir / "lm_solver_shards" / f"{preset}_{regime}_seed{seed}.json"
                if study_kind == "1d"
                else cache_dir / "lm_solver_2d_shards" / f"seed{seed}_{regime}.json"
            )
            payload = _read_json(path)
            spec = payload.get("spec", {})
            if (
                spec.get("seed") != seed
                or spec.get("problem_name") != problem
                or spec.get("test_regime") != regime
            ):
                raise ValueError(
                    f"{path}: shard specification does not match its filename"
                )
            rows = payload.get("rows", [])
            by_solver: dict[str, list[dict[str, Any]]] = {}
            for solver in ("mg", "pod_lspg"):
                selected = [row for row in rows if row.get("solver") == solver]
                _require_rows(selected, path=path, label=solver)
                by_solver[solver] = selected
            m1_seed_means.append(
                {
                    "display": display,
                    "problem": problem,
                    "regime": regime,
                    "study_kind": study_kind,
                    "seed": seed,
                    "n_samples": SAMPLES_PER_SEED,
                    "mg_state_mean": fmean(row["rel_err"] for row in by_solver["mg"]),
                    "pod_lspg_state_mean": fmean(
                        row["rel_err"] for row in by_solver["pod_lspg"]
                    ),
                    "mg_qoi_mean": fmean(row["qoi_err"] for row in by_solver["mg"]),
                    "pod_lspg_qoi_mean": fmean(
                        row["qoi_err"] for row in by_solver["pod_lspg"]
                    ),
                }
            )
            sources[_relative_source_path(path, cache_dir)] = _sha256(path)

    m2_samples: list[dict[str, Any]] = []
    for seed in SEEDS:
        precond_path = cache_dir / "precond_lspg_2d_shards" / f"seed{seed}_ood.json"
        unified_path = (
            cache_dir / "unified_rank_shards" / f"two_d_burgers_ood_seed{seed}.json"
        )
        precond = _read_json(precond_path)
        unified = _read_json(unified_path)
        if precond.get("seed") != seed or unified.get("seed") != seed:
            raise ValueError(
                f"seed mismatch in 2D-OOD retrospective shards for seed {seed}"
            )
        precond_rows = precond.get("rows", [])
        unified_rows = unified.get("rows", [])
        mg_rows = precond.get("spec", {}).get("published_cases", [])
        _require_rows(precond_rows, path=precond_path, label="preconditioned")
        _require_rows(mg_rows, path=precond_path, label="paired MG")
        _require_rows(unified_rows, path=unified_path, label="unified-rank")
        for expected_sample, (pre_row, mg_row, rank_row) in enumerate(
            zip(precond_rows, mg_rows, unified_rows)
        ):
            if (
                pre_row.get("sample") != expected_sample
                or rank_row.get("sample") != expected_sample
            ):
                raise ValueError(
                    f"seed {seed}: sample ordering mismatch at expected sample {expected_sample}"
                )
            arm = pre_row["precond"]
            rank_arm = rank_row["arms"]["precond"]
            m2_samples.append(
                {
                    "seed": seed,
                    "sample": expected_sample,
                    "precond_rel_err": arm["rel_err"],
                    "mg_rel_err": mg_row["mg_error_l2"],
                    "native_inv_stationarity": arm["inv_stationarity"],
                    "native_converged_inv": arm["converged_inv"],
                    "unified_inv_stationarity": {
                        rtol: rank_arm["s"][rtol] for rtol in RTOLS
                    },
                }
            )
        sources[_relative_source_path(precond_path, cache_dir)] = _sha256(precond_path)
        sources[_relative_source_path(unified_path, cache_dir)] = _sha256(unified_path)

    controlled_seed_means: list[dict[str, Any]] = []
    metric_comparison_samples: list[dict[str, Any]] = []
    for problem, display, preset, regime, study_kind in ALL_CELLS:
        for seed in SEEDS:
            if study_kind == "1d":
                lm_path = (
                    cache_dir
                    / "lm_solver_shards"
                    / f"{preset}_{regime}_seed{seed}.json"
                )
                pre_m_path = (
                    cache_dir
                    / "precond_lspg_shards"
                    / f"{preset}_{regime}_seed{seed}.json"
                )
                pre_p_path = (
                    cache_dir
                    / "precond_pod_lspg_shards"
                    / f"{problem}_{regime}_seed{seed}.json"
                )
            else:
                lm_path = (
                    cache_dir / "lm_solver_2d_shards" / f"seed{seed}_{regime}.json"
                )
                pre_m_path = (
                    cache_dir / "precond_lspg_2d_shards" / f"seed{seed}_{regime}.json"
                )
                pre_p_path = (
                    cache_dir
                    / "precond_pod_lspg_shards"
                    / f"two_d_burgers_{regime}_seed{seed}.json"
                )

            lm_payload = _read_json(lm_path)
            pre_m_payload = _read_json(pre_m_path)
            pre_p_payload = _read_json(pre_p_path)
            lm_rows = lm_payload.get("rows", [])
            solver_rows: dict[str, list[dict[str, Any]]] = {}
            for solver in ("mg", "mgp_lspg", "lm_lspg"):
                selected_rows = [row for row in lm_rows if row.get("solver") == solver]
                _require_rows(selected_rows, path=lm_path, label=solver)
                if [row.get("sample") for row in selected_rows] != list(
                    range(SAMPLES_PER_SEED)
                ):
                    raise ValueError(f"{lm_path}: {solver} sample ordering mismatch")
                solver_rows[solver] = selected_rows
            pre_m_rows = pre_m_payload.get("rows", [])
            pre_p_rows = pre_p_payload.get("rows", [])
            _require_rows(pre_m_rows, path=pre_m_path, label="pre-M")
            _require_rows(pre_p_rows, path=pre_p_path, label="affine controls")
            if [row.get("sample") for row in pre_m_rows] != list(
                range(SAMPLES_PER_SEED)
            ):
                raise ValueError(f"{pre_m_path}: pre-M sample ordering mismatch")
            if [row.get("sample") for row in pre_p_rows] != list(
                range(SAMPLES_PER_SEED)
            ):
                raise ValueError(f"{pre_p_path}: affine sample ordering mismatch")

            arm_means = {
                "mg": fmean(row["rel_err"] for row in solver_rows["mg"]),
                "newton_m": fmean(row["rel_err"] for row in solver_rows["mgp_lspg"]),
                "lm_m": fmean(row["rel_err"] for row in solver_rows["lm_lspg"]),
                "pod_g": fmean(row["pod_g"]["rel_err"] for row in pre_p_rows),
                "pod_l": fmean(row["pod_lspg"]["rel_err"] for row in pre_p_rows),
                "pre_m": fmean(row["precond"]["rel_err"] for row in pre_m_rows),
                "pre_p": fmean(
                    row["precond_pod_lspg"]["rel_err"] for row in pre_p_rows
                ),
            }
            if not all(
                math.isfinite(value) and value >= 0.0 for value in arm_means.values()
            ):
                raise ValueError(
                    f"non-finite or negative arm mean for {problem}/{regime}/seed{seed}"
                )
            controlled_seed_means.append(
                {
                    "display": display,
                    "problem": problem,
                    "regime": regime,
                    "study_kind": study_kind,
                    "seed": seed,
                    "n_samples": SAMPLES_PER_SEED,
                    "arm_state_error_means": arm_means,
                }
            )
            for sample in range(SAMPLES_PER_SEED):
                metric_comparison_samples.append(
                    {
                        "display": display,
                        "problem": problem,
                        "regime": regime,
                        "study_kind": study_kind,
                        "seed": seed,
                        "sample": sample,
                        "state_error": {
                            "mg": solver_rows["mg"][sample]["rel_err"],
                            "lm_m": solver_rows["lm_lspg"][sample]["rel_err"],
                            "pre_m": pre_m_rows[sample]["precond"]["rel_err"],
                            "pod_l": pre_p_rows[sample]["pod_lspg"]["rel_err"],
                            "pre_p": pre_p_rows[sample]["precond_pod_lspg"]["rel_err"],
                        },
                    }
                )
            for path in (lm_path, pre_m_path, pre_p_path):
                sources[_relative_source_path(path, cache_dir)] = _sha256(path)

    return {
        "schema_version": 2,
        "record": {
            "kind": "retrospective_development_record",
            "seeds": list(SEEDS),
            "prospective_outputs_read": False,
            "warning": (
                "These values come only from the frozen seeds 0--9 retrospective shards; "
                "they are not prospective-validation results."
            ),
        },
        "provenance": {
            "generator": "MGP_revision/code/scripts/derive_acceptance_revision_statistics.py",
            "source_directories": list(SOURCE_DIRS),
            "source_files_sha256": dict(sorted(sources.items())),
        },
        "m1_seed_means": m1_seed_means,
        "m2_2d_ood_samples": m2_samples,
        "controlled_contrast_seed_means": controlled_seed_means,
        "metric_comparison_samples": metric_comparison_samples,
    }


def _exact_two_sided_sign_p(negative: int, positive: int) -> float | None:
    n = negative + positive
    if n == 0:
        return None
    k = min(negative, positive)
    probability = 2.0 * sum(math.comb(n, j) for j in range(k + 1)) / (2**n)
    return min(1.0, probability)


def _holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda idx: (p_values[idx], idx))
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p_values[idx])
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted


def _linear_percentile(values: Iterable[float], probability: float) -> float:
    """Hyndman--Fan type-7 percentile, matching the manuscript convention."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a percentile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (1.0 - weight) * ordered[lower] + weight * ordered[upper]


def _group_by(
    items: Iterable[dict[str, Any]], *keys: str
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(tuple(item[key] for key in keys), []).append(item)
    return grouped


def derive_statistics(compact: dict[str, Any]) -> dict[str, Any]:
    if compact.get("record", {}).get("prospective_outputs_read") is not False:
        raise ValueError(
            "compact input is not explicitly sealed from prospective outputs"
        )

    seed_means = compact["m1_seed_means"]
    grouped = _group_by(seed_means, "problem", "regime")
    contrasts: list[dict[str, Any]] = []
    for problem, display, _preset, regime, study_kind in ALL_CELLS:
        rows = sorted(grouped[(problem, regime)], key=lambda row: row["seed"])
        if [row["seed"] for row in rows] != list(SEEDS):
            raise ValueError(f"M1 seed coverage mismatch for {problem}/{regime}")
        if any(row.get("study_kind") != study_kind for row in rows):
            raise ValueError(f"M1 study-kind mismatch for {problem}/{regime}")
        for metric, field in METRICS:
            mg_field = "mg_state_mean" if field == "rel_err" else "mg_qoi_mean"
            pod_field = (
                "pod_lspg_state_mean" if field == "rel_err" else "pod_lspg_qoi_mean"
            )
            differences = [row[mg_field] - row[pod_field] for row in rows]
            negative = sum(value < 0.0 for value in differences)
            positive = sum(value > 0.0 for value in differences)
            ties = len(differences) - negative - positive
            raw_p = _exact_two_sided_sign_p(negative, positive)
            contrasts.append(
                {
                    "display": display,
                    "problem": problem,
                    "regime": regime,
                    "study_kind": study_kind,
                    "metric": metric,
                    "mg_lower": negative,
                    "pod_lspg_lower": positive,
                    "ties": ties,
                    "effective_n": negative + positive,
                    "raw_p": raw_p,
                    "direction": (
                        "MG"
                        if negative > positive
                        else "POD-LSPG"
                        if positive > negative
                        else "even"
                    ),
                }
            )

    nondegenerate = [row for row in contrasts if row["raw_p"] is not None]
    family_p = [1.0 if row["raw_p"] is None else row["raw_p"] for row in contrasts]
    adjusted_20 = _holm_adjust(family_p)
    for row, p_for_family, value in zip(contrasts, family_p, adjusted_20):
        row["family_raw_p"] = p_for_family
        row["holm_p_20"] = value
        row["holm_below_0p05"] = value <= 0.05

    below_threshold = [row for row in contrasts if row["holm_below_0p05"]]
    nld_ood_qoi = next(
        row
        for row in contrasts
        if row["problem"] == "nonlinear_diffusion"
        and row["regime"] == "ood"
        and row["metric"] == "qoi"
    )

    samples = compact["m2_2d_ood_samples"]
    if len(samples) != len(SEEDS) * SAMPLES_PER_SEED:
        raise ValueError("M2/M3 compact input does not contain exactly 120 samples")
    by_seed = _group_by(samples, "seed")
    selected_by_seed: dict[str, int] = {}
    conditional_seed_means: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for seed in SEEDS:
        rows = sorted(by_seed[(seed,)], key=lambda row: row["sample"])
        _require_rows(rows, path=DEFAULT_INPUT, label=f"seed-{seed} compact")
        chosen = [row for row in rows if row["native_inv_stationarity"] <= 1e-4]
        selected.extend(chosen)
        selected_by_seed[str(seed)] = len(chosen)
        if chosen:
            conditional_seed_means.append(
                {
                    "seed": seed,
                    "selected_count": len(chosen),
                    "precond_mean": fmean(row["precond_rel_err"] for row in chosen),
                    "mg_mean": fmean(row["mg_rel_err"] for row in chosen),
                }
            )

    threshold_rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        native_by_seed = {
            str(seed): sum(
                row["native_inv_stationarity"] <= threshold for row in by_seed[(seed,)]
            )
            for seed in SEEDS
        }
        unified_by_rtol: dict[str, dict[str, Any]] = {}
        for rtol in RTOLS:
            counts = {
                str(seed): sum(
                    row["unified_inv_stationarity"][rtol] <= threshold
                    for row in by_seed[(seed,)]
                )
                for seed in SEEDS
            }
            unified_by_rtol[rtol] = {"by_seed": counts, "total": sum(counts.values())}
        threshold_rows.append(
            {
                "threshold": threshold,
                "native_by_seed": native_by_seed,
                "native_total": sum(native_by_seed.values()),
                "unified_by_rtol": unified_by_rtol,
            }
        )

    native_mask = [row["native_inv_stationarity"] <= 1e-4 for row in samples]
    unified_masks = {
        rtol: [row["unified_inv_stationarity"][rtol] <= 1e-4 for row in samples]
        for rtol in RTOLS
    }
    converged_flag_mask = [bool(row["native_converged_inv"]) for row in samples]

    controlled_seed_means = compact["controlled_contrast_seed_means"]
    controlled_grouped = _group_by(controlled_seed_means, "problem", "regime")
    controlled_cells: list[dict[str, Any]] = []
    controlled_contrasts: list[dict[str, Any]] = []
    for problem, display, _preset, regime, study_kind in ALL_CELLS:
        rows = sorted(
            controlled_grouped[(problem, regime)], key=lambda row: row["seed"]
        )
        if [row["seed"] for row in rows] != list(SEEDS):
            raise ValueError(
                f"controlled-contrast seed coverage mismatch for {problem}/{regime}"
            )
        if any(row["study_kind"] != study_kind for row in rows):
            raise ValueError(
                f"controlled-contrast study-kind mismatch for {problem}/{regime}"
            )
        for row in rows:
            if set(row["arm_state_error_means"]) != set(ARM_LABELS):
                raise ValueError(
                    f"controlled-contrast arm coverage mismatch for "
                    f"{problem}/{regime}/seed{row['seed']}"
                )

        arm_means = {
            arm: fmean(row["arm_state_error_means"][arm] for row in rows)
            for arm in ARM_LABELS
        }
        cell_contrasts: dict[str, dict[str, Any]] = {}
        for key, left, right, interpretation in CONTROLLED_CONTRASTS:
            left_mean = arm_means[left]
            right_mean = arm_means[right]
            ratio = left_mean / right_mean
            cell_contrasts[key] = {
                "left_arm": left,
                "right_arm": right,
                "interpretation": interpretation,
                "left_mean": left_mean,
                "right_mean": right_mean,
                "left_minus_right": left_mean - right_mean,
                "left_over_right": ratio,
                "log10_left_over_right": math.log10(ratio),
            }
            controlled_contrasts.append(
                {
                    "contrast_id": key,
                    "contrast_label": f"{ARM_LABELS[left]} / {ARM_LABELS[right]}",
                    "design_role": interpretation,
                    "problem": problem,
                    "regime": regime,
                    "numerator": left,
                    "denominator": right,
                    "numerator_mean_state_error": left_mean,
                    "denominator_mean_state_error": right_mean,
                    "log10_error_ratio": math.log10(ratio),
                }
            )
        controlled_cells.append(
            {
                "display": display,
                "problem": problem,
                "regime": regime,
                "study_kind": study_kind,
                "seed_count": len(rows),
                "samples_per_seed": SAMPLES_PER_SEED,
                "arm_state_error_mean": arm_means,
                "contrasts": cell_contrasts,
            }
        )

    factorial_rows: list[dict[str, Any]] = []
    factorial_figure_rows: list[dict[str, Any]] = []
    for problem, display, _preset, regime, study_kind in ALL_CELLS:
        rows = sorted(
            controlled_grouped[(problem, regime)], key=lambda row: row["seed"]
        )
        cell = next(
            item
            for item in controlled_cells
            if item["problem"] == problem and item["regime"] == regime
        )
        arm_means = cell["arm_state_error_mean"]
        cell_effects = {
            "trial_galerkin": math.log10(arm_means["pod_g"] / arm_means["mg"]),
            "trial_minimum_residual": math.log10(
                arm_means["pod_l"] / arm_means["newton_m"]
            ),
            "equation_manifold": math.log10(arm_means["mg"] / arm_means["newton_m"]),
            "equation_affine": math.log10(arm_means["pod_g"] / arm_means["pod_l"]),
        }
        cell_effects["interaction"] = (
            cell_effects["equation_manifold"] - cell_effects["equation_affine"]
        )
        seed_effects: list[dict[str, Any]] = []
        for row in rows:
            arm = row["arm_state_error_means"]
            effects = {
                "trial_galerkin": math.log10(arm["pod_g"] / arm["mg"]),
                "trial_minimum_residual": math.log10(arm["pod_l"] / arm["newton_m"]),
                "equation_manifold": math.log10(arm["mg"] / arm["newton_m"]),
                "equation_affine": math.log10(arm["pod_g"] / arm["pod_l"]),
            }
            effects["interaction"] = (
                effects["equation_manifold"] - effects["equation_affine"]
            )
            seed_effects.append({"seed": row["seed"], **effects})
        interaction_values = [row["interaction"] for row in seed_effects]
        negative = sum(value < 0.0 for value in interaction_values)
        positive = sum(value > 0.0 for value in interaction_values)
        ties = len(interaction_values) - negative - positive
        factorial_rows.append(
            {
                "display": display,
                "problem": problem,
                "regime": regime,
                "study_kind": study_kind,
                "cell_mean_log10_effects": cell_effects,
                "seed_effects": seed_effects,
                "interaction_seed_summary": {
                    "mean": fmean(interaction_values),
                    "median": _linear_percentile(interaction_values, 0.5),
                    "p2p5": _linear_percentile(interaction_values, 0.025),
                    "p97p5": _linear_percentile(interaction_values, 0.975),
                    "negative": negative,
                    "positive": positive,
                    "ties": ties,
                    "raw_sign_p": _exact_two_sided_sign_p(negative, positive),
                },
            }
        )
        factorial_figure_rows.append(
            {
                "contrast_id": "factorial_interaction",
                "contrast_label": r"Interaction $\Delta$",
                "design_role": "equation-by-trial-set interaction in the Newton 2x2 core",
                "problem": problem,
                "regime": regime,
                "log10_error_ratio": cell_effects["interaction"],
            }
        )

    interaction_adjusted = _holm_adjust(
        [row["interaction_seed_summary"]["raw_sign_p"] for row in factorial_rows]
    )
    for row, adjusted_p in zip(factorial_rows, interaction_adjusted):
        row["interaction_seed_summary"]["holm_p_10"] = adjusted_p

    one_d_trial_rows: list[dict[str, Any]] = []
    for row in factorial_rows:
        if row["study_kind"] != "1d":
            continue
        values = [seed["trial_galerkin"] for seed in row["seed_effects"]]
        negative = sum(value < 0.0 for value in values)
        positive = sum(value > 0.0 for value in values)
        ties = len(values) - negative - positive
        one_d_trial_rows.append(
            {
                "display": row["display"],
                "problem": row["problem"],
                "regime": row["regime"],
                "pod_g_lower": negative,
                "mg_lower": positive,
                "ties": ties,
                "raw_sign_p": _exact_two_sided_sign_p(negative, positive),
            }
        )
    one_d_trial_adjusted = _holm_adjust([row["raw_sign_p"] for row in one_d_trial_rows])
    for row, adjusted_p in zip(one_d_trial_rows, one_d_trial_adjusted):
        row["holm_p_8"] = adjusted_p

    metric_samples = compact["metric_comparison_samples"]
    metric_grouped = _group_by(metric_samples, "problem", "regime")
    metric_tests: list[dict[str, Any]] = []
    metric_definitions = (
        ("metric_manifold", "pre_m", "lm_m", "LM matched on learned trial set"),
        ("metric_affine", "pre_p", "pod_l", "Newton matched on affine trial set"),
    )
    for problem, display, _preset, regime, study_kind in ALL_CELLS:
        rows = sorted(
            metric_grouped[(problem, regime)],
            key=lambda row: (row["seed"], row["sample"]),
        )
        if len(rows) != len(SEEDS) * SAMPLES_PER_SEED:
            raise ValueError(f"metric sample coverage mismatch for {problem}/{regime}")
        for contrast_id, left, right, design_role in metric_definitions:
            differences = [
                row["state_error"][left] - row["state_error"][right] for row in rows
            ]
            seed_differences = []
            for seed in SEEDS:
                seed_rows = [row for row in rows if row["seed"] == seed]
                seed_differences.append(
                    fmean(row["state_error"][left] for row in seed_rows)
                    - fmean(row["state_error"][right] for row in seed_rows)
                )
            negative = sum(value < 0.0 for value in seed_differences)
            positive = sum(value > 0.0 for value in seed_differences)
            ties = len(seed_differences) - negative - positive
            metric_tests.append(
                {
                    "contrast_id": contrast_id,
                    "design_role": design_role,
                    "display": display,
                    "problem": problem,
                    "regime": regime,
                    "study_kind": study_kind,
                    "left_arm": left,
                    "right_arm": right,
                    "sample_count": len(rows),
                    "sample_left_lower": sum(value < 0.0 for value in differences),
                    "sample_right_lower": sum(value > 0.0 for value in differences),
                    "sample_ties": sum(value == 0.0 for value in differences),
                    "seed_left_lower": negative,
                    "seed_right_lower": positive,
                    "seed_ties": ties,
                    "raw_sign_p": _exact_two_sided_sign_p(negative, positive),
                    "non_left_lower_cases": [
                        {
                            "seed": row["seed"],
                            "sample": row["sample"],
                            "left": row["state_error"][left],
                            "right": row["state_error"][right],
                        }
                        for row, difference in zip(rows, differences)
                        if difference >= 0.0
                    ],
                }
            )
    metric_adjusted = _holm_adjust([row["raw_sign_p"] for row in metric_tests])
    for row, adjusted_p in zip(metric_tests, metric_adjusted):
        row["holm_p_20"] = adjusted_p

    two_d_ood_metric_rows = metric_grouped[("two_d_burgers", "ood")]
    two_d_ood_pre_m_mg = {
        "sample_count": len(two_d_ood_metric_rows),
        "pre_m_lower": sum(
            row["state_error"]["pre_m"] < row["state_error"]["mg"]
            for row in two_d_ood_metric_rows
        ),
        "pooled_sample_mean": {
            "pre_m": fmean(
                row["state_error"]["pre_m"] for row in two_d_ood_metric_rows
            ),
            "mg": fmean(row["state_error"]["mg"] for row in two_d_ood_metric_rows),
        },
        "seed_pre_m_lower": sum(
            fmean(
                row["state_error"]["pre_m"]
                for row in two_d_ood_metric_rows
                if row["seed"] == seed
            )
            < fmean(
                row["state_error"]["mg"]
                for row in two_d_ood_metric_rows
                if row["seed"] == seed
            )
            for seed in SEEDS
        ),
    }

    result = {
        "schema_version": 4,
        "record": compact["record"],
        "input_provenance": compact["provenance"],
        "m1_holm_family": {
            "configured_contrasts": len(contrasts),
            "nondegenerate_contrasts": len(nondegenerate),
            "tie_convention": "all-tie contrast enters the design-complete family with p=1",
            "contrasts": contrasts,
            "below_0p05_count": len(below_threshold),
            "below_0p05_favoring_mg": sum(
                row["direction"] == "MG" for row in below_threshold
            ),
            "below_0p05_favoring_pod_lspg": sum(
                row["direction"] == "POD-LSPG" for row in below_threshold
            ),
            "nld_ood_qoi_holm_p_20": nld_ood_qoi["holm_p_20"],
            "below_0p05_max_holm_p_20": max(
                row["holm_p_20"] for row in below_threshold
            ),
        },
        "m2_invariant_subset": {
            "scope": {
                "problem": "two_d_burgers",
                "regime": "ood",
                "selected_arm": "pre_m",
                "paired_comparison_arm": "mg",
                "diagnostic": "reparametrization-invariant projected stationarity",
            },
            "threshold": 1e-4,
            "selected_count": len(selected),
            "selected_count_by_seed": selected_by_seed,
            "paired_precond_lower_count": sum(
                row["precond_rel_err"] < row["mg_rel_err"] for row in selected
            ),
            "pooled_sample_mean": {
                "precond": fmean(row["precond_rel_err"] for row in selected),
                "mg": fmean(row["mg_rel_err"] for row in selected),
            },
            "equal_seed_conditional_mean": {
                "precond": fmean(row["precond_mean"] for row in conditional_seed_means),
                "mg": fmean(row["mg_mean"] for row in conditional_seed_means),
                "seed_count": len(conditional_seed_means),
            },
            "conditional_seed_means": conditional_seed_means,
            "estimand_note": (
                "The manuscript's 0.316/0.689 values are pooled means over the 48 selected "
                "samples, not equally weighted seed means."
            ),
        },
        "m3_threshold_ladder": {
            "scope": {
                "problem": "two_d_burgers",
                "regime": "ood",
                "arm": "pre_m",
                "diagnostic": "reparametrization-invariant projected stationarity",
            },
            "rows": threshold_rows,
            "native_1e_4_mask_matches_stored_flag": native_mask == converged_flag_mask,
            "native_1e_4_mask_matches_unified_all_rtols": all(
                native_mask == unified_masks[rtol] for rtol in RTOLS
            ),
            "unified_counts_identical_across_rtols": all(
                len({row["unified_by_rtol"][rtol]["total"] for rtol in RTOLS}) == 1
                for row in threshold_rows
            ),
        },
        "m4_metric_full_cohort": {
            "tests": metric_tests,
            "family_size": len(metric_tests),
            "manifold_sample_left_lower": sum(
                row["sample_left_lower"]
                for row in metric_tests
                if row["contrast_id"] == "metric_manifold"
            ),
            "affine_sample_left_lower": sum(
                row["sample_left_lower"]
                for row in metric_tests
                if row["contrast_id"] == "metric_affine"
            ),
            "sample_count_per_contrast": len(ALL_CELLS) * len(SEEDS) * SAMPLES_PER_SEED,
            "manifold_seed_left_lower": sum(
                row["seed_left_lower"]
                for row in metric_tests
                if row["contrast_id"] == "metric_manifold"
            ),
            "affine_seed_left_lower": sum(
                row["seed_left_lower"]
                for row in metric_tests
                if row["contrast_id"] == "metric_affine"
            ),
            "seed_count_per_contrast": len(ALL_CELLS) * len(SEEDS),
            "two_d_ood_pre_m_vs_mg": two_d_ood_pre_m_mg,
        },
        "factorial_core": {
            "estimand": (
                "Per seed, Delta = log10[(MG/Newton-M)/(POD-G/POD-L)]; positive "
                "means the Galerkin/minimum-residual error ratio is less favorable on "
                "the learned than on the affine trial set."
            ),
            "cells": factorial_rows,
            "figure_interactions": factorial_figure_rows,
            "one_d_galerkin_trial_family": one_d_trial_rows,
        },
        "controlled_contrasts": controlled_contrasts,
        "controlled_contrast_figure": {
            "estimand": (
                "For each cell and arm, the equal-weight mean of 10 seed-level state-error "
                "means; each seed-level mean averages its 12 held-out samples."
            ),
            "arm_order": list(ARM_LABELS),
            "arm_labels": ARM_LABELS,
            "contrast_order": [
                key for key, _left, _right, _note in CONTROLLED_CONTRASTS
            ],
            "cells": controlled_cells,
        },
    }
    _assert_canonical(result)
    return result


def _assert_close(
    actual: float, expected: float, label: str, *, atol: float = 5e-15
) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol):
        raise AssertionError(f"{label}: expected {expected!r}, found {actual!r}")


def _assert_canonical(result: dict[str, Any]) -> None:
    m1 = result["m1_holm_family"]
    assert m1["configured_contrasts"] == 20
    assert m1["nondegenerate_contrasts"] == 19
    assert m1["below_0p05_count"] == 9
    assert m1["below_0p05_favoring_mg"] == 5
    assert m1["below_0p05_favoring_pod_lspg"] == 4
    _assert_close(m1["nld_ood_qoi_holm_p_20"], 0.65625, "NLD-OOD QoI Holm p")
    _assert_close(
        m1["below_0p05_max_holm_p_20"],
        0.0390625,
        "nine-below-threshold Holm p",
    )

    m2 = result["m2_invariant_subset"]
    assert m2["selected_count"] == 48
    assert list(m2["selected_count_by_seed"].values()) == [4, 6, 5, 7, 6, 6, 2, 2, 3, 7]
    assert m2["paired_precond_lower_count"] == 48
    _assert_close(
        m2["pooled_sample_mean"]["precond"], 0.315644059257837, "pooled precond"
    )
    _assert_close(m2["pooled_sample_mean"]["mg"], 0.688868477180579, "pooled MG")
    _assert_close(
        m2["equal_seed_conditional_mean"]["precond"],
        0.313150365538567,
        "equal-seed precond",
    )
    _assert_close(
        m2["equal_seed_conditional_mean"]["mg"], 0.683229759879453, "equal-seed MG"
    )

    m3 = result["m3_threshold_ladder"]
    assert [row["native_total"] for row in m3["rows"]] == [66, 48, 32]
    assert m3["native_1e_4_mask_matches_stored_flag"]
    assert m3["native_1e_4_mask_matches_unified_all_rtols"]
    assert m3["unified_counts_identical_across_rtols"]
    for row in m3["rows"]:
        assert all(
            details["total"] == row["native_total"]
            for details in row["unified_by_rtol"].values()
        )

    m4 = result["m4_metric_full_cohort"]
    assert m4["family_size"] == 20
    assert m4["sample_count_per_contrast"] == 1200
    assert m4["manifold_sample_left_lower"] == 1199
    assert m4["affine_sample_left_lower"] == 1187
    assert m4["seed_count_per_contrast"] == 100
    assert m4["manifold_seed_left_lower"] == 100
    assert m4["affine_seed_left_lower"] == 100
    assert all(row["seed_left_lower"] == 10 for row in m4["tests"])
    assert all(row["seed_right_lower"] == 0 for row in m4["tests"])
    assert all(row["seed_ties"] == 0 for row in m4["tests"])
    assert all(math.isclose(row["holm_p_20"], 0.0390625) for row in m4["tests"])
    two_d_metric = m4["two_d_ood_pre_m_vs_mg"]
    assert two_d_metric["sample_count"] == 120
    assert two_d_metric["pre_m_lower"] == 120
    assert two_d_metric["seed_pre_m_lower"] == 10
    _assert_close(
        two_d_metric["pooled_sample_mean"]["pre_m"],
        0.3464904714951444,
        "2D-OOD full-cohort pre-M mean",
    )
    _assert_close(
        two_d_metric["pooled_sample_mean"]["mg"],
        0.7839653755103383,
        "2D-OOD full-cohort MG mean",
    )

    factorial = result["factorial_core"]
    assert len(factorial["cells"]) == 10
    assert len(factorial["figure_interactions"]) == 10
    assert (
        sum(
            row["interaction_seed_summary"]["holm_p_10"] <= 0.05
            for row in factorial["cells"]
        )
        == 4
    )
    assert {
        (row["problem"], row["regime"])
        for row in factorial["cells"]
        if row["interaction_seed_summary"]["holm_p_10"] <= 0.05
    } == {
        ("front_layer", "ood"),
        ("hydrologic_conductivity", "in_domain"),
        ("nonlinear_diffusion", "ood"),
        ("two_d_burgers", "ood"),
    }
    assert len(factorial["one_d_galerkin_trial_family"]) == 8
    assert (
        sum(row["holm_p_8"] <= 0.05 for row in factorial["one_d_galerkin_trial_family"])
        == 6
    )

    figure = result["controlled_contrast_figure"]
    flat_contrasts = result["controlled_contrasts"]
    assert figure["arm_order"] == list(ARM_LABELS)
    assert figure["contrast_order"] == [
        key for key, _left, _right, _note in CONTROLLED_CONTRASTS
    ]
    assert len(figure["cells"]) == len(ALL_CELLS) == 10
    assert len(flat_contrasts) == len(ALL_CELLS) * len(CONTROLLED_CONTRASTS) == 70
    assert all(
        set(row)
        == {
            "contrast_id",
            "contrast_label",
            "design_role",
            "problem",
            "regime",
            "numerator",
            "denominator",
            "numerator_mean_state_error",
            "denominator_mean_state_error",
            "log10_error_ratio",
        }
        for row in flat_contrasts
    )
    assert [(cell["problem"], cell["regime"]) for cell in figure["cells"]] == [
        (problem, regime) for problem, _display, _preset, regime, _kind in ALL_CELLS
    ]
    for cell in figure["cells"]:
        assert cell["seed_count"] == len(SEEDS)
        assert cell["samples_per_seed"] == SAMPLES_PER_SEED
        assert set(cell["arm_state_error_mean"]) == set(ARM_LABELS)
        assert set(cell["contrasts"]) == set(figure["contrast_order"])
        assert all(
            math.isfinite(value) and value > 0.0
            for value in cell["arm_state_error_mean"].values()
        )
        for contrast in cell["contrasts"].values():
            left = cell["arm_state_error_mean"][contrast["left_arm"]]
            right = cell["arm_state_error_mean"][contrast["right_arm"]]
            _assert_close(contrast["left_mean"], left, "controlled-contrast left mean")
            _assert_close(
                contrast["right_mean"], right, "controlled-contrast right mean"
            )
            _assert_close(
                contrast["left_minus_right"],
                left - right,
                "controlled-contrast difference",
            )
            _assert_close(
                contrast["left_over_right"],
                left / right,
                "controlled-contrast ratio",
            )
            _assert_close(
                contrast["log10_left_over_right"],
                math.log10(left / right),
                "controlled-contrast log ratio",
            )


def _fmt_p(value: float | None) -> str:
    return "---" if value is None else f"{value:.6f}"


def render_markdown(result: dict[str, Any]) -> str:
    m1 = result["m1_holm_family"]
    m2 = result["m2_invariant_subset"]
    m3 = result["m3_threshold_ladder"]
    m4 = result["m4_metric_full_cohort"]
    factorial = result["factorial_core"]
    figure = result["controlled_contrast_figure"]
    lines = [
        "# Retrospective Acceptance-Revision Statistics and Design Audits",
        "",
        "> **Scope.** This artifact is derived only from the frozen seeds 0--9 retrospective",
        "> shards. It does not read, summarize, or reveal prospective-validation output.",
        "> Holm-adjusted values calibrate seed consistency in this exploratory record; they",
        "> are not interpreted as confirmatory tests.",
        "",
        "## M1: complete MG-versus-POD-LSPG sign family",
        "",
        "The design-complete steady family contains all 20 problem/regime/metric contrasts "
        "(ten cells by state and QoI error). The fully tied front-layer OOD QoI enters "
        "with `p=1`, so the family is not selected after observing which endpoint varies. "
        "Nine unanimous contrasts have Holm-adjusted `p=0.0390625`: five favor MG and "
        "four favor POD-LSPG.",
        "",
        "| Problem | Regime | Metric | MG lower | POD-L lower | Ties | Exact p | Family p | Holm p (20) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in m1["contrasts"]:
        lines.append(
            "| {display} | {regime} | {metric} | {mg_lower}/10 | {pod_lspg_lower}/10 | "
            "{ties} | {raw} | {family_raw} | {holm20} |".format(
                **row,
                raw=_fmt_p(row["raw_p"]),
                family_raw=_fmt_p(row["family_raw_p"]),
                holm20=_fmt_p(row["holm_p_20"]),
            )
        )
    lines.extend(
        [
            "",
            f"Largest adjusted p among those nine: `{m1['below_0p05_max_holm_p_20']:.9f}`.",
            "The nonlinear-diffusion OOD QoI contrast is `8/10`, exact "
            f"`p=0.109375`, and adjusts to `{m1['nld_ood_qoi_holm_p_20']:.6f}`.",
            "",
            "## M2: 2D-OOD projected-stationarity subset",
            "",
            f"At `s <= {m2['threshold']:.0e}`, the selected count is "
            f"`{m2['selected_count']}/120`; preconditioned M-LSPG has lower paired state "
            f"error on `{m2['paired_precond_lower_count']}/{m2['selected_count']}` samples.",
            "",
            "Selected counts by seed: `"
            + ", ".join(str(value) for value in m2["selected_count_by_seed"].values())
            + "`.",
            "",
            "| Estimand | Preconditioned | MG |",
            "|---|---:|---:|",
            "| Pooled over 48 selected samples | "
            f"{m2['pooled_sample_mean']['precond']:.15f} | "
            f"{m2['pooled_sample_mean']['mg']:.15f} |",
            "| Equal-weight mean of 10 conditional seed means | "
            f"{m2['equal_seed_conditional_mean']['precond']:.15f} | "
            f"{m2['equal_seed_conditional_mean']['mg']:.15f} |",
            "",
            "Thus `0.316 versus 0.689` denotes the **pooled sample means**, not the "
            "equal-seed conditional estimand.",
            "",
            "## M3: invariant-stationarity threshold ladder",
            "",
            "| Threshold | Native total | Native counts by seed | Unified totals at rtol 1e-10/1e-12/1e-14 |",
            "|---:|---:|---|---|",
        ]
    )
    for row in m3["rows"]:
        native_counts = ",".join(str(value) for value in row["native_by_seed"].values())
        unified_totals = "/".join(
            str(row["unified_by_rtol"][rtol]["total"]) for rtol in RTOLS
        )
        lines.append(
            f"| `{row['threshold']:.0e}` | {row['native_total']} | `{native_counts}` | "
            f"`{unified_totals}` |"
        )
    lines.extend(
        [
            "",
            "Consistency checks:",
            "",
            f"- native `1e-4` mask matches stored `converged_inv`: "
            f"`{m3['native_1e_4_mask_matches_stored_flag']}`;",
            f"- native `1e-4` mask matches the unified-rank mask at every rtol: "
            f"`{m3['native_1e_4_mask_matches_unified_all_rtols']}`;",
            f"- unified counts are identical across all three rank cutoffs: "
            f"`{m3['unified_counts_identical_across_rtols']}`.",
            "",
            "## M4: full-cohort algorithm-matched metric contrasts",
            "",
            f"Across all ten cells, pre-M has lower state error than raw LM-M on "
            f"`{m4['manifold_sample_left_lower']}/{m4['sample_count_per_contrast']}` "
            f"individual pairs and `{m4['manifold_seed_left_lower']}/"
            f"{m4['seed_count_per_contrast']}` cell-seed means. Pre-P is lower than "
            f"raw POD-LSPG on `{m4['affine_sample_left_lower']}/"
            f"{m4['sample_count_per_contrast']}` pairs and "
            f"`{m4['affine_seed_left_lower']}/{m4['seed_count_per_contrast']}` "
            "cell-seed means. All 20 cellwise seed tests are 10/10 and have "
            "two-sided exact `p=0.001953125`, Holm-20 `p=0.0390625`.",
            "",
            "| Contrast | Problem | Regime | Sample left-lower | Seed left-lower | Holm p (20) |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in m4["tests"]:
        lines.append(
            f"| {row['contrast_id']} | {row['display']} | {row['regime']} | "
            f"{row['sample_left_lower']}/{row['sample_count']} | "
            f"{row['seed_left_lower']}/10 | {row['holm_p_20']:.6f} |"
        )
    two_d_full = m4["two_d_ood_pre_m_vs_mg"]
    lines.extend(
        [
            "",
            "The complete 2D-OOD cohort gives pre-M below MG on "
            f"`{two_d_full['pre_m_lower']}/{two_d_full['sample_count']}` samples and "
            f"`{two_d_full['seed_pre_m_lower']}/10` seed means; pooled means are "
            f"`{two_d_full['pooled_sample_mean']['pre_m']:.6f}` and "
            f"`{two_d_full['pooled_sample_mean']['mg']:.6f}`. The 48/48 "
            "projected-stationarity subset above is therefore a sensitivity analysis, "
            "not the primary accuracy estimand.",
            "",
            "## Paired Newton 2x2 factorial audit",
            "",
            factorial["estimand"],
            "",
            "| Problem | Regime | POD-G/MG | POD-L/Newton-M | MG/Newton-M | POD-G/POD-L | Delta cell | Delta seed mean [2.5%,97.5%] | Neg/Pos/Tie | Holm p (10) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in factorial["cells"]:
        effect = row["cell_mean_log10_effects"]
        summary = row["interaction_seed_summary"]
        lines.append(
            f"| {row['display']} | {row['regime']} | "
            f"{effect['trial_galerkin']:.4f} | {effect['trial_minimum_residual']:.4f} | "
            f"{effect['equation_manifold']:.4f} | {effect['equation_affine']:.4f} | "
            f"{effect['interaction']:.4f} | {summary['mean']:.4f} "
            f"[{summary['p2p5']:.4f},{summary['p97p5']:.4f}] | "
            f"{summary['negative']}/{summary['positive']}/{summary['ties']} | "
            f"{summary['holm_p_10']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The interaction is heterogeneous: four cells cross the descriptive Holm "
            "threshold and the directions occur on both sides. The table therefore does "
            "not support a universal representation-by-equation interaction.",
            "",
            "## Controlled-contrast figure values",
            "",
            figure["estimand"],
            "",
            "| Problem | Regime | MG | Newton-M | LM-M | POD-G | POD-L | pre-M | pre-P |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in figure["cells"]:
        arm = cell["arm_state_error_mean"]
        lines.append(
            f"| {cell['display']} | {cell['regime']} | {arm['mg']:.9g} | "
            f"{arm['newton_m']:.9g} | {arm['lm_m']:.9g} | {arm['pod_g']:.9g} | "
            f"{arm['pod_l']:.9g} | {arm['pre_m']:.9g} | {arm['pre_p']:.9g} |"
        )
    lines.extend(
        [
            "",
            "Each contrast is encoded as `left/right`; values below one favor the left "
            "arm in state error.",
            "",
            "| Problem | Regime | POD-G/MG | POD-L/Newton-M | MG/Newton-M | POD-G/POD-L | pre-M/LM-M | pre-P/POD-L | LM-M/Newton-M |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in figure["cells"]:
        contrasts_by_key = cell["contrasts"]
        ratios = [
            contrasts_by_key[key]["left_over_right"] for key in figure["contrast_order"]
        ]
        lines.append(
            f"| {cell['display']} | {cell['regime']} | "
            + " | ".join(f"{value:.6g}" for value in ratios)
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--refresh-from-shards",
        type=Path,
        metavar="CACHE_DIR",
        help=(
            "refresh the compact input from the frozen retrospective shard directories "
            "under CACHE_DIR before deriving the outputs"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.refresh_from_shards is not None:
        compact = build_compact_input(args.refresh_from_shards)
        _write_json(args.input, compact)
        input_action = "wrote"
    else:
        compact = _read_json(args.input)
        input_action = "used"
    result = derive_statistics(compact)
    _write_json(args.output_json, result)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(render_markdown(result), encoding="utf-8")
    print(f"{input_action} {args.input}")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_markdown}")
    print(
        "canonical: Holm-20 p<0.05 count=9 (5 MG/4 POD-LSPG), "
        "NLD-OOD QoI adjusted p=0.65625, selected=48, ladder=66/48/32, "
        "metric pairs=1199/1200 and 1187/1200"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
