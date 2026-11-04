#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


PROTOCOL_ID = "mgp-prospective-weight-validation-v1"
EXPECTED_RATIOS = {
    "P1": 0.4398271640591194,
    "P2": 0.2778309236026475,
    "P3": 0.320819815921235,
    "P4": 0.16047502624164534,
    "P5": 0.2993208161903696,
}
PROBLEM_LABELS = {
    "bratu_source": "Bratu source",
    "front_layer": "Front layer",
    "hydrologic_conductivity": "Hydraulic $K$",
    "nonlinear_diffusion": "Nonlin.\\ diffusion",
}
REGIME_LABELS = {"in_domain": "in-dom.", "ood": "OOD"}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def p_tex(value: float) -> str:
    if value < 0.001:
        exponent = int(math.floor(math.log10(value)))
        mantissa = value / (10.0**exponent)
        return rf"\({mantissa:.3g}\times10^{{{exponent}}}\)"
    return f"{value:.3g}"


def ratio_interval(row: dict) -> tuple[float, float, float]:
    lower, upper = row["bootstrap_mean_log_ratio_ci95"]
    return (
        row["geometric_ratio_from_mean"],
        math.exp(lower),
        math.exp(upper),
    )


def validate(summary: dict) -> None:
    if summary.get("protocol_id") != PROTOCOL_ID or not summary.get("confirmation_complete"):
        raise ValueError("not the complete frozen prospective protocol summary")
    primary = summary.get("primary_holm5", [])
    if [row.get("id") for row in primary] != list(EXPECTED_RATIOS):
        raise ValueError("primary family IDs/order changed")
    for row in primary:
        expected = EXPECTED_RATIOS[row["id"]]
        if not math.isclose(row["geometric_ratio_from_mean"], expected, rel_tol=0, abs_tol=1e-15):
            raise ValueError(f"primary ratio changed: {row['id']}")
        if (
            row["favorable_negative"] != 20
            or row["n_nonzero"] != 20
            or row["holm_adjusted_p"] != 9.5367431640625e-06
            or not row["prospectively_supported"]
        ):
            raise ValueError(f"primary gate changed: {row['id']}")

    cellwise = summary.get("secondary_cellwise_holm40", [])
    if len(cellwise) != 40 or sum(row["holm_adjusted_p"] < 0.05 for row in cellwise) != 38:
        raise ValueError("cellwise Holm-40 family changed")
    exceptions = {row["id"] for row in cellwise if row["holm_adjusted_p"] >= 0.05}
    if exceptions != {
        "P1:hydrologic_conductivity:in_domain",
        "P1:hydrologic_conductivity:ood",
    }:
        raise ValueError("unexpected cellwise exceptions")
    algorithm = summary.get("secondary_algorithm_holm8", [])
    historical = summary.get("secondary_historical_holm8", [])
    if len(algorithm) != 8 or any(row["holm_adjusted_p"] < 0.05 for row in algorithm):
        raise ValueError("algorithm Holm-8 family changed")
    if len(historical) != 8 or sum(row["holm_adjusted_p"] < 0.05 for row in historical) != 5:
        raise ValueError("historical Holm-8 family changed")


def validate_integrity(
    *,
    summary: dict,
    summary_path: Path,
    receipt_path: Path,
    post_look: dict,
    post_look_script_path: Path,
) -> None:
    receipt_sha256 = sha256_path(receipt_path)
    if summary.get("first_look_receipt_sha256") != receipt_sha256:
        raise ValueError("first-look receipt bytes do not match the summary")
    integrity = post_look.get("integrity", {})
    if integrity.get("first_look_receipt_sha256") != receipt_sha256:
        raise ValueError("post-look record is not linked to the receipt bytes")
    if integrity.get("first_look_summary_sha256") != sha256_path(summary_path):
        raise ValueError("post-look record is not linked to the summary bytes")
    if post_look.get("analysis_script_sha256") != sha256_path(post_look_script_path):
        raise ValueError("post-look record is not linked to the analysis-script bytes")


def validate_post_look(post_look: dict, summary: dict) -> None:
    if (
        post_look.get("protocol_id") != PROTOCOL_ID
        or post_look.get("analysis_class") != "post_first_look_descriptive_only"
        or post_look.get("inferential_scope", {}).get("new_inferential_tests") != 0
        or post_look.get("inferential_scope", {}).get("new_p_values") != 0
        or post_look.get("inferential_scope", {}).get("changes_first_look_decision") is not False
    ):
        raise ValueError("post-look record is not the frozen descriptive-only analysis")
    if post_look.get("integrity", {}).get("first_look_receipt_sha256") != summary.get(
        "first_look_receipt_sha256"
    ):
        raise ValueError("post-look record is not linked to the first-look receipt")
    qoi = post_look.get("qoi_effects", {}).get("within_cell_contrasts", [])
    residual = post_look.get("ambient_residual_effects", {}).get(
        "within_cell_contrasts", []
    )
    reliability = post_look.get("reliability_by_cell_arm", [])
    if len(qoi) != 40 or len(residual) != 40 or len(reliability) != 56:
        raise ValueError("post-look descriptive dimensions changed")
    if sum(row["sample_count"] for row in reliability) != 13440:
        raise ValueError("post-look reliability sample count changed")
    if sum(row["catastrophic_failure_count"] for row in reliability) != 0:
        raise ValueError("post-look catastrophic-failure count changed")


def render_primary(summary: dict) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Prospectively predeclared frozen-pipeline replication on twenty untouched seed blocks and eight equally weighted one-dimensional/hydraulic cells. Ratios exponentiate the seed-level mean log ratio; brackets are descriptive paired-seed bootstrap intervals, not the confirmatory gate. Negative log ratios favor the numerator, so ratios below one favor the first arm. The exact two-sided sign tests form one Holm-5 family. Cells and held-out samples are repeated components, not independent replicates.}\label{tab:prospective}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{c l c c c}",
        r"\toprule",
        r"ID & State-error contrast & Geometric ratio [95\% interval] & Favorable seeds & Holm-5 $p$ \\",
        r"\midrule",
    ]
    for row in summary["primary_holm5"]:
        ratio, lower, upper = ratio_interval(row)
        lines.append(
            f"{row['id']} & {row['label']} & {ratio:.3f} [{lower:.3f}, {upper:.3f}] & "
            f"{row['favorable_negative']}/{row['n_nonzero']} & {p_tex(row['holm_adjusted_p'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return "\n".join(lines)


def render_cellwise(summary: dict) -> str:
    lines = [
        r"\begin{center}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{c l l c c c}",
        r"\caption{Cellwise secondary state-error analyses from the prospective replication. All forty exact sign tests form one Holm-40 family. Ratios and intervals use the same seed-level log-ratio estimand as Table~\ref{tab:prospective}; intervals are descriptive. The two hydraulic P1 rows do not pass Holm-40, and the OOD point direction reverses.}\label{si:prospective-cellwise}\\",
        r"\toprule",
        r"ID & Problem & Regime & Ratio [95\% interval] & Favorable & Holm-40 $p$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"ID & Problem & Regime & Ratio [95\% interval] & Favorable & Holm-40 $p$ \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in summary["secondary_cellwise_holm40"]:
        ratio, lower, upper = ratio_interval(row)
        lines.append(
            f"{row['id'].split(':', 1)[0]} & {PROBLEM_LABELS[row['problem']]} & "
            f"{REGIME_LABELS[row['regime']]} & {ratio:.3f} [{lower:.3f}, {upper:.3f}] & "
            f"{row['favorable_negative']}/{row['n_nonzero']} & {p_tex(row['holm_adjusted_p'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{longtable}",
        r"\normalsize",
        r"\end{center}",
        "",
    ]
    return "\n".join(lines)


def render_secondary(summary: dict) -> str:
    lines = [
        r"\begin{center}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{l l l c c c}",
        r"\caption{Predeclared secondary algorithm and historical cross-arm state-error analyses. Each block is a separate Holm-8 family. The LM/Newton family supplies no adjusted rejection and therefore does not establish equivalence. The MG/POD-LSPG block is historical, crosses trial set and equation, and is noncausal.}\label{si:prospective-secondary}\\",
        r"\toprule",
        r"Family & Problem & Regime & Ratio [95\% interval] & Negative & Holm-8 $p$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Family & Problem & Regime & Ratio [95\% interval] & Negative & Holm-8 $p$ \\",
        r"\midrule",
        r"\endhead",
    ]
    for family, records in (
        ("LM-M/Newton-M", summary["secondary_algorithm_holm8"]),
        ("MG/POD-L", summary["secondary_historical_holm8"]),
    ):
        for row in records:
            ratio, lower, upper = ratio_interval(row)
            lines.append(
                f"{family} & {PROBLEM_LABELS[row['problem']]} & {REGIME_LABELS[row['regime']]} & "
                f"{ratio:.3f} [{lower:.3f}, {upper:.3f}] & "
                f"{row['favorable_negative']}/{row['n_nonzero']} & {p_tex(row['holm_adjusted_p'])} \\\\"
            )
        if family == "LM-M/Newton-M":
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{longtable}",
        r"\normalsize",
        r"\end{center}",
        "",
    ]
    return "\n".join(lines)


def render_descriptive_effects(post_look: dict) -> str:
    lines = [
        r"\begin{center}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{l c l l c c}",
        r"\caption{Post-first-look descriptive QoI-error and raw ambient-residual effects within cells. Each endpoint is first averaged over twelve held-out samples within a seed/cell. Ratios exponentiate the mean of twenty seed log ratios with the frozen \(10^{-12}\) guard; below one favors the first arm. QoIs and raw residuals are never pooled across problem families. All front-layer OOD QoI entries are suppressed as non-informative because the boundary-fallback metric is degenerate and its displayed ratios would be epsilon-driven. These rows add no tests or multiplicity family and do not alter the first-look decision.}\label{si:prospective-descriptive-effects}\\",
        r"\toprule",
        r"Metric & ID & Problem & Regime & Ratio & A$<$B / ties \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Metric & ID & Problem & Regime & Ratio & A$<$B / ties \\",
        r"\midrule",
        r"\endhead",
    ]
    blocks = (
        ("QoI error", post_look["qoi_effects"]["within_cell_contrasts"]),
        ("Raw residual", post_look["ambient_residual_effects"]["within_cell_contrasts"]),
    )
    for block_index, (metric, records) in enumerate(blocks):
        if block_index:
            lines.append(r"\midrule")
        for row in records:
            prefix = (
                f"{metric} & {row['contrast_id']} & {PROBLEM_LABELS[row['problem']]} & "
                f"{REGIME_LABELS[row['regime']]}"
            )
            if metric == "QoI error" and row["problem"] == "front_layer" and row["regime"] == "ood":
                lines.append(
                    prefix
                    + r" & \multicolumn{2}{c}{non-informative (degenerate fallback)} \\"
                )
                continue
            lines.append(
                prefix
                + " & "
                f"{row['geometric_ratio_from_mean_log_ratio']:.4g} & "
                f"{row['seeds_arm_a_below_arm_b']}/{row['seed_count']} / {row['seeds_equal']} \\\\"
            )
    lines += [
        r"\bottomrule",
        r"\end{longtable}",
        r"\normalsize",
        r"\end{center}",
        "",
    ]
    return "\n".join(lines)


def render_reliability(post_look: dict) -> str:
    lines = [
        r"\begin{landscape}",
        r"\begin{center}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{l l l c c c c c c c}",
        r"\caption{Post-first-look descriptive reliability by cell and arm. Each row contains \(240\) solves (twenty seeds by twelve held-out samples). ``Conv.'' is the frozen formal stopping flag; ``finite-NC'' is a finite returned state without that flag. No catastrophic return occurred. Iteration/time entries are medians. The last three columns count the arm-objective-coordinate projected-stationarity diagnostic at or below each threshold; raw and preconditioned objective coordinates are not directly comparable. These rows are descriptive and were not filtered into the primary endpoint.}\label{si:prospective-reliability}\\",
        r"\toprule",
        r"Problem & Reg. & Arm & Conv. & finite-NC & Iter. & Time (s) & $10^{-3}$ & $10^{-4}$ & $10^{-5}$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Problem & Reg. & Arm & Conv. & finite-NC & Iter. & Time (s) & $10^{-3}$ & $10^{-4}$ & $10^{-5}$ \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in post_look["reliability_by_cell_arm"]:
        thresholds = row["invariant_stationarity_summary"]["threshold_counts"]
        lines.append(
            f"{PROBLEM_LABELS[row['problem']]} & {REGIME_LABELS[row['regime']]} & "
            f"{row['arm_label']} & {row['formal_converged_count']}/240 & "
            f"{row['finite_return_nonconverged_count']}/240 & "
            f"{row['iteration_summary']['median']:.4g} & "
            f"{row['time_sec_summary']['median']:.4g} & "
            f"{thresholds['1e-03']['count_at_or_below']} & "
            f"{thresholds['1e-04']['count_at_or_below']} & "
            f"{thresholds['1e-05']['count_at_or_below']} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{longtable}",
        r"\normalsize",
        r"\end{center}",
        r"\end{landscape}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    package_root = Path(__file__).resolve().parents[1]
    default_study = package_root / "eval" / "prospective_weight_validation_2026-07-17"
    default_output = package_root / "artifacts" / "prospective"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=default_study / "first_look_output" / "FIRST_LOOK_SUMMARY.json",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=default_study / "first_look_output" / "FIRST_LOOK_RECEIPT.json",
    )
    parser.add_argument(
        "--post-look",
        type=Path,
        default=default_study / "first_look_output" / "POST_LOOK_DESCRIPTIVES_V1.json",
    )
    parser.add_argument(
        "--post-look-script",
        type=Path,
        default=default_study / "post_look_descriptives_v1.py",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    post_look = json.loads(args.post_look.read_text(encoding="utf-8"))
    validate(summary)
    validate_integrity(
        summary=summary,
        summary_path=args.summary,
        receipt_path=args.receipt,
        post_look=post_look,
        post_look_script_path=args.post_look_script,
    )
    validate_post_look(post_look, summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "prospective_primary_table.tex": render_primary(summary),
        "prospective_cellwise_table.tex": render_cellwise(summary),
        "prospective_secondary_table.tex": render_secondary(summary),
        "prospective_descriptive_effects_table.tex": render_descriptive_effects(post_look),
        "prospective_reliability_table.tex": render_reliability(post_look),
    }
    for name, contents in outputs.items():
        (args.output_dir / name).write_text(contents, encoding="utf-8")
    print("PASS: validated first/post look and rendered 5 prospective LaTeX tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
