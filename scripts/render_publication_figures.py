from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_MPL_DIR = ROOT / "artifacts" / ".mplconfig"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Bbox

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

PDF_METADATA = {
    "Creator": "MGP reproducibility package",
    "Producer": "Matplotlib",
    "CreationDate": datetime(2026, 7, 18, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 7, 18, tzinfo=timezone.utc),
}


ARTIFACTS = ROOT / "artifacts"
FAMILY_COLORS = {
    "nonlinear_diffusion": "#0072B2",
    "front_layer": "#009E73",
    "bratu_source": "#D55E00",
    "hydrologic_conductivity": "#CC79A7",
    "two_d_burgers": "#E69F00",
}
FAMILY_LABELS = {
    "nonlinear_diffusion": "Nonlinear diffusion",
    "front_layer": "Front layer",
    "bratu_source": "Bratu source",
    "hydrologic_conductivity": "Hydraulic conductivity",
    "two_d_burgers": "Two-dimensional Burgers",
}
SOLVER_STYLES = {
    "mgp_galerkin": ("o", "Manifold Galerkin"),
    "mgp_lspg": ("s", "Manifold-LSPG"),
}
REGIME_LABELS = {"in_domain": "In-domain", "ood": "Out-of-domain"}
HEADLINE_ROWS = [
    ("reference_front_layer", "in_domain"),
    ("reference_front_layer", "ood"),
    ("reference_bratu_source", "ood"),
    ("reference_hydrologic_conductivity", "in_domain"),
    ("reference_hydrologic_conductivity", "ood"),
]
HEADLINE_LABELS = {
    ("reference_front_layer", "in_domain"): "Front in-domain",
    ("reference_front_layer", "ood"): "Front out-of-domain",
    ("reference_bratu_source", "ood"): "Bratu-source out-of-domain",
    ("reference_hydrologic_conductivity", "in_domain"): "Hydraulic in-domain",
    ("reference_hydrologic_conductivity", "ood"): "Hydraulic out-of-domain",
}
SHORT_HEADLINE_LABELS = {
    ("reference_front_layer", "in_domain"): "Front in",
    ("reference_front_layer", "ood"): "Front out",
    ("reference_bratu_source", "ood"): "Bratu out",
    ("reference_hydrologic_conductivity", "in_domain"): "Hydraulic in",
    ("reference_hydrologic_conductivity", "ood"): "Hydraulic out",
}
HEADLINE_ANNOTATION_OFFSETS = {
    ("reference_front_layer", "in_domain"): (-8, -15),
    ("reference_front_layer", "ood"): (-46, 2),
    ("reference_bratu_source", "ood"): (7, 8),
    ("reference_hydrologic_conductivity", "in_domain"): (9, -14),
    ("reference_hydrologic_conductivity", "ood"): (2, 11),
}
FINALIZED_ROWS = [
    ("reference_bratu_source", "in_domain"),
    ("reference_bratu_source", "ood"),
    ("reference_front_layer", "in_domain"),
    ("reference_front_layer", "ood"),
    ("reference_hydrologic_conductivity", "in_domain"),
    ("reference_hydrologic_conductivity", "ood"),
    ("reference_nonlinear_diffusion", "in_domain"),
    ("reference_nonlinear_diffusion", "ood"),
]
FINALIZED_LABELS = {
    ("reference_bratu_source", "in_domain"): "Bratu in",
    ("reference_bratu_source", "ood"): "Bratu out",
    ("reference_front_layer", "in_domain"): "Front in",
    ("reference_front_layer", "ood"): "Front out",
    ("reference_hydrologic_conductivity", "in_domain"): "Hydraulic in",
    ("reference_hydrologic_conductivity", "ood"): "Hydraulic out",
    ("reference_nonlinear_diffusion", "in_domain"): "Diffusion in",
    ("reference_nonlinear_diffusion", "ood"): "Diffusion out",
}


def render_publication_figures(
    *,
    artifacts_dir: Path = ARTIFACTS,
    output_dir: Path | None = None,
) -> list[Path]:
    output_dir = output_dir or artifacts_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    expanded_error_rows = _read_csv(
        artifacts_dir / "publication_expanded_error_residual.csv"
    )
    expanded_speed_rows = _read_csv(
        artifacts_dir / "publication_expanded_speedup_residual.csv"
    )
    hydrologic_error_rows = _read_csv(
        artifacts_dir / "publication_hydrologic_error_residual.csv"
    )
    hydrologic_speed_rows = _read_csv(
        artifacts_dir / "publication_hydrologic_speedup_residual.csv"
    )
    highlighted_rows = _read_csv(artifacts_dir / "highlighted_row_uncertainty.csv")
    expanded_summary = _read_json(artifacts_dir / "publication_summary_expanded.json")
    hydrologic_summary = _read_json(
        artifacts_dir / "publication_summary_hydrologic.json"
    )
    acceptance_path = artifacts_dir / "acceptance_statistics.json"
    acceptance_summary = _read_json(acceptance_path) if acceptance_path.exists() else {}
    prospective_root = (
        ROOT / "eval" / "prospective_weight_validation_2026-07-17"
        / "first_look_output"
    )
    prospective_summary = _read_json(
        prospective_root / "FIRST_LOOK_SUMMARY.json"
    )
    prospective_2d_root = (
        ROOT / "eval" / "prospective_2d_scale_warmstart_2026-07-19"
        / "first_look_output"
    )
    prospective_2d_summary = _read_json(
        prospective_2d_root / "FIRST_LOOK_SUMMARY.json"
    )

    headline_png = output_dir / "figure1_solver_aware_headline.png"
    headline_pdf = output_dir / "figure1_solver_aware_headline.pdf"
    prospective_png = output_dir / "figure1_prospective_validation.png"
    prospective_pdf = output_dir / "figure1_prospective_validation.pdf"
    prospective_2d_png = output_dir / "figure2_prospective_2d_warmstart.png"
    prospective_2d_pdf = output_dir / "figure2_prospective_2d_warmstart.pdf"
    expanded_error_png = output_dir / "figure2_expanded_error_residual.png"
    expanded_error_pdf = output_dir / "figure2_expanded_error_residual.pdf"
    hydrologic_png = output_dir / "figure3_hydraulic_conductivity_benchmark.png"
    hydrologic_pdf = output_dir / "figure3_hydraulic_conductivity_benchmark.pdf"
    support_speed_png = output_dir / "figureS1_combined_speedup_residual.png"
    support_speed_pdf = output_dir / "figureS1_combined_speedup_residual.pdf"
    solver_delta_matrix_png = (
        output_dir / "figureS2_selected_gap_solver_delta_matrix.png"
    )
    solver_delta_matrix_pdf = (
        output_dir / "figureS2_selected_gap_solver_delta_matrix.pdf"
    )
    robustness_png = output_dir / "figure_solver_robustness.png"
    robustness_pdf = output_dir / "figure_solver_robustness.pdf"
    fields_png = output_dir / "figure_solution_fields.png"
    fields_pdf = output_dir / "figure_solution_fields.pdf"

    combined_error_rows = expanded_error_rows + hydrologic_error_rows
    combined_speed_rows = expanded_speed_rows + hydrologic_speed_rows
    solver_table_rows = (
        expanded_summary["solver_table"] + hydrologic_summary["solver_table"]
    )

    controlled_contrasts = list(acceptance_summary.get("controlled_contrasts", []))
    controlled_contrasts.extend(
        acceptance_summary.get("factorial_core", {}).get("figure_interactions", [])
    )
    if controlled_contrasts:
        _plot_controlled_contrast_headline_figure(
            controlled_contrasts,
            png_path=headline_png,
            pdf_path=headline_pdf,
        )
    else:
        _plot_solver_aware_headline_figure(
            combined_error_rows=combined_error_rows,
            highlighted_rows=highlighted_rows,
            png_path=headline_png,
            pdf_path=headline_pdf,
        )
    _plot_prospective_validation_figure(
        prospective_summary,
        png_path=prospective_png,
        pdf_path=prospective_pdf,
    )
    _plot_prospective_2d_warmstart_figure(
        prospective_2d_summary,
        png_path=prospective_2d_png,
        pdf_path=prospective_2d_pdf,
    )
    _plot_tradeoff_scatter(
        combined_speed_rows,
        x_key="speedup_vs_full",
        y_key="residual_gap_vs_pod_lspg",
        x_label="Speedup vs Full-Order Solve",
        y_label="Full-Order Residual Gap vs POD-LSPG",
        title=None,
        png_path=support_speed_png,
        pdf_path=support_speed_pdf,
        inside_legend_corner="upper left",
        legend_box=True,
    )
    _plot_error_residual(
        expanded_error_rows,
        expanded_error_png,
        expanded_error_pdf,
        title=None,
    )
    _plot_hydraulic_conductivity_benchmark(
        hydrologic_summary["solver_table"], hydrologic_png, hydrologic_pdf
    )
    _plot_solver_robustness(solver_table_rows, robustness_png, robustness_pdf)
    _plot_solution_fields(
        artifacts_dir / "solution_fields_front_layer.csv", fields_png, fields_pdf
    )
    _plot_selected_gap_solver_delta_matrix_figure(
        highlighted_rows=highlighted_rows,
        solver_table_rows=solver_table_rows,
        png_path=solver_delta_matrix_png,
        pdf_path=solver_delta_matrix_pdf,
    )
    return [
        robustness_png,
        robustness_pdf,
        headline_png,
        headline_pdf,
        prospective_png,
        prospective_pdf,
        prospective_2d_png,
        prospective_2d_pdf,
        expanded_error_png,
        expanded_error_pdf,
        hydrologic_png,
        hydrologic_pdf,
        fields_png,
        fields_pdf,
        support_speed_png,
        support_speed_pdf,
        solver_delta_matrix_png,
        solver_delta_matrix_pdf,
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _plot_prospective_validation_figure(
    summary: dict,
    *,
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Put the untouched-seed evidence and its cell heterogeneity first."""

    primary = list(summary["primary_holm5"])
    cellwise = list(summary["secondary_cellwise_holm40"])
    contrast_order = [row["id"] for row in primary]
    contrast_labels = {
        row["id"]: f'{row["id"]}: {row["label"]}' for row in primary
    }
    cell_order = [
        ("bratu_source", "in_domain", "Bratu\nin"),
        ("bratu_source", "ood", "Bratu\nOOD"),
        ("front_layer", "in_domain", "Front\nin"),
        ("front_layer", "ood", "Front\nOOD"),
        ("hydrologic_conductivity", "in_domain", "Hydraulic\nin"),
        ("hydrologic_conductivity", "ood", "Hydraulic\nOOD"),
        ("nonlinear_diffusion", "in_domain", "Diffusion\nin"),
        ("nonlinear_diffusion", "ood", "Diffusion\nOOD"),
    ]

    figure, (suite_axis, cell_axis) = plt.subplots(
        1,
        2,
        figsize=(10.2, 5.4),
        gridspec_kw={"width_ratios": [1.05, 1.75]},
    )
    y_positions = list(range(len(primary)))
    suite_color = "#0072B2"
    for row_index, row in enumerate(primary):
        seed_ratios = [math.exp(float(value)) for value in row["seed_values"]]
        offsets = [
            ((index % 5) - 2) * 0.025 + ((index // 5) - 1.5) * 0.006
            for index in range(len(seed_ratios))
        ]
        suite_axis.scatter(
            seed_ratios,
            [row_index + offset for offset in offsets],
            s=18,
            color="#777777",
            alpha=0.55,
            linewidths=0,
            zorder=2,
        )
        mean = float(row["geometric_ratio_from_mean"])
        low, high = [
            math.exp(float(value))
            for value in row["bootstrap_mean_log_ratio_ci95"]
        ]
        suite_axis.errorbar(
            mean,
            row_index,
            xerr=[[mean - low], [high - mean]],
            fmt="D",
            color=suite_color,
            ecolor=suite_color,
            elinewidth=2.1,
            capsize=3,
            markersize=6,
            zorder=4,
        )
        suite_axis.text(
            mean,
            row_index - 0.25,
            f"{mean:.3f}",
            ha="center",
            va="top",
            color=suite_color,
            fontsize=8.2,
            fontweight="semibold",
        )
    suite_axis.axvline(1.0, color="black", linestyle="--", linewidth=0.9)
    suite_axis.set_xscale("log")
    suite_axis.set_xlim(0.08, 1.25)
    suite_axis.set_xticks([0.1, 0.2, 0.5, 1.0], ["0.1", "0.2", "0.5", "1"])
    suite_axis.set_yticks(
        y_positions,
        [contrast_labels[row["id"]].replace(" / ", " /\n") for row in primary],
    )
    suite_axis.set_xlabel("Returned-state error ratio\n(numerator / denominator)")
    suite_axis.set_title("(a) Suite-level seed contrasts", loc="left", fontsize=11.5)
    suite_axis.grid(axis="x", alpha=0.18, linewidth=0.6)
    suite_axis.invert_yaxis()
    suite_axis.tick_params(labelsize=8.7)

    lookup = {
        (row["id"].split(":", 1)[0], row["problem"], row["regime"]): row
        for row in cellwise
    }
    ratio_matrix: list[list[float]] = []
    log_matrix: list[list[float]] = []
    for contrast_id in contrast_order:
        ratios = [
            float(lookup[(contrast_id, problem, regime)]["geometric_ratio_from_mean"])
            for problem, regime, _ in cell_order
        ]
        ratio_matrix.append(ratios)
        log_matrix.append([math.log10(value) for value in ratios])

    image = cell_axis.imshow(
        log_matrix,
        cmap="RdBu_r",
        vmin=-1.25,
        vmax=1.25,
        aspect="auto",
    )
    cell_axis.set_xticks(
        range(len(cell_order)), [label for _, _, label in cell_order]
    )
    cell_axis.set_yticks(range(len(contrast_order)), contrast_order)
    cell_axis.set_title("(b) Cellwise heterogeneity", loc="left", fontsize=11.5)
    cell_axis.tick_params(axis="x", labelsize=8.3)
    cell_axis.tick_params(axis="y", labelsize=9)
    cell_axis.set_xticks(
        [value - 0.5 for value in range(1, len(cell_order))], minor=True
    )
    cell_axis.set_yticks(
        [value - 0.5 for value in range(1, len(contrast_order))], minor=True
    )
    cell_axis.grid(which="minor", color="white", linewidth=1.0)
    cell_axis.tick_params(which="minor", bottom=False, left=False)
    for row_index, contrast_id in enumerate(contrast_order):
        for column_index, (problem, regime, _) in enumerate(cell_order):
            record = lookup[(contrast_id, problem, regime)]
            ratio = ratio_matrix[row_index][column_index]
            marker = r"$\dagger$" if float(record["holm_adjusted_p"]) > 0.05 else ""
            text_color = "white" if abs(log_matrix[row_index][column_index]) > 0.65 else "black"
            cell_axis.text(
                column_index,
                row_index,
                f"{ratio:.2f}{marker}",
                ha="center",
                va="center",
                fontsize=8.1,
                color=text_color,
                fontweight="semibold",
            )
    colorbar = figure.colorbar(image, ax=cell_axis, fraction=0.036, pad=0.025)
    colorbar.set_label(r"$\log_{10}$ error ratio", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    figure.text(
        0.51,
        0.015,
        r"Ratios below 1 favor the numerator. $\dagger$ marks the two Holm-40 non-rejections.",
        ha="center",
        va="bottom",
        fontsize=8.7,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    figure.savefig(png_path, dpi=220)
    figure.savefig(pdf_path, metadata=PDF_METADATA)
    plt.close(figure)


def _plot_prospective_2d_warmstart_figure(
    summary: dict,
    *,
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Visualize the two frozen warm-start families and per-arm sensitivity."""

    primary = {row["id"]: row for row in summary["primary_family"]}
    sensitivity = {row["id"]: row for row in summary["sensitivity_family"]}
    contrast_ids = [f"D{index}" for index in range(1, 7)]
    missing = [
        contrast_id
        for contrast_id in contrast_ids
        if contrast_id not in primary or contrast_id not in sensitivity
    ]
    if missing:
        raise ValueError(f"prospective 2D summary lacks contrasts: {missing}")

    figure, (contrast_axis, arm_axis) = plt.subplots(
        1,
        2,
        figsize=(10.5, 5.6),
        gridspec_kw={"width_ratios": [1.22, 1.0]},
    )
    strategies = [
        (primary, "Nearest-training start", "#0072B2", -0.13, "o"),
        (sensitivity, "Training-mean start", "#D55E00", 0.13, "s"),
    ]
    y_positions = list(range(len(contrast_ids)))
    for family, label, color, offset, marker in strategies:
        for row_index, contrast_id in enumerate(contrast_ids):
            row = family[contrast_id]
            ratio = float(row["geometric_ratio"])
            low, high = [float(value) for value in row["bootstrap_ratio_interval"]]
            passes = bool(row["passes_expected_direction"])
            contrast_axis.errorbar(
                ratio,
                row_index + offset,
                xerr=[[ratio - low], [high - ratio]],
                fmt=marker,
                color=color,
                ecolor=color,
                markerfacecolor=color if passes else "white",
                markeredgecolor=color,
                markeredgewidth=1.4,
                elinewidth=1.8,
                capsize=3,
                markersize=6.5,
                label=label if row_index == 0 else None,
                zorder=3,
            )
    contrast_axis.axvline(1.0, color="black", linestyle="--", linewidth=0.9)
    contrast_axis.set_xscale("log")
    contrast_axis.set_xlim(0.4, 2.25)
    contrast_axis.set_xticks(
        [0.5, 0.75, 1.0, 1.5, 2.0], ["0.5", "0.75", "1", "1.5", "2"]
    )
    contrast_axis.set_yticks(
        y_positions,
        [
            f'{contrast_id}: {primary[contrast_id]["label"]}'.replace(" / ", " /\n")
            for contrast_id in contrast_ids
        ],
    )
    contrast_axis.invert_yaxis()
    contrast_axis.set_xlabel("Error ratio\n(first method / second method)")
    contrast_axis.set_title("(a) Prespecified comparisons", loc="left", fontsize=11.5)
    contrast_axis.grid(axis="x", alpha=0.2, linewidth=0.6)
    contrast_axis.tick_params(labelsize=8.8)
    contrast_axis.legend(loc="lower left", frameon=False, fontsize=8.6)

    arm_order = ["MG", "Newton-M-LSPG", "POD-G", "POD-LSPG"]
    warm_lookup = {
        (str(row["regime"]), str(row["label"])): row
        for row in summary["warm_start_effects"]
    }
    regime_styles = [
        ("in_domain", "In-domain", "#009E73", -0.13, "o"),
        ("ood", "Out-of-domain", "#CC79A7", 0.13, "s"),
    ]
    for regime, regime_label, color, offset, marker in regime_styles:
        for row_index, arm_label in enumerate(arm_order):
            row = warm_lookup[(regime, arm_label)]
            ratio = float(row["geometric_ratio"])
            low, high = [float(value) for value in row["bootstrap_ratio_interval"]]
            arm_axis.errorbar(
                ratio,
                row_index + offset,
                xerr=[[ratio - low], [high - ratio]],
                fmt=marker,
                color=color,
                ecolor=color,
                markerfacecolor=color if regime == "in_domain" else "white",
                markeredgecolor=color,
                markeredgewidth=1.4,
                elinewidth=1.8,
                capsize=3,
                markersize=6.5,
                label=regime_label if row_index == 0 else None,
                zorder=3,
            )
    arm_axis.axvspan(0.86, 3.55, ymin=0.5, ymax=1.0, color="#f2f2f2", zorder=0)
    arm_axis.axvline(1.0, color="black", linestyle="--", linewidth=0.9)
    arm_axis.set_xscale("log")
    arm_axis.set_xlim(0.86, 3.55)
    arm_axis.set_xticks([1.0, 1.25, 1.5, 2.0, 3.0], ["1", "1.25", "1.5", "2", "3"])
    arm_axis.set_yticks(range(len(arm_order)), arm_order)
    arm_axis.invert_yaxis()
    arm_axis.set_xlabel("Training-mean / nearest-start error ratio")
    arm_axis.set_title("(b) Per-arm start sensitivity", loc="left", fontsize=11.5)
    arm_axis.grid(axis="x", alpha=0.2, linewidth=0.6)
    arm_axis.tick_params(labelsize=8.8)
    arm_axis.legend(loc="lower right", frameon=False, fontsize=8.6)
    arm_axis.text(
        0.98,
        0.91,
        "Learned arms",
        transform=arm_axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        color="#555555",
    )
    arm_axis.text(
        0.98,
        0.42,
        "Affine arms",
        transform=arm_axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        color="#555555",
    )

    figure.subplots_adjust(left=0.18, right=0.985, top=0.92, bottom=0.20, wspace=0.35)
    figure.savefig(png_path, dpi=220)
    figure.savefig(pdf_path, metadata=PDF_METADATA)
    plt.close(figure)


def _plot_controlled_contrast_headline_figure(
    rows: list[dict],
    *,
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Plot the design-matched state-error contrasts that carry the paper.

    Every point is a problem/regime cell mean. The horizontal coordinate is
    ``log10(mean error of numerator / mean error of denominator)``; negative
    values favor the numerator. The plot deliberately separates the Newton
    2-by-2 core from the algorithm and metric extensions.
    """
    panel_ids = [
        (
            r"(a) Newton $2\times2$ core",
            [
                "trial_galerkin",
                "trial_minimum_residual",
                "equation_manifold",
                "equation_affine",
                "factorial_interaction",
            ],
        ),
        (
            "(b) Algorithm and metric controls",
            ["algorithm_manifold", "metric_manifold", "metric_affine"],
        ),
    ]
    by_id: dict[str, list[dict]] = {}
    for row in rows:
        by_id.setdefault(str(row["contrast_id"]), []).append(row)

    if "factorial_interaction" not in by_id and {
        "equation_manifold",
        "equation_affine",
    } <= set(by_id):
        affine = {
            (str(r["problem"]), str(r["regime"])): r for r in by_id["equation_affine"]
        }
        for r in by_id["equation_manifold"]:
            key = (str(r["problem"]), str(r["regime"]))
            if key in affine:
                interaction = dict(r)
                interaction["contrast_id"] = "factorial_interaction"
                interaction["contrast_label"] = r"Interaction $\Delta$"
                interaction["log10_error_ratio"] = float(
                    r["log10_error_ratio"]
                ) - float(affine[key]["log10_error_ratio"])
                by_id.setdefault("factorial_interaction", []).append(interaction)

    missing = [cid for _, ids in panel_ids for cid in ids if cid not in by_id]
    if missing:
        raise ValueError(f"acceptance_statistics.json lacks contrasts: {missing}")

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 5.4), sharex=True)
    problem_order = [
        "bratu_source",
        "front_layer",
        "hydrologic_conductivity",
        "nonlinear_diffusion",
        "two_d_burgers",
    ]
    offsets = {
        problem: offset
        for problem, offset in zip(problem_order, [-0.20, -0.10, 0.0, 0.10, 0.20])
    }

    for panel_index, (axis, (title, contrast_ids)) in enumerate(zip(axes, panel_ids)):
        for y, contrast_id in enumerate(contrast_ids):
            contrast_rows = by_id[contrast_id]
            for row in contrast_rows:
                problem = str(row["problem"])
                regime = str(row["regime"])
                value = float(row["log10_error_ratio"])
                color = FAMILY_COLORS[problem]
                marker = "o" if regime == "in_domain" else "s"
                facecolor = color if regime == "in_domain" else "none"
                axis.scatter(
                    value,
                    y + offsets[problem],
                    s=48,
                    marker=marker,
                    facecolors=facecolor,
                    edgecolors=color,
                    linewidths=1.4,
                    zorder=3,
                )

            values = sorted(float(row["log10_error_ratio"]) for row in contrast_rows)
            median = (
                0.5 * (values[4] + values[5])
                if len(values) == 10
                else values[len(values) // 2]
            )
            axis.scatter(
                median,
                y,
                s=72,
                marker="D",
                facecolors="black",
                edgecolors="white",
                linewidths=0.7,
                zorder=4,
            )

        labels = [
            str(by_id[cid][0]["contrast_label"]).replace(" / ", " /\n")
            for cid in contrast_ids
        ]
        axis.set_yticks(range(len(contrast_ids)), labels)
        axis.invert_yaxis()
        axis.axvline(0.0, color="black", linewidth=1.0, linestyle="--", zorder=1)
        axis.grid(axis="x", alpha=0.22, linewidth=0.6)
        axis.set_title(title, loc="left", fontsize=11.5, fontweight="semibold")
        xlabel = "$\\log_{10}$(mean state-error ratio)\nNegative values favor numerator"
        if panel_index == 0:
            xlabel += "\n(interaction row = learned minus affine equation effect)"
        axis.set_xlabel(xlabel)
        axis.tick_params(labelsize=9.5)

    problem_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=6.5,
            markerfacecolor=FAMILY_COLORS[name],
            markeredgecolor=FAMILY_COLORS[name],
            label=FAMILY_LABELS[name],
        )
        for name in problem_order
    ]
    regime_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="None", color="black", label="In-domain"
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="black",
            label="Out-of-domain",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            color="black",
            label="Across-cell median",
        ),
    ]
    legend = axes[1].legend(
        handles=problem_handles + regime_handles,
        loc="upper left",
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="0.4",
        framealpha=0.95,
        fontsize=7.6,
        handletextpad=0.4,
        columnspacing=1.0,
        labelspacing=0.4,
        borderpad=0.5,
    )
    legend.set_zorder(6)
    figure.subplots_adjust(left=0.15, right=0.985, top=0.93, bottom=0.16, wspace=0.36)
    figure.savefig(png_path, dpi=220)
    figure.savefig(pdf_path, metadata=PDF_METADATA)
    plt.close(figure)


def _plot_solver_aware_headline_figure(
    *,
    combined_error_rows: list[dict[str, str]],
    highlighted_rows: list[dict[str, str]],
    png_path: Path,
    pdf_path: Path,
) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(10.0, 5.2), gridspec_kw={"width_ratios": [1.0, 1.2]}
    )
    _plot_same_manifold_shift_panel(axes[0], combined_error_rows)
    _plot_headline_audit_panel(axes[1], highlighted_rows)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.13, wspace=0.32)
    figure.savefig(png_path, dpi=200)
    figure.savefig(pdf_path, metadata=PDF_METADATA)
    plt.close(figure)


def _plot_same_manifold_shift_panel(axis: plt.Axes, rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (row["preset_name"], row["test_regime"])
        grouped.setdefault(key, {})[row["solver_variant"]] = row

    for key, pair in grouped.items():
        mg_row = pair.get("mgp_galerkin")
        lspg_row = pair.get("mgp_lspg")
        if mg_row is None or lspg_row is None:
            continue
        if key in HEADLINE_ROWS:
            continue
        color = FAMILY_COLORS.get(mg_row["problem_name"], "#444444")
        face_color = color if mg_row["test_regime"] == "in_domain" else "none"
        x0 = float(mg_row["error_gap_vs_pod_lspg"])
        y0 = float(mg_row["residual_gap_vs_pod_lspg"])
        x1 = float(lspg_row["error_gap_vs_pod_lspg"])
        y1 = float(lspg_row["residual_gap_vs_pod_lspg"])
        axis.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "linewidth": 1.2,
                "alpha": 0.7,
            },
            zorder=2,
        )
        axis.scatter(
            x0,
            y0,
            s=58,
            marker="o",
            facecolors=face_color,
            edgecolors=color,
            linewidths=1.3,
            alpha=0.8,
            zorder=3,
        )
        axis.scatter(
            x1,
            y1,
            s=58,
            marker="s",
            facecolors=face_color,
            edgecolors=color,
            linewidths=1.3,
            alpha=0.8,
            zorder=3,
        )

    for key in HEADLINE_ROWS:
        mg_row = grouped[key]["mgp_galerkin"]
        lspg_row = grouped[key]["mgp_lspg"]
        color = FAMILY_COLORS.get(mg_row["problem_name"], "#444444")
        face_color = color if mg_row["test_regime"] == "in_domain" else "none"
        x0 = float(mg_row["error_gap_vs_pod_lspg"])
        y0 = float(mg_row["residual_gap_vs_pod_lspg"])
        x1 = float(lspg_row["error_gap_vs_pod_lspg"])
        y1 = float(lspg_row["residual_gap_vs_pod_lspg"])
        axis.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "linewidth": 2.0,
                "alpha": 0.95,
            },
            zorder=3,
        )
        axis.scatter(
            x0,
            y0,
            s=92,
            marker="o",
            facecolors=face_color,
            edgecolors=color,
            linewidths=2.0,
            zorder=4,
        )
        axis.scatter(
            x1,
            y1,
            s=92,
            marker="s",
            facecolors=face_color,
            edgecolors=color,
            linewidths=2.0,
            zorder=4,
        )

    x_values = [float(row["error_gap_vs_pod_lspg"]) for row in rows]
    y_values = [float(row["residual_gap_vs_pod_lspg"]) for row in rows]
    x_margin = 0.12 * max(0.5, max(x_values) - min(x_values))
    y_margin = 0.12 * max(0.3, max(y_values) - min(y_values))
    axis.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
    axis.set_ylim(min(0.0, min(y_values) - y_margin), max(y_values) + 2.4 * y_margin)
    axis.axvline(0.0, color="black", linewidth=0.9, linestyle="--")
    axis.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    axis.set_xlabel("Mean state-error gap vs POD-LSPG", fontsize=11.5)
    axis.set_ylabel("Mean full-order residual gap vs POD-LSPG", fontsize=11.5)
    axis.tick_params(labelsize=10.5)
    axis.text(
        0.02,
        0.98,
        "(a)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
    )
    axis.grid(alpha=0.18, linewidth=0.5)

    label_positions = {
        ("reference_front_layer", "ood"): (0.20, 0.55),
        ("reference_bratu_source", "ood"): (0.14, 0.31),
        ("reference_hydrologic_conductivity", "ood"): (-0.30, 0.33),
        ("reference_front_layer", "in_domain"): (-0.32, 0.025),
        ("reference_hydrologic_conductivity", "in_domain"): (0.22, 0.025),
    }
    for key in HEADLINE_ROWS:
        mg_row = grouped[key]["mgp_galerkin"]
        x0 = float(mg_row["error_gap_vs_pod_lspg"])
        y0 = float(mg_row["residual_gap_vs_pod_lspg"])
        tx, ty = label_positions[key]
        color = FAMILY_COLORS.get(mg_row["problem_name"], "#444444")
        axis.annotate(
            SHORT_HEADLINE_LABELS[key],
            xy=(x0, y0),
            xytext=(tx, ty),
            textcoords="data",
            fontsize=10.0,
            fontweight="semibold",
            color=color,
            ha="center",
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": "#999999",
                "linewidth": 0.8,
                "shrinkA": 2,
                "shrinkB": 3,
            },
            zorder=5,
        )

    problem_families = [
        "front_layer",
        "bratu_source",
        "hydrologic_conductivity",
        "nonlinear_diffusion",
    ]
    problem_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=7,
            markerfacecolor=FAMILY_COLORS[name],
            markeredgecolor=FAMILY_COLORS[name],
            label=FAMILY_LABELS[name],
        )
        for name in problem_families
    ]
    legend = axis.legend(
        handles=problem_handles,
        frameon=True,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=8.8,
        labelspacing=0.3,
        handletextpad=0.5,
        borderpad=0.4,
    )
    legend.get_frame().set_edgecolor("0.4")
    legend.get_frame().set_linewidth(0.7)
    legend.get_frame().set_boxstyle("Square", pad=0.4)


def _plot_headline_audit_panel(axis: plt.Axes, rows: list[dict[str, str]]) -> None:
    row_lookup = {(row["preset_name"], row["test_regime"]): row for row in rows}
    metrics = [
        (
            "mean_error_gap_vs_pod_lspg",
            "error_gap_ci95_low",
            "error_gap_ci95_high",
            "#1f77b4",
            "State gap",
            0.18,
        ),
        (
            "mean_qoi_gap_vs_pod_lspg",
            "qoi_gap_ci95_low",
            "qoi_gap_ci95_high",
            "#ff7f0e",
            "QoI gap",
            0.0,
        ),
        (
            "mean_residual_gap_vs_pod_lspg",
            "residual_gap_ci95_low",
            "residual_gap_ci95_high",
            "#2ca02c",
            "Residual gap",
            -0.18,
        ),
    ]
    y_positions = list(range(len(HEADLINE_ROWS)))
    x_lows: list[float] = []
    x_highs: list[float] = []

    for y, key in zip(y_positions, HEADLINE_ROWS):
        row = row_lookup[key]
        for mean_key, low_key, high_key, color, _, offset in metrics:
            mean = float(row[mean_key])
            low = float(row[low_key])
            high = float(row[high_key])
            x_lows.append(low)
            x_highs.append(high)
            axis.errorbar(
                mean,
                y + offset,
                xerr=[[mean - low], [high - mean]],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=2.0,
                capsize=3,
                markersize=6,
                zorder=3,
            )

    axis.axvline(0.0, color="black", linewidth=0.9, linestyle="--")
    axis.set_yticks(y_positions, [SHORT_HEADLINE_LABELS[key] for key in HEADLINE_ROWS])
    axis.tick_params(axis="y", rotation=30)
    for tick_label in axis.get_yticklabels():
        tick_label.set_ha("right")
        tick_label.set_rotation_mode("anchor")
    axis.invert_yaxis()
    axis.set_ylim(len(HEADLINE_ROWS) - 0.6, -0.9)
    x_span = max(x_highs) - min(x_lows)
    x_margin = 0.08 * max(0.5, x_span)
    axis.set_xlim(min(x_lows) - x_margin, max(x_highs) + x_margin)
    axis.set_xlabel(
        "Gap vs POD-LSPG (2.5--97.5% seed-percentile intervals)", fontsize=11.5
    )
    axis.tick_params(labelsize=10.5)
    axis.text(
        0.02,
        0.98,
        "(b)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
    )
    axis.grid(axis="x", alpha=0.18, linewidth=0.5)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            linestyle="-",
            linewidth=2.0,
            markersize=6,
            label=label,
        )
        for _, _, _, color, label, _ in metrics
    ]
    legend = axis.legend(
        handles=legend_handles,
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.90),
        fontsize=9.5,
        borderpad=0.4,
    )
    legend.get_frame().set_edgecolor("0.4")
    legend.get_frame().set_linewidth(0.7)
    legend.get_frame().set_boxstyle("Square", pad=0.4)


def _plot_error_residual(
    rows: list[dict[str, str]],
    png_path: Path,
    pdf_path: Path,
    *,
    title: str | None = "Error-Residual Tradeoff Against POD-LSPG",
) -> None:
    _plot_tradeoff_scatter(
        rows,
        x_key="error_gap_vs_pod_lspg",
        y_key="residual_gap_vs_pod_lspg",
        x_label="Error Gap vs POD-LSPG",
        y_label="Full-Order Residual Gap vs POD-LSPG",
        title=title,
        png_path=png_path,
        pdf_path=pdf_path,
        inside_legend_corner="upper right",
    )


def _plot_speed_residual(
    rows: list[dict[str, str]],
    png_path: Path,
    pdf_path: Path,
    *,
    title: str = "Speed-Residual Tradeoff Against POD-LSPG",
) -> None:
    _plot_tradeoff_scatter(
        rows,
        x_key="speedup_vs_full",
        y_key="residual_gap_vs_pod_lspg",
        x_label="Speedup vs Full-Order Solve",
        y_label="Full-Order Residual Gap vs POD-LSPG",
        title=title,
        png_path=png_path,
        pdf_path=pdf_path,
    )


def _plot_tradeoff_scatter(
    rows: list[dict[str, str]],
    *,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    title: str | None,
    png_path: Path,
    pdf_path: Path,
    inside_legend_corner: str | None = None,
    legend_box: bool = False,
) -> None:
    figure, axis = plt.subplots(figsize=(7.8, 5.8))
    present_families: list[str] = []
    for row in rows:
        marker, _ = SOLVER_STYLES[row["solver_variant"]]
        color = FAMILY_COLORS.get(row["problem_name"], "#444444")
        face_color = color if row["test_regime"] == "in_domain" else "none"
        if row["problem_name"] not in present_families:
            present_families.append(row["problem_name"])
        axis.scatter(
            float(row[x_key]),
            float(row[y_key]),
            s=80,
            marker=marker,
            facecolors=face_color,
            edgecolors=color,
            linewidths=1.6,
            alpha=0.95,
        )

    if "error_gap" in x_key:
        axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    if title:
        axis.set_title(title)
    axis.grid(alpha=0.15, linewidth=0.5)
    axis.tick_params(labelsize=10)

    solver_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="None",
            markersize=8,
            markerfacecolor="#ffffff",
            markeredgecolor="#111111",
            label=label,
        )
        for marker, label in SOLVER_STYLES.values()
    ]
    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=8,
            markerfacecolor=color,
            markeredgecolor=color,
            label=FAMILY_LABELS.get(problem_name, problem_name.replace("_", " ")),
        )
        for problem_name in present_families
        for color in [FAMILY_COLORS.get(problem_name, "#444444")]
    ]
    regime_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=8,
            markerfacecolor=("#111111" if regime == "in_domain" else "none"),
            markeredgecolor="#111111",
            label=REGIME_LABELS[regime],
        )
        for regime in ("in_domain", "ood")
    ]
    blocks = [
        ("Solver", solver_handles),
        ("Problem", family_handles),
        ("Regime", regime_handles),
    ]
    if inside_legend_corner == "upper right":
        anchor_x, legend_loc, anchor_y = 0.985, "upper right", 0.985
        legend_fontsize, line_height, block_gap = 8.5, 0.043, 0.028
    elif inside_legend_corner == "upper left":
        anchor_x, legend_loc, anchor_y = 0.015, "upper left", 0.985
        legend_fontsize, line_height, block_gap = 8.5, 0.043, 0.028
    else:
        anchor_x, legend_loc, anchor_y = 1.02, "upper left", 1.00
        legend_fontsize, line_height, block_gap = 10, 0.052, 0.035
    legends = []
    for index, (block_title, handles) in enumerate(blocks):
        legend = axis.legend(
            handles=handles,
            frameon=False,
            loc=legend_loc,
            bbox_to_anchor=(anchor_x, anchor_y),
            title=block_title,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            borderaxespad=0.0,
        )
        if index < len(blocks) - 1:
            axis.add_artist(legend)
        legends.append(legend)
        anchor_y -= (1 + len(handles)) * line_height + block_gap

    if inside_legend_corner:
        figure.tight_layout()
    else:
        figure.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))

    if legend_box and legends:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        union = Bbox.union([leg.get_window_extent(renderer) for leg in legends])
        inv = axis.transAxes.inverted()
        x0, y0 = inv.transform((union.x0, union.y0))
        x1, y1 = inv.transform((union.x1, union.y1))
        pad_x, pad_y = 0.018, 0.014
        axis.add_patch(
            Rectangle(
                (x0 - pad_x, y0 - pad_y),
                (x1 - x0) + 2 * pad_x,
                (y1 - y0) + 2 * pad_y,
                transform=axis.transAxes,
                fill=False,
                edgecolor="#444444",
                linewidth=0.8,
                zorder=6,
            )
        )

    figure.savefig(png_path, dpi=200)
    figure.savefig(pdf_path, metadata=PDF_METADATA)
    plt.close(figure)


def _plot_hydraulic_conductivity_benchmark(
    rows: list[dict], png_path: Path, pdf_path: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(7.4, 3.9))
    regimes = ["in_domain", "ood"]
    labels = ["In-domain", "Out-of-domain"]
    mgp_rows = {
        row["test_regime"]: row
        for row in rows
        if row["preset_name"] == "reference_hydrologic_conductivity"
    }
    metrics = [
        ("mgp_mean_error_l2", "lspg_mean_error_l2", "Relative State Error"),
        ("mgp_mean_qoi_error", "lspg_mean_qoi_error", "Outlet-Flux Proxy Error"),
        ("mgp_mean_residual", "lspg_mean_residual", "Full-order residual"),
    ]
    colors = {"mgp": "#0072B2", "lspg": "#D55E00"}
    hatches = {"mgp": "///", "lspg": "\\\\\\"}
    panel_letters = ["(a)", "(b)", "(c)"]
    x = list(range(len(regimes)))
    width = 0.34
    for panel_index, (axis, (mgp_key, lspg_key, title)) in enumerate(
        zip(axes, metrics)
    ):
        mgp_values = [mgp_rows[regime][mgp_key] for regime in regimes]
        lspg_values = [mgp_rows[regime][lspg_key] for regime in regimes]
        mgp_bars = axis.bar(
            [value - width / 2 for value in x],
            mgp_values,
            width=width,
            color=colors["mgp"],
            edgecolor="#222222",
            hatch=hatches["mgp"],
            label="Manifold Galerkin",
        )
        lspg_bars = axis.bar(
            [value + width / 2 for value in x],
            lspg_values,
            width=width,
            color=colors["lspg"],
            edgecolor="#222222",
            hatch=hatches["lspg"],
            label="POD-LSPG",
        )
        axis.text(
            0.04,
            0.97,
            panel_letters[panel_index],
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=12.5,
            fontweight="semibold",
        )
        axis.set_xticks(x, labels)
        axis.set_ylabel(title, fontsize=11.0)
        axis.tick_params(labelsize=10.0)
        headroom = 1.7 if panel_index == 0 else 1.32
        axis.set_ylim(0.0, max(max(mgp_values), max(lspg_values)) * headroom)
        for bars in (mgp_bars, lspg_bars):
            for bar in bars:
                height = bar.get_height()
                axis.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    rotation=90,
                )
    legend = axes[0].legend(
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(0.03, 0.92),
        fontsize=8.0,
        borderpad=0.3,
        labelspacing=0.3,
        handlelength=1.4,
        handletextpad=0.5,
        alignment="left",
    )
    legend.get_frame().set_edgecolor("0.4")
    legend.get_frame().set_linewidth(0.7)
    legend.get_frame().set_boxstyle("Square", pad=0.25)
    figure.tight_layout()
    figure.savefig(png_path, dpi=200)
    figure.savefig(pdf_path, metadata=PDF_METADATA)
    plt.close(figure)


def _plot_solver_robustness(
    solver_table_rows: list[dict], png_path: Path, pdf_path: Path
) -> None:
    row_lookup = {(r["preset_name"], r["test_regime"]): r for r in solver_table_rows}
    order = [
        ("reference_bratu_source", "in_domain"),
        ("reference_bratu_source", "ood"),
        ("reference_front_layer", "in_domain"),
        ("reference_front_layer", "ood"),
        ("reference_nonlinear_diffusion", "in_domain"),
        ("reference_nonlinear_diffusion", "ood"),
        ("reference_hydrologic_conductivity", "in_domain"),
        ("reference_hydrologic_conductivity", "ood"),
    ]
    order = [key for key in order if key in row_lookup]
    labels = [FINALIZED_LABELS[key] for key in order]
    series = [
        ("Manifold Galerkin", "mgp_failed_case_count", "#0072B2", "///"),
        ("Manifold-LSPG", "mgp_lspg_failed_case_count", "#D55E00", "\\\\\\"),
        ("POD-LSPG", "pod_failed_case_count", "#7f7f7f", ".."),
    ]
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    y_positions = list(range(len(order)))
    bar_h = 0.26
    offsets = [bar_h, 0.0, -bar_h]
    for (name, key, color, hatch), offset in zip(series, offsets):
        values = []
        for k in order:
            row = row_lookup[k]
            total = float(row.get("total_case_count") or row.get("num_runs") or 1)
            values.append(100.0 * (total - float(row.get(key, 0))) / total)
        bars = axis.barh(
            [yi + offset for yi in y_positions],
            values,
            height=bar_h,
            color=color,
            edgecolor="#222222",
            linewidth=0.6,
            hatch=hatch,
            label=name,
        )
        for bar, value in zip(bars, values):
            if value < 99.5:
                axis.text(
                    value + 1.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0f}",
                    va="center",
                    ha="left",
                    fontsize=7.5,
                )
    axis.set_yticks(y_positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 105)
    axis.set_xticks([0, 20, 40, 60, 80, 100])
    axis.set_xlabel("Converged solves (% of held-out samples)", fontsize=11.5)
    axis.tick_params(labelsize=10.5)
    axis.grid(axis="x", alpha=0.18, linewidth=0.5)
    axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fontsize=9.5,
        columnspacing=1.6,
        handletextpad=0.6,
    )
    figure.tight_layout()
    figure.savefig(png_path, dpi=200)
    figure.savefig(pdf_path, metadata=PDF_METADATA)
    plt.close(figure)


def _plot_solution_fields(field_csv: Path, png_path: Path, pdf_path: Path) -> None:
    if not field_csv.exists():
        return None
    rows = _read_csv(field_csv)
    x = [float(r["x"]) for r in rows]
    u_fom = [float(r["u_fom"]) for r in rows]

    def rel_state_error(col: str) -> float:
        num = sum((float(r[col]) - float(r["u_fom"])) ** 2 for r in rows) ** 0.5
        den = sum(float(r["u_fom"]) ** 2 for r in rows) ** 0.5
        return num / den if den else float("nan")

    series = [
        ("u_manifold_galerkin", "Manifold Galerkin", "#0072B2", (0, (5, 1.5))),
        ("u_pod_lspg", "POD-LSPG", "#009E73", (0, (4, 1.2, 1, 1.2))),
        ("u_manifold_lspg", "Manifold-LSPG", "#D55E00", (0, (1, 1.4))),
    ]
    figure, axis = plt.subplots(figsize=(7.0, 4.6))
    axis.plot(
        x, u_fom, color="#111111", linewidth=2.6, label="Full-order solution", zorder=2
    )
    for col, name, color, dash in series:
        y = [float(r[col]) for r in rows]
        axis.plot(
            x,
            y,
            color=color,
            linewidth=2.0,
            linestyle=dash,
            label=f"{name} (rel. state error {rel_state_error(col):.3f})",
            zorder=3,
        )
    axis.set_xlabel("x", fontsize=11.5)
    axis.set_ylabel("Solution u(x)", fontsize=11.5)
    axis.set_xlim(0.0, 1.0)
    axis.tick_params(labelsize=10.5)
    axis.grid(alpha=0.18, linewidth=0.5)
    axis.legend(frameon=False, fontsize=9.5, loc="upper left")
    figure.tight_layout()
    figure.savefig(png_path, dpi=200)
    figure.savefig(pdf_path, metadata=PDF_METADATA)
    plt.close(figure)


def _plot_selected_gap_solver_delta_matrix_figure(
    *,
    highlighted_rows: list[dict[str, str]],
    solver_table_rows: list[dict],
    png_path: Path,
    pdf_path: Path,
) -> None:
    figure, axes = plt.subplots(
        2, 1, figsize=(7.4, 9.6), gridspec_kw={"height_ratios": [1.0, 1.10]}
    )
    _plot_selected_gap_matrix_panel(axes[0], highlighted_rows)
    _plot_solver_delta_matrix_panel(axes[1], solver_table_rows)
    figure.subplots_adjust(left=0.18, right=0.98, top=0.96, bottom=0.12, hspace=0.6)
    figure.savefig(png_path, dpi=220)
    figure.savefig(pdf_path, metadata=PDF_METADATA)
    plt.close(figure)


def _plot_selected_gap_matrix_panel(axis: plt.Axes, rows: list[dict[str, str]]) -> None:
    row_lookup = {(row["preset_name"], row["test_regime"]): row for row in rows}
    metric_specs = [
        (
            "mean_error_gap_vs_pod_lspg",
            "error_gap_ci95_low",
            "error_gap_ci95_high",
            "State gap",
        ),
        (
            "mean_qoi_gap_vs_pod_lspg",
            "qoi_gap_ci95_low",
            "qoi_gap_ci95_high",
            "QoI gap",
        ),
        (
            "mean_residual_gap_vs_pod_lspg",
            "residual_gap_ci95_low",
            "residual_gap_ci95_high",
            "Full-order residual",
        ),
    ]
    raw_matrix: list[list[float]] = []
    interval_crosses_zero: list[list[bool]] = []
    for key in HEADLINE_ROWS:
        row = row_lookup[key]
        raw_values: list[float] = []
        cross_flags: list[bool] = []
        for mean_key, low_key, high_key, _ in metric_specs:
            mean = float(row[mean_key])
            low = float(row[low_key])
            high = float(row[high_key])
            raw_values.append(mean)
            cross_flags.append(low <= 0.0 <= high)
        raw_matrix.append(raw_values)
        interval_crosses_zero.append(cross_flags)

    scaled_matrix = _column_scaled(
        [[-value for value in values] for values in raw_matrix]
    )
    axis.imshow(scaled_matrix, cmap="PRGn", vmin=-1.0, vmax=1.0, aspect="auto")
    axis.text(
        0.0,
        1.0,
        "(a)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="semibold",
    )
    axis.set_xticks(
        range(len(metric_specs)),
        [label for _, _, _, label in metric_specs],
        rotation=25,
        ha="right",
    )
    axis.set_yticks(
        range(len(HEADLINE_ROWS)), [SHORT_HEADLINE_LABELS[key] for key in HEADLINE_ROWS]
    )
    axis.set_ylim(len(HEADLINE_ROWS) - 0.5, -0.95)
    axis.set_xlim(-0.5, 5.2)
    axis.set_xticks([value - 0.5 for value in range(1, len(metric_specs))], minor=True)
    axis.set_yticks([value - 0.5 for value in range(1, len(HEADLINE_ROWS))], minor=True)
    axis.grid(which="minor", color="#dddddd", linewidth=0.8)
    axis.tick_params(which="minor", bottom=False, left=False)

    for i, row_values in enumerate(raw_matrix):
        for j, value in enumerate(row_values):
            axis.text(
                j,
                i,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="semibold",
            )
            if interval_crosses_zero[i][j]:
                axis.add_patch(
                    Circle(
                        (j + 0.33, i - 0.33),
                        radius=0.07,
                        facecolor="white",
                        edgecolor="#222222",
                        linewidth=0.8,
                    )
                )

    seeds_col_x = 3.45
    failed_col_x = 4.70
    axis.axvline(2.6, color="#bbbbbb", linewidth=1.0)
    axis.text(
        seeds_col_x,
        -0.68,
        "Lower-error seeds\nstate / QoI",
        fontsize=8.3,
        fontweight="semibold",
        ha="center",
        va="bottom",
    )
    axis.text(
        failed_col_x,
        -0.68,
        "Failed solves\nMG / M-LSPG",
        fontsize=8.3,
        fontweight="semibold",
        ha="center",
        va="bottom",
    )
    for i, key in enumerate(HEADLINE_ROWS):
        row = row_lookup[key]
        axis.text(
            seeds_col_x,
            i,
            f"{row['mgp_error_wins_vs_lspg']}/{row['mgp_qoi_wins_vs_lspg']}",
            ha="center",
            va="center",
            fontsize=9,
        )
        axis.text(
            failed_col_x,
            i,
            f"{row['mgp_failed_case_count']}/{row['mgp_lspg_failed_case_count']}",
            ha="center",
            va="center",
            fontsize=9,
        )


def _plot_solver_delta_matrix_panel(
    axis: plt.Axes, solver_table_rows: list[dict]
) -> None:
    row_lookup = {
        (row["preset_name"], row["test_regime"]): row for row in solver_table_rows
    }
    column_labels = [
        "State Δ",
        "QoI Δ",
        "Residual Δ",
        "MG speedup -\nM-LSPG speedup",
        "Additional\nfailed solves",
    ]
    raw_matrix: list[list[float]] = []
    for key in FINALIZED_ROWS:
        row = row_lookup[key]
        raw_matrix.append(
            [
                float(row["mgp_lspg_mean_error_l2"]) - float(row["mgp_mean_error_l2"]),
                float(row["mgp_lspg_mean_qoi_error"])
                - float(row["mgp_mean_qoi_error"]),
                float(row["mgp_lspg_mean_residual"]) - float(row["mgp_mean_residual"]),
                float(row["mgp_speedup_vs_full"])
                - float(row["mgp_lspg_speedup_vs_full"]),
                float(row["mgp_lspg_failed_case_count"])
                - float(row["mgp_failed_case_count"]),
            ]
        )

    scaled_matrix = _column_scaled(raw_matrix)
    axis.imshow(scaled_matrix, cmap="PRGn", vmin=-1.0, vmax=1.0, aspect="auto")
    axis.set_ylim(len(FINALIZED_ROWS) - 0.5, -1.2)
    axis.text(
        0.0,
        1.0,
        "(b)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="semibold",
    )
    axis.set_xticks(range(len(column_labels)), column_labels, rotation=25, ha="right")
    axis.set_yticks(
        range(len(FINALIZED_ROWS)), [FINALIZED_LABELS[key] for key in FINALIZED_ROWS]
    )
    axis.set_xticks([value - 0.5 for value in range(1, len(column_labels))], minor=True)
    axis.set_yticks(
        [value - 0.5 for value in range(1, len(FINALIZED_ROWS))], minor=True
    )
    axis.grid(which="minor", color="#dddddd", linewidth=0.8)
    axis.tick_params(which="minor", bottom=False, left=False)

    for i, row_values in enumerate(raw_matrix):
        for j, value in enumerate(row_values):
            display = f"{value:+.3f}" if j < 4 else f"{int(round(value)):+d}"
            axis.text(
                j,
                i,
                display,
                ha="center",
                va="center",
                fontsize=8.8,
                fontweight="semibold",
            )


def _column_scaled(matrix: list[list[float]]) -> list[list[float]]:
    if not matrix:
        return matrix
    num_cols = len(matrix[0])
    scales = []
    for col in range(num_cols):
        max_abs = max(abs(row[col]) for row in matrix)
        scales.append(max_abs if max_abs > 0 else 1.0)
    return [[row[col] / scales[col] for col in range(num_cols)] for row in matrix]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render publication figures from the summary CSVs."
    )
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    for path in render_publication_figures(
        artifacts_dir=args.artifacts_dir, output_dir=args.output_dir
    ):
        print(path)


if __name__ == "__main__":
    main()
