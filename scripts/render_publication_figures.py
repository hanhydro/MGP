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
from matplotlib import text as mpl_text

PRINT_WIDTH = 5.69

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "mathtext.bf": "Arial:bold",
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


def render_publication_figures(
    *,
    artifacts_dir: Path = ARTIFACTS,
    output_dir: Path | None = None,
) -> list[Path]:
    output_dir = output_dir or artifacts_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    acceptance_summary = _read_json(artifacts_dir / "acceptance_statistics.json")
    prospective_summary = _read_json(
        ROOT / "eval" / "prospective_weight_validation_2026-07-17"
        / "first_look_output" / "FIRST_LOOK_SUMMARY.json"
    )
    prospective_2d_summary = _read_json(
        ROOT / "eval" / "prospective_2d_scale_warmstart_2026-07-19"
        / "first_look_output" / "FIRST_LOOK_SUMMARY.json"
    )

    headline_png = output_dir / "figure1_solver_aware_headline.png"
    headline_pdf = output_dir / "figure1_solver_aware_headline.pdf"
    prospective_png = output_dir / "figure1_prospective_validation.png"
    prospective_pdf = output_dir / "figure1_prospective_validation.pdf"
    prospective_2d_png = output_dir / "figure2_prospective_2d_warmstart.png"
    prospective_2d_pdf = output_dir / "figure2_prospective_2d_warmstart.pdf"
    fields_png = output_dir / "figure_solution_fields.png"
    fields_pdf = output_dir / "figure_solution_fields.pdf"

    controlled_contrasts = list(acceptance_summary["controlled_contrasts"])
    controlled_contrasts.extend(
        acceptance_summary["factorial_core"]["figure_interactions"]
    )
    _plot_controlled_contrast_headline_figure(
        controlled_contrasts,
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
    _plot_solution_fields(
        artifacts_dir / "solution_fields_front_layer.csv", fields_png, fields_pdf
    )
    return [
        headline_png,
        headline_pdf,
        prospective_png,
        prospective_pdf,
        prospective_2d_png,
        prospective_2d_pdf,
        fields_png,
        fields_pdf,
    ]


def _fit_to_print(figure: plt.Figure, font_scale: float = 0.88, height: float | None = None) -> None:
    figure.canvas.draw()
    tick_labels = set()
    for axis in figure.axes:
        for which, ticks in (("x", axis.get_xticklabels()), ("y", axis.get_yticklabels())):
            if ticks:
                axis.tick_params(axis=which, labelsize=ticks[0].get_fontsize() * font_scale)
            tick_labels.update(id(t) for t in ticks)
            tick_labels.update(id(t) for t in (axis.get_xticklabels(minor=True) if which == "x" else axis.get_yticklabels(minor=True)))
    for text in figure.findobj(mpl_text.Text):
        if id(text) not in tick_labels and text.get_text():
            text.set_fontsize(text.get_fontsize() * font_scale)
    fig_width, fig_height = figure.get_size_inches()
    figure.set_size_inches(PRINT_WIDTH, height or fig_height * PRINT_WIDTH / fig_width)


def _panel_title(axis: plt.Axes, title: str, **kwargs) -> None:
    letter, _, rest = title.partition(" ")
    axis.set_title(rf"$\mathbf{{{letter}}}$ {rest}", math_fontfamily="custom", **kwargs)


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
        ("hydrologic_conductivity", "in_domain", "Hydr.\nin"),
        ("hydrologic_conductivity", "ood", "Hydr.\nOOD"),
        ("nonlinear_diffusion", "in_domain", "Diff.\nin"),
        ("nonlinear_diffusion", "ood", "Diff.\nOOD"),
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
            row_index - 0.33,
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
    _panel_title(suite_axis, "(a) Seed contrasts", loc="left", fontsize=10)
    suite_axis.grid(axis="x", alpha=0.18, linewidth=0.6)
    suite_axis.set_ylim(len(primary) - 0.5, -0.62)
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
    _panel_title(cell_axis, "(b) Cellwise heterogeneity", loc="left", fontsize=10)
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
    _fit_to_print(figure)
    figure.tight_layout()
    figure.savefig(png_path, dpi=220)
    figure.savefig(pdf_path, metadata=PDF_METADATA)
    plt.close(figure)


def _plot_prospective_2d_warmstart_figure(
    summary: dict,
    *,
    png_path: Path,
    pdf_path: Path,
) -> None:

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
    contrast_axis.set_xlim(0.36, 2.25)
    contrast_axis.set_xticks(
        [0.4, 0.5, 0.75, 1.0, 1.5, 2.0], ["0.4", "0.5", "0.75", "1", "1.5", "2"]
    )
    contrast_axis.xaxis.set_minor_formatter(plt.NullFormatter())
    contrast_axis.set_yticks(
        y_positions,
        [
            f'{contrast_id}: {primary[contrast_id]["label"]}'.replace(" / ", " /\n")
            for contrast_id in contrast_ids
        ],
    )
    contrast_axis.invert_yaxis()
    contrast_axis.set_xlabel("Error ratio\n(first method / second method)")
    _panel_title(contrast_axis, "(a) Prespecified comparisons", loc="left", fontsize=10)
    contrast_axis.grid(axis="x", alpha=0.2, linewidth=0.6)
    contrast_axis.tick_params(labelsize=8.8)
    contrast_axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1, frameon=False, fontsize=8.6)

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
    arm_axis.set_xticks([1.0, 1.5, 2.0, 3.0], ["1", "1.5", "2", "3"])
    arm_axis.xaxis.set_minor_formatter(plt.NullFormatter())
    arm_axis.set_yticks(range(len(arm_order)), arm_order)
    arm_axis.invert_yaxis()
    arm_axis.set_xlabel("Error ratio\n(training-mean / nearest start)")
    _panel_title(arm_axis, "(b) Per-arm start sensitivity", loc="left", fontsize=10)
    arm_axis.grid(axis="x", alpha=0.2, linewidth=0.6)
    arm_axis.tick_params(labelsize=8.8)
    arm_axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1, frameon=False, fontsize=8.6)
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

    _fit_to_print(figure)
    figure.set_layout_engine("constrained")
    figure.savefig(png_path, dpi=220)
    figure.savefig(pdf_path, metadata=PDF_METADATA)
    plt.close(figure)


def _plot_controlled_contrast_headline_figure(
    rows: list[dict],
    *,
    png_path: Path,
    pdf_path: Path,
) -> None:
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
        _panel_title(axis, title, loc="left", fontsize=10)
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
    legend = figure.legend(
        handles=problem_handles + regime_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8.0,
        handletextpad=0.4,
        columnspacing=1.0,
        labelspacing=0.4,
        borderpad=0.5,
    )
    legend.set_zorder(6)
    _fit_to_print(figure, height=3.9)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(png_path, dpi=220)
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
