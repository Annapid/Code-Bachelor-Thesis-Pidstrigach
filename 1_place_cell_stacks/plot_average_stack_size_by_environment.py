#!/usr/bin/env python
"""Plot average 1 Hz place-cell stack size by environment.

This script reuses the compact CSV written by ``analyze_place_cell_stacks.py``.
It does not reload cached experiments or recompute ratemaps.

For each rat and environment it first averages ``active_cell_count`` over all
grid positions. It then plots the environment mean and standard deviation
across rats.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


DEFAULT_EXPERIMENT_ORDER = (
    "exp_scales_a",
    "exp_scales_b",
    "exp_scales_c",
    "exp_scales_d",
)

EXPERIMENT_LABELS = {
    "exp_scales_a": "A",
    "exp_scales_b": "B",
    "exp_scales_c": "C",
    "exp_scales_d": "D",
    "exp_scales_a2": "A2",
}

EXPERIMENT_COLORS = {
    "exp_scales_a": "#C8E4FC",
    "exp_scales_b": "#88B888",
    "exp_scales_c": "#CCE0C8",
    "exp_scales_d": "#006498",
    "exp_scales_a2": "#C8E4FC",
}


def find_repo_root(start: Optional[Path] = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for path in (start.resolve(), *start.resolve().parents):
        if (path / "MultiplePFAnalysis" / "__init__.py").is_file():
            return path
    raise RuntimeError("Could not find repository root containing MultiplePFAnalysis")


REPO_ROOT = find_repo_root()


def environment_label(experiment_id: str, arena: str | None = None) -> str:
    if arena:
        return arena
    return EXPERIMENT_LABELS.get(experiment_id, experiment_id)


def read_rat_environment_means(
    csv_path: Path,
    *,
    threshold_hz: float,
    value_column: str = "active_cell_count",
) -> tuple[dict[tuple[str, str], float], dict[str, str]]:
    """Return a within-rat/environment mean for one saved summary column."""
    grouped_counts: dict[tuple[str, str], list[float]] = defaultdict(list)
    environment_labels: dict[str, str] = {}

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "rat_id",
            "experiment_id",
            "arena",
            "threshold_hz",
            value_column,
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(sorted(missing))}"
            )

        for row in reader:
            row_threshold = float(row["threshold_hz"])
            if not np.isclose(row_threshold, threshold_hz):
                continue
            rat_id = row["rat_id"]
            experiment_id = row["experiment_id"]
            grouped_counts[(rat_id, experiment_id)].append(
                float(row[value_column])
            )
            environment_labels[experiment_id] = environment_label(
                experiment_id,
                row.get("arena"),
            )

    if not grouped_counts:
        raise ValueError(
            f"No rows with threshold_hz={threshold_hz:g} found in {csv_path}"
        )

    rat_environment_means = {
        key: float(np.mean(values))
        for key, values in grouped_counts.items()
        if values
    }
    return rat_environment_means, environment_labels


def summarize_by_environment(
    rat_environment_means: dict[tuple[str, str], float],
    experiment_order: tuple[str, ...],
) -> list[dict]:
    rows = []
    for experiment_id in experiment_order:
        rat_values = sorted(
            (
                (rat_id, value)
                for (rat_id, row_experiment_id), value in rat_environment_means.items()
                if row_experiment_id == experiment_id
            ),
            key=lambda item: item[0],
        )
        if not rat_values:
            continue
        values = np.asarray([value for _, value in rat_values], dtype=float)
        rows.append(
            {
                "experiment_id": experiment_id,
                "rat_ids": [rat_id for rat_id, _ in rat_values],
                "rat_means": values,
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n_rats": int(len(values)),
            }
        )
    if not rows:
        raise ValueError("None of the requested environments were present in the CSV")
    return rows


def paired_t_test(values_a: np.ndarray, values_b: np.ndarray) -> tuple[float, float]:
    """Two-sided paired t-test for rat-level environment means."""
    differences = np.asarray(values_b, dtype=float) - np.asarray(values_a, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) < 2:
        return np.nan, np.nan

    mean_difference = float(np.mean(differences))
    sd_difference = float(np.std(differences, ddof=1))
    if sd_difference == 0.0:
        if mean_difference == 0.0:
            return 0.0, 1.0
        return float(np.sign(mean_difference) * np.inf), 0.0

    t_statistic = mean_difference / (sd_difference / np.sqrt(len(differences)))
    p_value = float(2.0 * stats.t.sf(abs(t_statistic), df=len(differences) - 1))
    return float(t_statistic), p_value


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values."""
    p_values_array = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values_array), np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(p_values_array))
    if len(finite_indices) == 0:
        return adjusted.tolist()

    ordered_indices = finite_indices[np.argsort(p_values_array[finite_indices])]
    running_max = 0.0
    m = len(ordered_indices)
    for rank, original_index in enumerate(ordered_indices):
        raw_adjusted = (m - rank) * p_values_array[original_index]
        running_max = max(running_max, raw_adjusted)
        adjusted[original_index] = min(1.0, running_max)
    return adjusted.tolist()


def pairwise_environment_tests(summaries: list[dict]) -> list[dict]:
    """Pairwise paired t-tests using rats as repeated units."""
    rows = []
    raw_p_values = []
    for first_index, second_index in combinations(range(len(summaries)), 2):
        first = summaries[first_index]
        second = summaries[second_index]
        first_by_rat = dict(zip(first["rat_ids"], first["rat_means"]))
        second_by_rat = dict(zip(second["rat_ids"], second["rat_means"]))
        paired_rat_ids = sorted(set(first_by_rat).intersection(second_by_rat))
        first_values = np.asarray([first_by_rat[rat_id] for rat_id in paired_rat_ids])
        second_values = np.asarray([second_by_rat[rat_id] for rat_id in paired_rat_ids])
        differences = second_values - first_values
        t_statistic, p_value = paired_t_test(first_values, second_values)
        raw_p_values.append(p_value)
        rows.append(
            {
                "environment_1": first["experiment_id"],
                "environment_2": second["experiment_id"],
                "n_paired_rats": len(paired_rat_ids),
                "mean_environment_1": float(np.mean(first_values)),
                "mean_environment_2": float(np.mean(second_values)),
                "mean_difference_2_minus_1": float(np.mean(differences)),
                "sd_difference": (
                    float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
                ),
                "paired_rat_ids": ";".join(paired_rat_ids),
                "paired_t_statistic": t_statistic,
                "p_paired_ttest_two_sided": p_value,
            }
        )

    adjusted_p_values = holm_adjust(raw_p_values)
    for row, adjusted_p in zip(rows, adjusted_p_values):
        row["p_holm"] = adjusted_p
        row["significant_holm_0p05"] = bool(np.isfinite(adjusted_p) and adjusted_p < 0.05)
    return rows


def write_pairwise_tests_csv(
    rows: list[dict],
    output_path: Path,
    environment_labels: dict[str, str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "environment_1",
        "environment_1_label",
        "environment_2",
        "environment_2_label",
        "n_paired_rats",
        "mean_environment_1",
        "mean_environment_2",
        "mean_difference_2_minus_1",
        "sd_difference",
        "paired_rat_ids",
        "paired_t_statistic",
        "p_paired_ttest_two_sided",
        "p_holm",
        "significant_holm_0p05",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "environment_1_label": environment_label(
                        row["environment_1"],
                        environment_labels.get(row["environment_1"]),
                    ),
                    "environment_2_label": environment_label(
                        row["environment_2"],
                        environment_labels.get(row["environment_2"]),
                    ),
                }
            )


def significance_label(p_value: float, *, include_ns_p_value: bool = False) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if include_ns_p_value:
        return f"ns, p={p_value:.3g}"
    return "ns"


def add_significance_brackets(
    ax,
    pairwise_rows: list[dict],
    experiment_ids: list[str],
    y_start: float,
    y_step: float,
    *,
    annotate_all: bool = False,
    include_ns_p_value: bool = False,
) -> float:
    """Annotate pairwise comparisons."""
    experiment_to_x = {experiment_id: index for index, experiment_id in enumerate(experiment_ids)}
    y = y_start
    any_brackets = False
    for row in pairwise_rows:
        if not annotate_all and not row["significant_holm_0p05"]:
            continue
        if row["environment_1"] not in experiment_to_x or row["environment_2"] not in experiment_to_x:
            continue
        x1 = experiment_to_x[row["environment_1"]]
        x2 = experiment_to_x[row["environment_2"]]
        ax.plot([x1, x1, x2, x2], [y, y + y_step * 0.25, y + y_step * 0.25, y],
                color="black", linewidth=1.0)
        ax.text(
            (x1 + x2) / 2.0,
            y + y_step * 0.3,
            significance_label(
                row["p_holm"],
                include_ns_p_value=include_ns_p_value,
            ),
            ha="center",
            va="bottom",
            fontsize=9,
        )
        y += y_step
        any_brackets = True
    return y if any_brackets else y_start


def plot_average_stack_size(
    summaries: list[dict],
    environment_labels: dict[str, str],
    output_path: Path,
    *,
    threshold_hz: float,
    dpi: int,
    pairwise_rows: list[dict] | None = None,
    textless: bool = False,
    tick_label_size: int = 14,
    annotate_all_pairwise: bool = False,
    ylabel: str = "Average stack size\n(active place cells per grid position)",
    title: str = "Average 1 Hz place-cell stack size by environment",
    annotation_unit: float = 1.0,
    minimum_y_max: float = 1.0,
    base_y_padding: float = 2.0,
) -> None:
    labels = [
        environment_label(row["experiment_id"], environment_labels.get(row["experiment_id"]))
        for row in summaries
    ]
    means = np.asarray([row["mean"] for row in summaries], dtype=float)
    sds = np.asarray([row["sd"] for row in summaries], dtype=float)
    colors = [
        EXPERIMENT_COLORS.get(row["experiment_id"], "#4C78A8")
        for row in summaries
    ]
    x = np.arange(len(summaries))

    # All six pairwise brackets need additional vertical room, especially when
    # the outcome is a fraction with a compact y-axis.
    figure_height = 6.4 if annotate_all_pairwise else 4.8
    fig, ax = plt.subplots(figsize=(7.2, figure_height))
    bars = ax.bar(
        x,
        means,
        yerr=sds,
        capsize=6,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
        label="Mean ± SD across rats",
    )

    for row_index, row in enumerate(summaries):
        rat_means = np.asarray(row["rat_means"], dtype=float)
        jitter = np.linspace(-0.11, 0.11, len(rat_means)) if len(rat_means) > 1 else [0.0]
        ax.scatter(
            row_index + np.asarray(jitter),
            rat_means,
            s=34,
            color="white",
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
        )
        if not textless and not annotate_all_pairwise:
            ax.text(
                bars[row_index].get_x() + bars[row_index].get_width() / 2.0,
                means[row_index] + sds[row_index] + 0.45,
                f"n={row['n_rats']}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=tick_label_size if textless else None)
    if textless:
        ax.tick_params(axis="both", which="major", labelsize=tick_label_size, length=6, width=1.4)
    else:
        ax.set_xlabel("Environment")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{title}\n"
            "positions averaged within rat; mean ± SD across rats"
        )
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.margins(x=0.08)
    y_max = max(minimum_y_max, float(np.nanmax(means + sds)) + base_y_padding)
    if pairwise_rows and not textless:
        bracket_top = add_significance_brackets(
            ax,
            pairwise_rows,
            [row["experiment_id"] for row in summaries],
            y_start=float(np.nanmax(means + sds)) + annotation_unit * (2.2 if annotate_all_pairwise else 1.0),
            y_step=annotation_unit * (1.45 if annotate_all_pairwise else 1.0),
            annotate_all=annotate_all_pairwise,
            include_ns_p_value=annotate_all_pairwise,
        )
        y_max = max(
            y_max,
            bracket_top + annotation_unit * (1.3 if annotate_all_pairwise else 1.0),
        )
    ax.set_ylim(0, y_max)

    if not textless:
        threshold_note = f"threshold = {threshold_hz:g} Hz"
        if pairwise_rows:
            threshold_note += "\np: paired t-test, Holm corrected"
        ax.text(
            0.99,
            0.02,
            threshold_note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="0.35",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_results_dir = (
        REPO_ROOT / "MultiplePFAnalysis" / "results" / "stacks_with_ratemaps"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=default_results_dir / "overall" / "linearized_stack_activity_all_rats.csv",
        help="CSV generated by analyze_place_cell_stacks.py.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=default_results_dir / "average_stack_size_by_environment_1hz.png",
        help="Output figure path. Defaults to the root of results/stacks_with_ratemaps.",
    )
    parser.add_argument(
        "--no-text-output-path",
        type=Path,
        default=default_results_dir / "average_stack_size_by_environment_1hz_no_text.png",
        help="Text-free output figure path. Set to 'none' to skip.",
    )
    parser.add_argument(
        "--significance-figure-output-path",
        type=Path,
        default=default_results_dir / "average_stack_size_by_environment_1hz_with_significance.png",
        help="Barplot with all pairwise Holm-corrected significance annotations. Set to 'none' to skip.",
    )
    parser.add_argument(
        "--significance-output-csv",
        type=Path,
        default=default_results_dir / "average_stack_size_by_environment_1hz_pairwise_significance.csv",
        help="CSV with paired rat-level pairwise environment tests.",
    )
    parser.add_argument(
        "--fraction-output-path",
        type=Path,
        default=default_results_dir / "average_stack_fraction_by_environment_1hz.png",
        help="Fraction-of-place-cells barplot output path. Set to 'none' to skip.",
    )
    parser.add_argument(
        "--fraction-no-text-output-path",
        type=Path,
        default=default_results_dir / "average_stack_fraction_by_environment_1hz_no_text.png",
        help="Text-free fraction barplot output path. Set to 'none' to skip.",
    )
    parser.add_argument(
        "--fraction-significance-figure-output-path",
        type=Path,
        default=default_results_dir / "average_stack_fraction_by_environment_1hz_with_significance.png",
        help="Fraction barplot with all pairwise Holm-corrected annotations. Set to 'none' to skip.",
    )
    parser.add_argument(
        "--fraction-significance-output-csv",
        type=Path,
        default=default_results_dir / "average_stack_fraction_by_environment_1hz_pairwise_significance.csv",
        help="CSV with paired rat-level pairwise tests of active-place-cell fractions.",
    )
    parser.add_argument(
        "--experiment-order",
        nargs="+",
        default=list(DEFAULT_EXPERIMENT_ORDER),
        help="Environment IDs to plot, in x-axis order.",
    )
    parser.add_argument(
        "--threshold-hz",
        type=float,
        default=1.0,
        help="Only rows with this saved active-cell threshold are used.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--no-text-tick-label-size",
        type=int,
        default=16,
        help="Tick-label font size for the text-free plot. Default is 16.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rat_environment_means, environment_labels = read_rat_environment_means(
        args.input_csv,
        threshold_hz=args.threshold_hz,
    )
    summaries = summarize_by_environment(
        rat_environment_means,
        tuple(args.experiment_order),
    )
    pairwise_rows = pairwise_environment_tests(summaries)
    write_pairwise_tests_csv(
        pairwise_rows,
        args.significance_output_csv,
        environment_labels,
    )
    plot_average_stack_size(
        summaries,
        environment_labels,
        args.output_path,
        threshold_hz=args.threshold_hz,
        dpi=args.dpi,
        pairwise_rows=pairwise_rows,
    )
    print(f"Saved {args.output_path}")
    print(f"Saved {args.significance_output_csv}")
    if str(args.significance_figure_output_path).lower() != "none":
        plot_average_stack_size(
            summaries,
            environment_labels,
            args.significance_figure_output_path,
            threshold_hz=args.threshold_hz,
            dpi=args.dpi,
            pairwise_rows=pairwise_rows,
            annotate_all_pairwise=True,
        )
        print(f"Saved {args.significance_figure_output_path}")
    if str(args.no_text_output_path).lower() != "none":
        plot_average_stack_size(
            summaries,
            environment_labels,
            args.no_text_output_path,
            threshold_hz=args.threshold_hz,
            dpi=args.dpi,
            textless=True,
            tick_label_size=args.no_text_tick_label_size,
        )
        print(f"Saved {args.no_text_output_path}")

    fraction_rat_environment_means, _ = read_rat_environment_means(
        args.input_csv,
        threshold_hz=args.threshold_hz,
        value_column="active_place_cell_fraction",
    )
    fraction_summaries = summarize_by_environment(
        fraction_rat_environment_means,
        tuple(args.experiment_order),
    )
    fraction_pairwise_rows = pairwise_environment_tests(fraction_summaries)
    write_pairwise_tests_csv(
        fraction_pairwise_rows,
        args.fraction_significance_output_csv,
        environment_labels,
    )
    fraction_plot_kwargs = {
        "ylabel": "Average stack fraction\n(active place-cell fraction per grid position)",
        "title": "Average 1 Hz place-cell stack fraction by environment",
        "annotation_unit": 0.025,
        "minimum_y_max": 0.25,
        "base_y_padding": 0.04,
    }
    if str(args.fraction_output_path).lower() != "none":
        plot_average_stack_size(
            fraction_summaries,
            environment_labels,
            args.fraction_output_path,
            threshold_hz=args.threshold_hz,
            dpi=args.dpi,
            pairwise_rows=fraction_pairwise_rows,
            **fraction_plot_kwargs,
        )
        print(f"Saved {args.fraction_output_path}")
    if str(args.fraction_significance_figure_output_path).lower() != "none":
        plot_average_stack_size(
            fraction_summaries,
            environment_labels,
            args.fraction_significance_figure_output_path,
            threshold_hz=args.threshold_hz,
            dpi=args.dpi,
            pairwise_rows=fraction_pairwise_rows,
            annotate_all_pairwise=True,
            **fraction_plot_kwargs,
        )
        print(f"Saved {args.fraction_significance_figure_output_path}")
    if str(args.fraction_no_text_output_path).lower() != "none":
        plot_average_stack_size(
            fraction_summaries,
            environment_labels,
            args.fraction_no_text_output_path,
            threshold_hz=args.threshold_hz,
            dpi=args.dpi,
            textless=True,
            tick_label_size=args.no_text_tick_label_size,
            **fraction_plot_kwargs,
        )
        print(f"Saved {args.fraction_no_text_output_path}")


if __name__ == "__main__":
    main()
