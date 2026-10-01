#!/usr/bin/env python3
"""Plot the mean ratemap of all active cells in one place-cell stack.

Examples:
    python MultiplePFAnalysis/1_place_cell_stacks/plot_average_stack_ratemap.py \
        MultiplePFAnalysis/results/stacks_with_ratemaps/R2470/exp_scales_a/grid_r08_c01_x4p38_y93p75.png

    python MultiplePFAnalysis/1_place_cell_stacks/plot_average_stack_ratemap.py \
        grid_r08_c01_x4p38_y93p75.png --rat-id R2470 --experiment-id exp_scales_a

The output is written to the repository root by default.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_place_cell_stacks as stacks  # noqa: E402


STACK_FILENAME_RE = re.compile(r"grid_r(?P<row>\d+)_c(?P<column>\d+)_x[^_]+_y[^.]+\.png$")


def parse_stack_filename(path: Path) -> tuple[int, int]:
    match = STACK_FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Could not parse row/column from stack filename: {path.name}")
    return int(match.group("row")) - 1, int(match.group("column")) - 1


def context_from_path(path: Path) -> tuple[str | None, str | None]:
    """Infer rat and experiment from an existing stack output path."""
    parent = path.parent
    if parent.name == "grid_ratemaps_no_ticks":
        experiment_dir = parent.parent
        rat_dir = experiment_dir.parent
    else:
        experiment_dir = parent
        rat_dir = experiment_dir.parent

    if rat_dir.name and experiment_dir.name.startswith("exp_scales_"):
        return rat_dir.name, experiment_dir.name
    return None, None


def find_stack_context(
    stack_figure: Path,
    results_dir: Path,
    rat_id: str | None,
    experiment_id: str | None,
) -> tuple[str, str]:
    if rat_id and experiment_id:
        return rat_id, experiment_id

    if stack_figure.exists():
        inferred_rat, inferred_experiment = context_from_path(stack_figure.resolve())
        rat_id = rat_id or inferred_rat
        experiment_id = experiment_id or inferred_experiment
        if rat_id and experiment_id:
            return rat_id, experiment_id

    matches = []
    for path in results_dir.glob(f"*/*/{stack_figure.name}"):
        inferred_rat, inferred_experiment = context_from_path(path)
        if inferred_rat and inferred_experiment:
            matches.append((inferred_rat, inferred_experiment, path))
    for path in results_dir.glob(f"*/*/grid_ratemaps_no_ticks/{stack_figure.name}"):
        inferred_rat, inferred_experiment = context_from_path(path)
        if inferred_rat and inferred_experiment:
            matches.append((inferred_rat, inferred_experiment, path))

    filtered = [
        (match_rat, match_experiment, path)
        for match_rat, match_experiment, path in matches
        if (rat_id is None or rat_id == match_rat)
        and (experiment_id is None or experiment_id == match_experiment)
    ]
    unique_contexts = sorted({(match_rat, match_experiment) for match_rat, match_experiment, _ in filtered})

    if len(unique_contexts) == 1:
        return unique_contexts[0]
    if not unique_contexts:
        raise ValueError(
            "Could not infer rat/environment. Pass --rat-id and --experiment-id, "
            "or provide the full path to an existing stack PNG."
        )

    options = "\n".join(f"  --rat-id {match_rat} --experiment-id {match_experiment}" for match_rat, match_experiment in unique_contexts)
    raise ValueError(
        f"Stack filename is ambiguous; it exists for {len(unique_contexts)} rat/environment combinations:\n{options}"
    )


def threshold_from_summary_csv(results_dir: Path, rat_id: str, experiment_id: str) -> float | None:
    csv_path = results_dir / "overall" / "linearized_stack_activity_all_rats.csv"
    if not csv_path.is_file():
        return None
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("rat_id") == rat_id and row.get("experiment_id") == experiment_id:
                try:
                    return float(row["threshold_hz"])
                except (KeyError, TypeError, ValueError):
                    return None
    return None


def plot_average_ratemap(
    mean_ratemap: np.ndarray,
    spatial_window: np.ndarray,
    x_position: float,
    y_position: float,
    output_path: Path,
    title: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.6), constrained_layout=True)
    image = ax.imshow(
        mean_ratemap,
        origin="lower",
        cmap=stacks.RATE_CMAP,
        extent=spatial_window,
        aspect="equal",
    )
    ax.plot(
        x_position,
        y_position,
        marker="*",
        markersize=13,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.9,
    )
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label("Mean firing rate (Hz)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_average_ratemap_clean(
    mean_ratemap: np.ndarray,
    spatial_window: np.ndarray,
    x_position: float,
    y_position: float,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.6), constrained_layout=True)
    ax.imshow(
        mean_ratemap,
        origin="lower",
        cmap=stacks.RATE_CMAP,
        extent=spatial_window,
        aspect="equal",
    )
    ax.plot(
        x_position,
        y_position,
        marker="*",
        markersize=13,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.9,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)


def clean_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_no_ticks_text{output_path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "stack_figure",
        type=Path,
        help="Stack PNG filename or full path, e.g. grid_r08_c01_x4p38_y93p75.png.",
    )
    parser.add_argument("--rat-id", help="Rat ID, required when stack_figure is only an ambiguous filename.")
    parser.add_argument("--experiment-id", help="Experiment/environment ID, e.g. exp_scales_a.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=stacks.REPO_ROOT / "MultiplePFAnalysis" / "cache" / "preprocessed",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=stacks.REPO_ROOT / "MultiplePFAnalysis" / "results" / "stacks_with_ratemaps",
    )
    parser.add_argument("--grid-rows", type=int, default=10)
    parser.add_argument("--grid-columns", type=int, default=10)
    parser.add_argument(
        "--threshold-hz",
        type=float,
        default=None,
        help=(
            "Exact active-cell threshold. If omitted, the script uses the threshold "
            "from stacks_with_ratemaps/overall/linearized_stack_activity_all_rats.csv "
            "when available; otherwise it falls back to the stack-analysis defaults."
        ),
    )
    parser.add_argument("--threshold-fraction", type=float, default=1.0)
    parser.add_argument("--min-threshold-hz", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to the repository root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stack_path = args.stack_figure.expanduser()
    row_index, column_index = parse_stack_filename(stack_path)
    if not (0 <= row_index < args.grid_rows and 0 <= column_index < args.grid_columns):
        raise ValueError(
            f"Stack row/column out of range for {args.grid_rows}x{args.grid_columns}: "
            f"row={row_index + 1}, column={column_index + 1}"
        )

    rat_id, experiment_id = find_stack_context(
        stack_path,
        args.results_dir.expanduser(),
        args.rat_id,
        args.experiment_id,
    )

    threshold_hz = args.threshold_hz
    if threshold_hz is None:
        threshold_hz = threshold_from_summary_csv(args.results_dir.expanduser(), rat_id, experiment_id)

    if threshold_hz is None:
        threshold_fraction = args.threshold_fraction
        min_threshold_hz = args.min_threshold_hz
    else:
        threshold_fraction = 0.0
        min_threshold_hz = threshold_hz

    experiments = stacks.load_cached_experiments(args.cache_dir, rat_id)
    experiment = stacks.select_experiments(experiments, [experiment_id])[0]
    result = stacks.build_environment_result(
        experiment,
        n_rows=args.grid_rows,
        n_columns=args.grid_columns,
        threshold_fraction=threshold_fraction,
        min_threshold_hz=min_threshold_hz,
        show_progress=False,
    )

    cells_in_stack = result["active_cells_per_stack"][row_index][column_index]
    if not cells_in_stack:
        raise ValueError(
            f"No active cells in {rat_id} {experiment_id} row {row_index + 1}, column {column_index + 1} "
            f"at threshold {result['threshold_hz']:.3f} Hz"
        )

    ratemaps = np.stack(
        [result["cell_rows"][cell_info["cell_index"]]["ratemap"] for cell_info in cells_in_stack],
        axis=0,
    )
    mean_ratemap = np.nanmean(ratemaps, axis=0)
    x_position = float(result["x_centers"][column_index])
    y_position = float(result["y_centers"][row_index])

    output_path = args.output
    if output_path is None:
        output_path = (
            stacks.REPO_ROOT
            / f"average_ratemap_{rat_id}_{experiment_id}_r{row_index + 1:02d}_c{column_index + 1:02d}.png"
        )
    else:
        output_path = output_path.expanduser()
        if not output_path.is_absolute():
            output_path = stacks.REPO_ROOT / output_path

    title = (
        f"{rat_id} {result['arena_label']} row {row_index + 1}, column {column_index + 1}: "
        f"mean of {len(cells_in_stack)} active cells\n"
        f"stack position ({x_position:.2f}, {y_position:.2f}) cm, threshold {result['threshold_hz']:.3f} Hz"
    )
    plot_average_ratemap(
        mean_ratemap,
        result["spatial_window"],
        x_position,
        y_position,
        output_path,
        title,
        args.dpi,
    )
    clean_path = clean_output_path(output_path)
    plot_average_ratemap_clean(
        mean_ratemap,
        result["spatial_window"],
        x_position,
        y_position,
        clean_path,
        args.dpi,
    )

    print(f"Wrote {output_path}")
    print(f"Wrote {clean_path}")
    print(f"Included {len(cells_in_stack)} active cells")


if __name__ == "__main__":
    main()
