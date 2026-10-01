#!/usr/bin/env python
"""Generate place-cell stack analysis figures for one or more rats.

The script loads cached experiments one rat at a time, samples every place-cell
ratemap at 100 equally spaced bin centers per environment, and writes figures
under ``<output>/<rat>/<environment>``:

- the place-cell ratemap correlation matrix
- the active-cell count heatmap
- the grid positions over the recorded trajectory
- up to 100 ratemap stack plots

It also writes cross-rat plots under ``<output>/overall``. These show active
cell counts and active-place-cell fractions over row-major linearized grid
bins, with one line per rat and one subplot per environment.
"""

from __future__ import annotations
import matplotlib 
import argparse
import csv
from copy import copy
import gc
from pathlib import Path
import sys
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm


DEFAULT_EXPERIMENT_IDS = (
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

RATE_CMAP = copy(plt.cm.jet)
RATE_CMAP.set_bad("white")


def find_repo_root(start: Optional[Path] = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for path in (start.resolve(), *start.resolve().parents):
        if (path / "MultiplePFAnalysis" / "__init__.py").is_file():
            return path
    raise RuntimeError("Could not find repository root containing MultiplePFAnalysis")


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Pickle loading needs the dataclass definitions registered under this package.
import MultiplePFAnalysis  # noqa: E402,F401


def experiment_label(experiment_id: str) -> str:
    return EXPERIMENT_LABELS.get(experiment_id, experiment_id)


def load_cached_experiments(cache_dir: Path, rat_id: str) -> list:
    cache_file = Path(cache_dir).expanduser() / f"{rat_id}_experiments.npy"
    if not cache_file.is_file():
        raise FileNotFoundError(f"Cached experiments not found: {cache_file}")
    return np.load(cache_file, allow_pickle=True).tolist()


def discover_cached_rat_ids(cache_dir: Path) -> list[str]:
    rat_ids = []
    for cache_file in sorted(Path(cache_dir).expanduser().glob("*_experiments.npy")):
        rat_ids.append(cache_file.name[: -len("_experiments.npy")])
    if not rat_ids:
        raise FileNotFoundError(
            f"No *_experiments.npy files found in {Path(cache_dir).expanduser()}"
        )
    return rat_ids


def select_experiments(experiments: list, experiment_ids: Sequence[str]) -> list:
    lookup = {
        experiment.info.get("experiment_id"): experiment
        for experiment in experiments
    }
    missing = [experiment_id for experiment_id in experiment_ids if experiment_id not in lookup]
    if missing:
        raise KeyError(f"Missing requested environments: {', '.join(missing)}")
    return [lookup[experiment_id] for experiment_id in experiment_ids]


def ratemap_extent(spatial: dict) -> np.ndarray:
    spatial_window = np.asarray(spatial["spatial_window"], dtype=float)
    if spatial_window.shape != (4,):
        raise ValueError(f"Expected spatial_window with four values, got {spatial_window}")
    return spatial_window


def place_cell_rows(experiment) -> list[dict]:
    rows = []
    for unit_index, unit in enumerate(experiment.units):
        if unit.analysis.get("category") != "place_cell":
            continue
        spatial = unit.analysis.get("spatial_ratemaps")
        if not spatial or "spike_rates_smoothed" not in spatial:
            continue
        ratemap = np.asarray(spatial["spike_rates_smoothed"], dtype=float)
        if ratemap.ndim != 2 or not np.any(np.isfinite(ratemap)):
            continue
        if np.nanmax(ratemap) <= 0:
            continue
        rows.append(
            {
                "unit_index": unit_index,
                "unit": unit,
                "label": f"T{unit.tetrode_nr}C{unit.cluster_id}",
                "ratemap": ratemap,
                "spatial": spatial,
            }
        )
    return rows


def population_mean_rate(cell_rows: list[dict]) -> float:
    finite_values = [
        row["ratemap"][np.isfinite(row["ratemap"])]
        for row in cell_rows
        if np.any(np.isfinite(row["ratemap"]))
    ]
    if not finite_values:
        return np.nan
    return float(np.mean(np.concatenate(finite_values)))


def equal_grid(spatial_window, n_rows: int = 10, n_columns: int = 10) -> dict:
    x_min, x_max, y_min, y_max = np.asarray(spatial_window, dtype=float)
    if n_rows <= 0 or n_columns <= 0:
        raise ValueError("Grid dimensions must be positive")
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Invalid spatial window: {spatial_window}")

    x_edges = np.linspace(x_min, x_max, n_columns + 1)
    y_edges = np.linspace(y_min, y_max, n_rows + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    xx, yy = np.meshgrid(x_centers, y_centers)
    return {
        "x_edges": x_edges,
        "y_edges": y_edges,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "xx": xx,
        "yy": yy,
    }


def ratemap_bin_centers(ratemap: np.ndarray, spatial_window) -> tuple[np.ndarray, np.ndarray]:
    x_min, x_max, y_min, y_max = np.asarray(spatial_window, dtype=float)
    n_y, n_x = ratemap.shape
    x_edges = np.linspace(x_min, x_max, n_x + 1)
    y_edges = np.linspace(y_min, y_max, n_y + 1)
    return (x_edges[:-1] + x_edges[1:]) / 2.0, (y_edges[:-1] + y_edges[1:]) / 2.0


def interpolate_ratemap(
    ratemap: np.ndarray,
    spatial_window,
    x_positions: np.ndarray,
    y_positions: np.ndarray,
) -> np.ndarray:
    """Interpolate finite ratemap values without treating invalid bins as firing."""
    orig_x, orig_y = ratemap_bin_centers(ratemap, spatial_window)
    valid = np.isfinite(ratemap)
    values = np.where(valid, ratemap, 0.0)

    value_interpolator = RegularGridInterpolator(
        (orig_y, orig_x),
        values,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    weight_interpolator = RegularGridInterpolator(
        (orig_y, orig_x),
        valid.astype(float),
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    xx, yy = np.meshgrid(x_positions, y_positions)
    points = np.column_stack((yy.ravel(), xx.ravel()))
    interpolated_values = value_interpolator(points)
    interpolated_weights = weight_interpolator(points)

    output = np.full(interpolated_values.shape, np.nan, dtype=float)
    usable = interpolated_weights > 0.5
    output[usable] = interpolated_values[usable] / interpolated_weights[usable]
    return output.reshape(len(y_positions), len(x_positions))


def compute_ratemap_correlation_matrix(
    cell_rows: list[dict],
    *,
    description: str = "Correlation matrix",
    show_progress: bool = True,
) -> np.ndarray:
    n_cells = len(cell_rows)
    matrix = np.full((n_cells, n_cells), np.nan, dtype=float)
    row_indices = tqdm(
        range(n_cells),
        desc=description,
        unit="cell",
        leave=False,
        disable=not show_progress,
    )
    for row_index in row_indices:
        ratemap_1 = cell_rows[row_index]["ratemap"].ravel()
        for column_index in range(row_index, n_cells):
            ratemap_2 = cell_rows[column_index]["ratemap"].ravel()
            if ratemap_1.shape != ratemap_2.shape:
                continue
            valid = np.isfinite(ratemap_1) & np.isfinite(ratemap_2)
            if np.sum(valid) < 4:
                continue
            values_1 = ratemap_1[valid]
            values_2 = ratemap_2[valid]
            if np.std(values_1) == 0 or np.std(values_2) == 0:
                continue
            correlation = float(np.corrcoef(values_1, values_2)[0, 1])
            matrix[row_index, column_index] = correlation
            matrix[column_index, row_index] = correlation
    return matrix


def build_environment_result(
    experiment,
    *,
    n_rows: int = 10,
    n_columns: int = 10,
    threshold_fraction: float = 1.0,
    min_threshold_hz: float = 1.0,
    show_progress: bool = True,
) -> dict:
    cell_rows = place_cell_rows(experiment)
    if not cell_rows:
        raise ValueError(f"No usable place cells in {experiment.info.get('experiment_id')}")

    windows = {tuple(ratemap_extent(row["spatial"])) for row in cell_rows}
    if len(windows) != 1:
        raise ValueError(f"Place-cell ratemaps use inconsistent spatial windows: {windows}")
    spatial_window = np.asarray(next(iter(windows)), dtype=float)
    grid = equal_grid(spatial_window, n_rows=n_rows, n_columns=n_columns)
    experiment_id = experiment.info.get("experiment_id", experiment.session)
    arena_label = experiment_label(experiment_id)

    cell_stack = np.stack(
        [
            interpolate_ratemap(
                row["ratemap"],
                spatial_window,
                grid["x_centers"],
                grid["y_centers"],
            )
            for row in tqdm(
                cell_rows,
                desc=f"{arena_label}: interpolate ratemaps",
                unit="cell",
                leave=False,
                disable=not show_progress,
            )
        ],
        axis=0,
    )
    mean_rate_hz = population_mean_rate(cell_rows)
    population_threshold_hz = threshold_fraction * mean_rate_hz
    threshold_hz = max(population_threshold_hz, min_threshold_hz)
    active_mask = np.isfinite(cell_stack) & (cell_stack >= threshold_hz)
    active_counts = np.sum(active_mask, axis=0)

    active_cells_per_stack = []
    for row_index in range(n_rows):
        row_stacks = []
        for column_index in range(n_columns):
            cell_indices = np.flatnonzero(active_mask[:, row_index, column_index])
            stack = [
                {
                    "cell_index": int(cell_index),
                    "rate": float(cell_stack[cell_index, row_index, column_index]),
                }
                for cell_index in cell_indices
            ]
            stack.sort(key=lambda item: item["rate"], reverse=True)
            row_stacks.append(stack)
        active_cells_per_stack.append(row_stacks)

    return {
        "experiment": experiment,
        "experiment_id": experiment_id,
        "arena_label": arena_label,
        "cell_rows": cell_rows,
        "cell_stack": cell_stack,
        "active_counts": active_counts,
        "active_cells_per_stack": active_cells_per_stack,
        "population_mean_hz": mean_rate_hz,
        "population_threshold_hz": population_threshold_hz,
        "threshold_hz": threshold_hz,
        "threshold_fraction": threshold_fraction,
        "min_threshold_hz": min_threshold_hz,
        "spatial_window": spatial_window,
        **grid,
    }


def plot_correlation_matrix(
    result: dict,
    rat_id: str,
    *,
    show_progress: bool = True,
):
    matrix = compute_ratemap_correlation_matrix(
        result["cell_rows"],
        description=f"{result['arena_label']}: correlation rows",
        show_progress=show_progress,
    )
    labels = [row["label"] for row in result["cell_rows"]]

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(
        matrix,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        interpolation="nearest",
    )
    tick_step = max(1, len(labels) // 22)
    tick_positions = np.arange(0, len(labels), tick_step)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(
        [labels[index] for index in tick_positions],
        rotation=90,
        fontsize=6,
    )
    ax.set_yticklabels(
        [labels[index] for index in tick_positions],
        fontsize=6,
    )
    ax.set_title(
        f"{rat_id} environment {result['arena_label']}: "
        f"place-cell ratemap correlations\n"
        f"n={len(labels)} cells"
    )
    fig.colorbar(image, ax=ax, label="Pearson r", shrink=0.82)
    fig.tight_layout()
    return fig


def plot_active_cell_count_heatmap(result: dict, rat_id: str):
    active_counts = result["active_counts"]
    x_min, x_max, y_min, y_max = result["spatial_window"]

    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(
        active_counts,
        origin="lower",
        extent=(x_min, x_max, y_min, y_max),
        aspect="equal",
        cmap="viridis",
        interpolation="nearest",
    )

    midpoint = (np.nanmin(active_counts) + np.nanmax(active_counts)) / 2.0
    for row_index, y_position in enumerate(result["y_centers"]):
        for column_index, x_position in enumerate(result["x_centers"]):
            count = int(active_counts[row_index, column_index])
            ax.text(
                x_position,
                y_position,
                str(count),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if count <= midpoint else "black",
            )

    ax.set_xticks(result["x_edges"])
    ax.set_yticks(result["y_edges"])
    ax.grid(color="white", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title(
        f"{rat_id} environment {result['arena_label']}: "
        f"active place cells on "
        f"{len(result['x_centers'])} x {len(result['y_centers'])} grid\n"
        f"threshold={result['threshold_hz']:.3f} Hz"
    )
    fig.colorbar(image, ax=ax, label="Number of active place cells", shrink=0.82)
    fig.tight_layout()
    return fig


def plot_grid_positions(result: dict):
    """Create a grid-position validation figure."""
    experiment = result["experiment"]
    x_min, x_max, y_min, y_max = result["spatial_window"]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    if experiment.position is not None:
        xy = np.asarray(experiment.position.xy, dtype=float)
        valid = np.all(np.isfinite(xy), axis=1)
        xy = xy[valid]
        if len(xy):
            stride = max(1, len(xy) // 25000)
            ax.plot(
                xy[::stride, 0],
                xy[::stride, 1],
                color="0.78",
                linewidth=0.35,
                alpha=0.55,
                label="Position trace",
                zorder=1,
            )

    for x_value in result["x_edges"]:
        ax.axvline(x_value, color="0.65", linewidth=0.7, zorder=2)
    for y_value in result["y_edges"]:
        ax.axhline(y_value, color="0.65", linewidth=0.7, zorder=2)

    ax.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
            zorder=3,
        )
    )
    ax.scatter(
        result["xx"].ravel(),
        result["yy"].ravel(),
        s=18,
        color="#d62728",
        edgecolor="white",
        linewidth=0.3,
        label="Stack positions",
        zorder=4,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title(
        f"{result['arena_label']} ({result['experiment_id']}): "
        f"{len(result['x_centers'])} x {len(result['y_centers'])} stack grid"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def save_figure(fig, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def filename_coordinate(value: float) -> str:
    return f"{value:.2f}".replace("-", "m").replace(".", "p")


def plot_stack_similarity_matrix(
    axis,
    result: dict,
    cells_in_stack: list[dict],
) -> None:
    cell_rows = [
        result["cell_rows"][cell_info["cell_index"]]
        for cell_info in cells_in_stack
    ]
    labels = [f"#{index + 1}" for index in range(len(cell_rows))]
    matrix = compute_ratemap_correlation_matrix(
        cell_rows,
        show_progress=False,
    )

    image = axis.imshow(
        matrix,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        interpolation="nearest",
    )
    tick_positions = np.arange(len(labels))
    tick_fontsize = max(8, min(12, 130 // max(1, len(labels))))
    axis.set_xticks(tick_positions)
    axis.set_yticks(tick_positions)
    axis.set_xticklabels(labels, rotation=90, fontsize=tick_fontsize)
    axis.set_yticklabels(labels, fontsize=tick_fontsize)
    axis.set_title("Ratemap similarity within stack\n(Pearson r)", fontsize=11)
    divider = make_axes_locatable(axis)
    colorbar_axis = divider.append_axes("right", size="5%", pad=0.08)
    colorbar = axis.figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Pearson r", fontsize=10)
    colorbar.ax.tick_params(labelsize=10)


def plot_stack(
    result: dict,
    row_index: int,
    column_index: int,
    output_path: Path,
    dpi: int = 100,
    simplified: bool = False,
) -> None:
    cells_in_stack = result["active_cells_per_stack"][row_index][column_index]
    if not cells_in_stack:
        return

    x_position = result["x_centers"][column_index]
    y_position = result["y_centers"][row_index]
    n_cells = len(cells_in_stack)
    figure_height = max(4.5, 2.15 * n_cells)
    similarity_width = max(4.8, min(8.0, 0.34 * n_cells + 3.5))
    fig = plt.figure(
        figsize=(3.4 + similarity_width, figure_height),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        n_cells,
        2,
        width_ratios=(3.4, similarity_width),
        wspace=0.35,
    )
    ratemap_axes = [fig.add_subplot(grid[row_index_in_figure, 0]) for row_index_in_figure in range(n_cells)]
    similarity_axis = fig.add_subplot(grid[:, 1])

    for stack_cell_number, (axis, cell_info) in enumerate(zip(ratemap_axes, cells_in_stack), start=1):
        cell_row = result["cell_rows"][cell_info["cell_index"]]
        ratemap = cell_row["ratemap"]
        spatial_window = ratemap_extent(cell_row["spatial"])
        axis.imshow(
            ratemap,
            origin="lower",
            cmap=RATE_CMAP,
            extent=spatial_window,
            aspect="equal",
        )
        axis.plot(
            x_position,
            y_position,
            marker="*",
            markersize=8,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.7,
        )
        axis.set_title(
            f"#{stack_cell_number} | max {np.nanmax(ratemap):.2f} Hz | "
            f"active {cell_info['rate']:.2f} Hz",
            fontsize=11 if simplified else 9,
        )
        if simplified:
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xlabel("")
            axis.set_ylabel("")
        else:
            axis.set_xticks([spatial_window[0], x_position, spatial_window[1]])
            axis.set_yticks([spatial_window[2], y_position, spatial_window[3]])
            axis.tick_params(labelsize=7)
            axis.set_xlabel("x (cm)", fontsize=7)
            axis.set_ylabel("y (cm)", fontsize=7)

    plot_stack_similarity_matrix(similarity_axis, result, cells_in_stack)

    if not simplified:
        fig.suptitle(
            f"{result['arena_label']} | row {row_index + 1}, column {column_index + 1} | "
            f"({x_position:.2f}, {y_position:.2f}) cm | {n_cells} active cells\n"
            f"threshold {result['threshold_hz']:.3f} Hz",
            fontsize=9,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_environment_outputs(
    result: dict,
    output_dir: Path,
    *,
    rat_id: str,
    dpi: int = 100,
    summary_dpi: int = 180,
    max_stack_plots: int = 100,
    dry_run: bool = False,
    show_progress: bool = True,
) -> int:
    experiment_id = result["experiment_id"]
    environment_dir = output_dir / experiment_id
    summary_figures = (
        (
            "correlation_matrix.png",
            lambda: plot_correlation_matrix(
                result,
                rat_id,
                show_progress=show_progress,
            ),
        ),
        (
            "active_cell_count_heatmap.png",
            lambda: plot_active_cell_count_heatmap(result, rat_id),
        ),
        (
            "grid_positions.png",
            lambda: plot_grid_positions(result),
        ),
    )

    summary_iterator = tqdm(
        summary_figures,
        desc=f"{result['arena_label']}: summary figures",
        unit="figure",
        leave=False,
        disable=not show_progress,
    )
    for filename, create_figure in summary_iterator:
        summary_iterator.set_postfix_str(filename, refresh=False)
        output_path = environment_dir / filename
        if dry_run:
            tqdm.write(f"[dry-run] summary: {output_path}")
        else:
            save_figure(create_figure(), output_path, dpi=summary_dpi)

    saved_count = 0
    grid_positions = [
        (row_index, column_index, x_position, y_position)
        for row_index, y_position in enumerate(result["y_centers"])
        for column_index, x_position in enumerate(result["x_centers"])
    ]
    stack_iterator = tqdm(
        grid_positions,
        desc=f"{result['arena_label']}: stack plots",
        unit="position",
        leave=False,
        disable=not show_progress,
    )
    for row_index, column_index, x_position, y_position in stack_iterator:
        if saved_count >= max_stack_plots:
            break
        cells_in_stack = result["active_cells_per_stack"][row_index][column_index]
        stack_iterator.set_postfix(
            saved=saved_count,
            active_cells=len(cells_in_stack),
            refresh=False,
        )
        if not cells_in_stack:
            continue
        filename = (
            f"grid_r{row_index + 1:02d}_c{column_index + 1:02d}_"
            f"x{filename_coordinate(x_position)}_y{filename_coordinate(y_position)}.png"
        )
        output_path = environment_dir / filename
        simplified_output_path = environment_dir / "grid_ratemaps_no_ticks" / filename
        if dry_run:
            tqdm.write(f"[dry-run] stack: {output_path}")
            tqdm.write(f"[dry-run] simplified stack: {simplified_output_path}")
        else:
            plot_stack(result, row_index, column_index, output_path, dpi=dpi)
            plot_stack(
                result,
                row_index,
                column_index,
                simplified_output_path,
                dpi=dpi,
                simplified=True,
            )
        saved_count += 1
        stack_iterator.set_postfix(
            saved=saved_count,
            active_cells=len(cells_in_stack),
            refresh=False,
        )
    return saved_count


def compact_linearized_summary(result: dict, rat_id: str) -> dict:
    active_counts = np.asarray(result["active_counts"], dtype=int)
    n_place_cells = len(result["cell_rows"])
    active_fractions = active_counts.astype(float) / n_place_cells
    return {
        "rat_id": rat_id,
        "experiment_id": result["experiment_id"],
        "arena_label": result["arena_label"],
        "active_counts": active_counts.ravel(order="C"),
        "active_fractions": active_fractions.ravel(order="C"),
        "n_place_cells": n_place_cells,
        "threshold_hz": result["threshold_hz"],
        "n_rows": active_counts.shape[0],
        "n_columns": active_counts.shape[1],
        "x_centers": np.asarray(result["x_centers"], dtype=float),
        "y_centers": np.asarray(result["y_centers"], dtype=float),
    }


def plot_all_rats_linearized(
    summaries: list[dict],
    experiment_ids: Sequence[str],
    *,
    value_key: str,
    ylabel: str,
    title: str,
):
    n_environments = len(experiment_ids)
    n_columns = min(2, n_environments)
    n_rows = int(np.ceil(n_environments / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(16, 4.8 * n_rows),
        squeeze=False,
        sharex=True,
    )
    flat_axes = axes.ravel()

    for axis_index, experiment_id in enumerate(experiment_ids):
        axis = flat_axes[axis_index]
        environment_summaries = [
            summary
            for summary in summaries
            if summary["experiment_id"] == experiment_id
        ]
        for summary in environment_summaries:
            values = np.asarray(summary[value_key], dtype=float)
            linearized_bins = np.arange(1, len(values) + 1)
            axis.plot(
                linearized_bins,
                values,
                linewidth=1.4,
                alpha=0.9,
                label=summary["rat_id"],
            )

        if environment_summaries:
            n_grid_columns = environment_summaries[0]["n_columns"]
            n_linearized_bins = len(environment_summaries[0][value_key])
            for boundary in range(n_grid_columns, n_linearized_bins, n_grid_columns):
                axis.axvline(boundary + 0.5, color="0.85", linewidth=0.6, zorder=0)
            axis.set_xlim(1, n_linearized_bins)

        axis.set_title(f"Environment {experiment_label(experiment_id)}")
        axis.set_xlabel("Linearized grid bin (row-major: r01c01 to r10c10)")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="0.9", linewidth=0.7)
        axis.legend(fontsize=8, ncol=2)

    for axis in flat_axes[n_environments:]:
        axis.set_visible(False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def write_linearized_summary_csv(summaries: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rat_id",
        "experiment_id",
        "arena",
        "linearized_bin",
        "grid_row",
        "grid_column",
        "x_cm",
        "y_cm",
        "active_cell_count",
        "active_place_cell_fraction",
        "n_place_cells",
        "threshold_hz",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            n_columns = summary["n_columns"]
            for flat_index, (count, fraction) in enumerate(
                zip(summary["active_counts"], summary["active_fractions"])
            ):
                row_index, column_index = divmod(flat_index, n_columns)
                writer.writerow(
                    {
                        "rat_id": summary["rat_id"],
                        "experiment_id": summary["experiment_id"],
                        "arena": summary["arena_label"],
                        "linearized_bin": flat_index + 1,
                        "grid_row": row_index + 1,
                        "grid_column": column_index + 1,
                        "x_cm": summary["x_centers"][column_index],
                        "y_cm": summary["y_centers"][row_index],
                        "active_cell_count": int(count),
                        "active_place_cell_fraction": float(fraction),
                        "n_place_cells": summary["n_place_cells"],
                        "threshold_hz": summary["threshold_hz"],
                    }
                )


def save_all_rats_outputs(
    summaries: list[dict],
    experiment_ids: Sequence[str],
    output_dir: Path,
    *,
    summary_dpi: int,
    dry_run: bool,
) -> None:
    overall_dir = output_dir / "overall"
    outputs = (
        (
            "active_cell_counts_by_linearized_bin.png",
            lambda: plot_all_rats_linearized(
                summaries,
                experiment_ids,
                value_key="active_counts",
                ylabel="Number of active place cells",
                title="Active place cells per linearized stack bin",
            ),
        ),
        (
            "active_place_cell_fractions_by_linearized_bin.png",
            lambda: plot_all_rats_linearized(
                summaries,
                experiment_ids,
                value_key="active_fractions",
                ylabel="Fraction of usable place cells active",
                title="Fraction of place cells active per linearized stack bin",
            ),
        ),
    )
    for filename, create_figure in outputs:
        output_path = overall_dir / filename
        if dry_run:
            tqdm.write(f"[dry-run] overall: {output_path}")
        else:
            save_figure(create_figure(), output_path, dpi=summary_dpi)

    csv_path = overall_dir / "linearized_stack_activity_all_rats.csv"
    if dry_run:
        tqdm.write(f"[dry-run] overall: {csv_path}")
    else:
        write_linearized_summary_csv(summaries, csv_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    rat_group = parser.add_mutually_exclusive_group()
    rat_group.add_argument(
        "--rat-id",
        help="Analyze one rat. Retained as a convenience alias.",
    )
    rat_group.add_argument(
        "--rat-ids",
        nargs="+",
        help=(
            "Rat IDs to analyze. By default all *_experiments.npy caches in "
            "--cache-dir are analyzed."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "MultiplePFAnalysis" / "cache" / "preprocessed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "MultiplePFAnalysis" / "results" / "stacks_with_ratemaps",
    )
    parser.add_argument(
        "--experiment-ids",
        nargs="+",
        default=list(DEFAULT_EXPERIMENT_IDS),
        help="Environment IDs to plot. Defaults to A, B, C, and D.",
    )
    parser.add_argument("--grid-rows", type=int, default=10)
    parser.add_argument("--grid-columns", type=int, default=10)
    parser.add_argument(
        "--threshold-fraction",
        type=float,
        default=1.0,
        help="Active-cell threshold as a fraction of the population mean firing rate.",
    )
    parser.add_argument(
        "--min-threshold-hz",
        type=float,
        default=1.0,
        help=(
            "Minimum firing rate for inclusion in a stack. The effective "
            "threshold is the larger of this value and the population-based threshold."
        ),
    )
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument(
        "--summary-dpi",
        type=int,
        default=180,
        help="Resolution for correlation, active-cell, and grid-position figures.",
    )
    parser.add_argument(
        "--max-stack-plots",
        type=int,
        default=100,
        help="Maximum non-empty stack plots per environment; the 10 x 10 grid caps this at 100.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_stack_plots < 0:
        raise ValueError("--max-stack-plots must be non-negative")
    if args.threshold_fraction < 0:
        raise ValueError("--threshold-fraction must be non-negative")
    if args.min_threshold_hz < 0:
        raise ValueError("--min-threshold-hz must be non-negative")

    show_progress = not args.no_progress
    if args.rat_id:
        rat_ids = [args.rat_id]
    elif args.rat_ids:
        rat_ids = args.rat_ids
    else:
        rat_ids = discover_cached_rat_ids(args.cache_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tqdm.write(
        f"Generating figures for {len(rat_ids)} rats x "
        f"{len(args.experiment_ids)} environments in "
        f"{args.output_dir}"
    )
    all_summaries = []
    rat_iterator = tqdm(
        rat_ids,
        desc="Rats",
        unit="rat",
        disable=not show_progress,
    )
    for rat_id in rat_iterator:
        rat_iterator.set_postfix_str(rat_id, refresh=True)
        tqdm.write(f"Loading cached experiments for {rat_id}...")
        experiments = load_cached_experiments(args.cache_dir, rat_id)
        selected_experiments = select_experiments(experiments, args.experiment_ids)
        rat_output_dir = args.output_dir / rat_id

        environment_iterator = tqdm(
            selected_experiments,
            desc=f"{rat_id}: environments",
            unit="environment",
            leave=False,
            disable=not show_progress,
        )
        for experiment in environment_iterator:
            experiment_id = experiment.info.get("experiment_id", experiment.session)
            arena_label = experiment_label(experiment_id)
            environment_iterator.set_postfix_str(
                f"{arena_label} ({experiment_id})",
                refresh=True,
            )
            result = build_environment_result(
                experiment,
                n_rows=args.grid_rows,
                n_columns=args.grid_columns,
                threshold_fraction=args.threshold_fraction,
                min_threshold_hz=args.min_threshold_hz,
                show_progress=show_progress,
            )
            all_summaries.append(compact_linearized_summary(result, rat_id))
            saved_count = save_environment_outputs(
                result,
                rat_output_dir,
                rat_id=rat_id,
                dpi=args.dpi,
                summary_dpi=args.summary_dpi,
                max_stack_plots=min(
                    args.max_stack_plots,
                    args.grid_rows * args.grid_columns,
                ),
                dry_run=args.dry_run,
                show_progress=show_progress,
            )
            tqdm.write(
                f"{rat_id} {result['experiment_id']}: "
                f"{len(result['cell_rows'])} place cells, "
                f"threshold={result['threshold_hz']:.3f} Hz, "
                f"{saved_count} non-empty stack plots"
            )
            del result

        del selected_experiments
        del experiments
        gc.collect()

    save_all_rats_outputs(
        all_summaries,
        args.experiment_ids,
        args.output_dir,
        summary_dpi=args.summary_dpi,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
