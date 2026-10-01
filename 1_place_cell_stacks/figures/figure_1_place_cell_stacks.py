#!/usr/bin/env python3
"""Create a multi-panel paper figure summarizing place-cell stacks.

The main figure tells the stack-analysis story at three levels:

A. Stack fraction across positions in one example recording.
B. Three example stacks (four highest-rate cells plus the mean of the full stack).
C. Mean ratemap of every stack across positions in the example recording.
D. Rat-level mean stack fraction in each arena (bar, individual rats, mean +/- SD).

A supplementary figure contains the stack fraction across normalized positions,
averaged across rats for arenas A-D, and grids of mean stack ratemaps for all
four environments in the example rat.

Example panels are recomputed from cached experiments. Population panels are
read from the compact CSV written by ``analyze_place_cell_stacks.py``.

Example
-------
python MultiplePFAnalysis/1_place_cell_stacks/figures/figure_1_place_cell_stacks.py

Use raw active-cell counts instead of fractions:

python MultiplePFAnalysis/1_place_cell_stacks/figures/figure_1_place_cell_stacks.py \
    --metric count
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import analyze_place_cell_stacks as stacks  # noqa: E402


DEFAULT_EXPERIMENT_ORDER = (
    "exp_scales_a",
    "exp_scales_b",
    "exp_scales_c",
    "exp_scales_d",
)

ENVIRONMENT_COLORS = {
    "exp_scales_a": "#C8E4FC",
    "exp_scales_b": "#88B888",
    "exp_scales_c": "#CCE0C8",
    "exp_scales_d": "#006498",
}

EXAMPLE_COLORS = ("#D55E00", "#009E73", "#7B3294")

METRIC_COLUMNS = {
    "fraction": "active_place_cell_fraction",
    "count": "active_cell_count",
}

METRIC_LABELS = {
    "fraction": "Active place-cell fraction",
    "count": "Active place cells",
}


def configure_plot_style() -> None:
    """Use compact, journal-friendly Matplotlib defaults."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "svg.fonttype": "none",
        }
    )


def environment_label(experiment_id: str) -> str:
    return stacks.EXPERIMENT_LABELS.get(experiment_id, experiment_id)


def parse_example_bin(text: str) -> tuple[int, int]:
    """Parse a one-based ROW,COLUMN argument and return zero-based indices."""
    try:
        row_text, column_text = text.split(",", maxsplit=1)
        row_index = int(row_text) - 1
        column_index = int(column_text) - 1
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            f"Expected ROW,COLUMN with one-based integers, got {text!r}"
        ) from error
    if row_index < 0 or column_index < 0:
        raise argparse.ArgumentTypeError("Example ROW and COLUMN must both be at least 1")
    return row_index, column_index


def read_summary_rows(
    csv_path: Path,
    *,
    threshold_hz: float,
    experiment_order: Sequence[str],
) -> list[dict]:
    """Read and type-convert the relevant stack summary rows."""
    metric_columns = set(METRIC_COLUMNS.values())
    required = {
        "rat_id",
        "experiment_id",
        "arena",
        "grid_row",
        "grid_column",
        "threshold_hz",
        "n_place_cells",
        *metric_columns,
    }
    rows: list[dict] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{csv_path} is missing columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            if row["experiment_id"] not in experiment_order:
                continue
            if not np.isclose(float(row["threshold_hz"]), threshold_hz):
                continue
            rows.append(
                {
                    **row,
                    "grid_row": int(row["grid_row"]),
                    "grid_column": int(row["grid_column"]),
                    "threshold_hz": float(row["threshold_hz"]),
                    "n_place_cells": int(row["n_place_cells"]),
                    "active_cell_count": float(row["active_cell_count"]),
                    "active_place_cell_fraction": float(
                        row["active_place_cell_fraction"]
                    ),
                }
            )
    if not rows:
        raise ValueError(
            f"No rows at threshold {threshold_hz:g} Hz for the requested "
            f"environments in {csv_path}"
        )
    return rows


def threshold_from_csv(
    csv_path: Path,
    *,
    rat_id: str,
    experiment_id: str,
) -> float:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (
                row.get("rat_id") == rat_id
                and row.get("experiment_id") == experiment_id
            ):
                return float(row["threshold_hz"])
    raise ValueError(
        f"No summary rows found for {rat_id} {experiment_id} in {csv_path}"
    )


def validate_example_bins(
    example_bins: Sequence[tuple[int, int]],
    result: dict,
) -> None:
    n_rows, n_columns = result["active_counts"].shape
    for row_index, column_index in example_bins:
        if not (0 <= row_index < n_rows and 0 <= column_index < n_columns):
            raise ValueError(
                f"Example bin r{row_index + 1},c{column_index + 1} is outside "
                f"the {n_rows} x {n_columns} grid"
            )
        if not result["active_cells_per_stack"][row_index][column_index]:
            raise ValueError(
                f"Example bin r{row_index + 1},c{column_index + 1} has no "
                f"active cells at {result['threshold_hz']:.3g} Hz"
            )


def add_panel_label(axis, label: str, *, x: float = -0.18, y: float = 1.08) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        clip_on=False,
    )


def nanmean_without_warning(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute a NaN-aware mean while preserving all-NaN bins as NaN."""
    finite = np.isfinite(values)
    count = np.sum(finite, axis=axis)
    total = np.nansum(values, axis=axis)
    output = np.full(np.asarray(total).shape, np.nan, dtype=float)
    np.divide(total, count, out=output, where=count > 0)
    return output


def plot_example_spatial_map(
    axis,
    result: dict,
    example_bins: Sequence[tuple[int, int]],
    *,
    metric: str,
) -> None:
    values = result["active_counts"].astype(float)
    if metric == "fraction":
        values /= len(result["cell_rows"])

    image = axis.imshow(
        values,
        origin="lower",
        extent=result["spatial_window"],
        aspect="equal",
        cmap="viridis",
        interpolation="nearest",
    )
    for example_number, ((row_index, column_index), color) in enumerate(
        zip(example_bins, EXAMPLE_COLORS),
        start=1,
    ):
        x_position = result["x_centers"][column_index]
        y_position = result["y_centers"][row_index]
        axis.scatter(
            x_position,
            y_position,
            s=80,
            facecolors="none",
            edgecolors=color,
            linewidths=1.8,
            zorder=3,
        )
        axis.text(
            x_position,
            y_position,
            str(example_number),
            color="white",
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=4,
        )

    axis.set_xlabel("x (cm)")
    axis.set_ylabel("y (cm)")
    axis.set_title(
        f"Example session: {result['arena_label']}\n"
        f"active at $\\geq${result['threshold_hz']:g} Hz"
    )
    colorbar = axis.figure.colorbar(
        image,
        ax=axis,
        fraction=0.042,
        pad=0.025,
    )
    colorbar.set_label("Fraction" if metric == "fraction" else "Count")
    colorbar.outline.set_linewidth(0.6)
    add_panel_label(axis, "A")


def plot_ratemap(
    axis,
    ratemap: np.ndarray,
    spatial_window: np.ndarray,
    *,
    x_position: float,
    y_position: float,
    vmax: float | None = None,
) -> None:
    axis.imshow(
        ratemap,
        origin="lower",
        extent=spatial_window,
        aspect="equal",
        cmap=stacks.RATE_CMAP,
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )
    axis.plot(
        x_position,
        y_position,
        marker="*",
        markersize=4.5,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.45,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(0.5)


def perspective_coordinates(
    shape: tuple[int, int],
    *,
    left: float,
    right: float,
    bottom: float,
    height: float,
    back_inset: float = 0.12,
    rotation_degrees: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return curvilinear cell edges that project a ratemap onto a trapezoid."""
    n_rows, n_columns = shape
    u, v = np.meshgrid(
        np.linspace(0.0, 1.0, n_columns + 1),
        np.linspace(0.0, 1.0, n_rows + 1),
    )
    width = right - left
    x = left + width * (
        back_inset * v + u * (1.0 - 2.0 * back_inset * v)
    )
    y = bottom + height * v
    if rotation_degrees:
        angle = np.deg2rad(rotation_degrees)
        center_x = (left + right) / 2.0
        center_y = bottom + height / 2.0
        x_offset = x - center_x
        y_offset = y - center_y
        x = center_x + x_offset * np.cos(angle) - y_offset * np.sin(angle)
        y = center_y + x_offset * np.sin(angle) + y_offset * np.cos(angle)
    return x, y


def perspective_point(
    u: float,
    v: float,
    *,
    left: float,
    right: float,
    bottom: float,
    height: float,
    back_inset: float = 0.12,
    rotation_degrees: float = 0.0,
) -> tuple[float, float]:
    width = right - left
    x = left + width * (
        back_inset * v + u * (1.0 - 2.0 * back_inset * v)
    )
    y = bottom + height * v
    if rotation_degrees:
        angle = np.deg2rad(rotation_degrees)
        center_x = (left + right) / 2.0
        center_y = bottom + height / 2.0
        x_offset = x - center_x
        y_offset = y - center_y
        x = center_x + x_offset * np.cos(angle) - y_offset * np.sin(angle)
        y = center_y + x_offset * np.sin(angle) + y_offset * np.cos(angle)
    return float(x), float(y)


def plot_tilted_ratemap(
    axis,
    ratemap: np.ndarray,
    spatial_window: np.ndarray,
    *,
    x_position: float,
    y_position: float,
    left: float,
    right: float,
    bottom: float,
    height: float,
    rotation_degrees: float,
    outline_color: str = "0.35",
    outline_width: float = 0.35,
) -> tuple[float, float]:
    """Draw a ratemap as a perspective layer and return the marker position."""
    x_edges, y_edges = perspective_coordinates(
        ratemap.shape,
        left=left,
        right=right,
        bottom=bottom,
        height=height,
        rotation_degrees=rotation_degrees,
    )
    finite_values = ratemap[np.isfinite(ratemap)]
    vmax = float(np.max(finite_values)) if len(finite_values) else 1.0
    axis.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(ratemap),
        cmap=stacks.RATE_CMAP,
        vmin=0.0,
        vmax=vmax if vmax > 0 else 1.0,
        shading="flat",
        rasterized=True,
        zorder=2,
    )
    corners_x = [
        x_edges[0, 0],
        x_edges[0, -1],
        x_edges[-1, -1],
        x_edges[-1, 0],
        x_edges[0, 0],
    ]
    corners_y = [
        y_edges[0, 0],
        y_edges[0, -1],
        y_edges[-1, -1],
        y_edges[-1, 0],
        y_edges[0, 0],
    ]
    axis.plot(
        corners_x,
        corners_y,
        color=outline_color,
        linewidth=outline_width,
        zorder=3,
    )

    x_min, x_max, y_min, y_max = np.asarray(spatial_window, dtype=float)
    u = np.clip((x_position - x_min) / (x_max - x_min), 0.0, 1.0)
    v = np.clip((y_position - y_min) / (y_max - y_min), 0.0, 1.0)
    marker_x, marker_y = perspective_point(
        float(u),
        float(v),
        left=left,
        right=right,
        bottom=bottom,
        height=height,
        rotation_degrees=rotation_degrees,
    )
    axis.plot(
        marker_x,
        marker_y,
        marker="*",
        markersize=4.0,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.45,
        zorder=5,
    )
    return marker_x, marker_y


def plot_example_stacks(
    figure,
    grid_spec,
    result: dict,
    example_bins: Sequence[tuple[int, int]],
    *,
    max_example_cells: int,
    rotation_degrees: float,
) -> None:
    """Plot top active cells as tilted layers and the full-stack mean below."""
    subgrid = grid_spec.subgridspec(
        2,
        len(example_bins),
        height_ratios=(0.13, 1.0),
        wspace=0.14,
        hspace=0.03,
    )
    header_axis = figure.add_subplot(subgrid[0, :])
    header_axis.axis("off")
    header_axis.text(
        0.0,
        0.75,
        "B",
        transform=header_axis.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="center",
    )
    header_axis.text(
        0.08,
        0.75,
        "Representative tilted stacks",
        transform=header_axis.transAxes,
        fontsize=8.5,
        ha="left",
        va="center",
    )
    rotation_note = (
        ""
        if np.isclose(rotation_degrees, 0.0)
        else (
            f"rotation: {abs(rotation_degrees):g}° "
            f"{'clockwise' if rotation_degrees < 0 else 'counterclockwise'}; "
        )
    )
    header_axis.text(
        1.0,
        0.15,
        f"{rotation_note}labels: firing rate at selected position",
        transform=header_axis.transAxes,
        fontsize=6,
        color="0.4",
        ha="right",
        va="center",
    )

    for example_index, ((row_index, column_index), color) in enumerate(
        zip(example_bins, EXAMPLE_COLORS)
    ):
        axis = figure.add_subplot(subgrid[1, example_index])
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.axis("off")

        cells_in_stack = result["active_cells_per_stack"][row_index][column_index]
        displayed_cells = cells_in_stack[:max_example_cells]
        x_position = float(result["x_centers"][column_index])
        y_position = float(result["y_centers"][row_index])
        axis.text(
            0.5,
            0.985,
            f"{example_index + 1}   r{row_index + 1},c{column_index + 1}   "
            f"n={len(cells_in_stack)}",
            color=color,
            fontsize=6.8,
            fontweight="bold",
            ha="center",
            va="top",
        )

        map_left = 0.06
        map_right = 0.72
        map_height = 0.065
        if len(displayed_cells) > 1:
            cell_bases = np.linspace(0.82, 0.31, len(displayed_cells))
        else:
            cell_bases = np.asarray([0.62])
        marker_points = []
        for cell_number, (cell_info, map_bottom) in enumerate(
            zip(displayed_cells, cell_bases),
            start=1,
        ):
            cell_row = result["cell_rows"][cell_info["cell_index"]]
            marker_points.append(
                plot_tilted_ratemap(
                    axis,
                    cell_row["ratemap"],
                    stacks.ratemap_extent(cell_row["spatial"]),
                    x_position=x_position,
                    y_position=y_position,
                    left=map_left,
                    right=map_right,
                    bottom=float(map_bottom),
                    height=map_height,
                    rotation_degrees=rotation_degrees,
                )
            )
            axis.text(
                0.76,
                float(map_bottom) + map_height * 0.46,
                f"#{cell_number}\n{cell_info['rate']:.2f} Hz",
                fontsize=5.2,
                ha="left",
                va="center",
                linespacing=1.05,
            )

        all_ratemaps = np.stack(
            [
                result["cell_rows"][cell_info["cell_index"]]["ratemap"]
                for cell_info in cells_in_stack
            ],
            axis=0,
        )
        mean_ratemap = nanmean_without_warning(all_ratemaps, axis=0)
        mean_bottom = 0.075
        mean_marker = plot_tilted_ratemap(
            axis,
            mean_ratemap,
            result["spatial_window"],
            x_position=x_position,
            y_position=y_position,
            left=map_left,
            right=map_right,
            bottom=mean_bottom,
            height=map_height,
            rotation_degrees=rotation_degrees,
            outline_color=color,
            outline_width=0.9,
        )
        axis.text(
            0.76,
            mean_bottom + map_height * 0.46,
            "stack\nmean",
            color=color,
            fontsize=5.4,
            fontweight="bold",
            ha="left",
            va="center",
            linespacing=1.05,
        )

        all_marker_points = [*marker_points, mean_marker]
        axis.plot(
            [point[0] for point in all_marker_points],
            [point[1] for point in all_marker_points],
            color="#08306B",
            linestyle=(0, (1.5, 1.2)),
            linewidth=1.1,
            zorder=4,
        )

def compute_mean_stack_ratemaps(result: dict) -> dict[tuple[int, int], np.ndarray]:
    mean_ratemaps: dict[tuple[int, int], np.ndarray] = {}
    n_rows, n_columns = result["active_counts"].shape
    for row_index in range(n_rows):
        for column_index in range(n_columns):
            cells_in_stack = result["active_cells_per_stack"][row_index][column_index]
            if not cells_in_stack:
                continue
            ratemaps = np.stack(
                [
                    result["cell_rows"][cell_info["cell_index"]]["ratemap"]
                    for cell_info in cells_in_stack
                ],
                axis=0,
            )
            mean_ratemaps[(row_index, column_index)] = nanmean_without_warning(
                ratemaps,
                axis=0,
            )
    return mean_ratemaps


def plot_mean_stack_mosaic(
    axis,
    result: dict,
    *,
    panel_label: str = "C",
    title: str = "Mean ratemap per stack\nacross sampled positions",
) -> None:
    mean_ratemaps = compute_mean_stack_ratemaps(result)
    finite_values = np.concatenate(
        [ratemap[np.isfinite(ratemap)] for ratemap in mean_ratemaps.values()]
    )
    vmax = float(np.percentile(finite_values, 99.5))
    norm = Normalize(vmin=0.0, vmax=vmax)

    x_edges = result["x_edges"]
    y_edges = result["y_edges"]
    tile_half_width = (x_edges[1] - x_edges[0]) * 0.41
    tile_half_height = (y_edges[1] - y_edges[0]) * 0.41
    image = None
    for (row_index, column_index), mean_ratemap in mean_ratemaps.items():
        x_position = result["x_centers"][column_index]
        y_position = result["y_centers"][row_index]
        image = axis.imshow(
            mean_ratemap,
            origin="lower",
            cmap=stacks.RATE_CMAP,
            norm=norm,
            interpolation="nearest",
            extent=(
                x_position - tile_half_width,
                x_position + tile_half_width,
                y_position - tile_half_height,
                y_position + tile_half_height,
            ),
            aspect="equal",
        )

    axis.set_xlim(x_edges[0], x_edges[-1])
    axis.set_ylim(y_edges[0], y_edges[-1])
    axis.set_aspect("equal")
    axis.set_xlabel("x (cm)")
    axis.set_ylabel("y (cm)")
    axis.set_title(title, loc="left", pad=3)
    if image is not None:
        colorbar = axis.figure.colorbar(
            image,
            ax=axis,
            fraction=0.042,
            pad=0.025,
        )
        colorbar.set_label("Hz")
        colorbar.outline.set_linewidth(0.6)
    add_panel_label(axis, panel_label, x=-0.20, y=1.10)


def mean_spatial_grids(
    rows: Iterable[dict],
    *,
    metric: str,
    experiment_order: Sequence[str],
) -> dict[str, np.ndarray]:
    """Average each normalized grid bin across rats."""
    value_column = METRIC_COLUMNS[metric]
    rows = list(rows)
    n_rows = max(row["grid_row"] for row in rows)
    n_columns = max(row["grid_column"] for row in rows)
    output: dict[str, np.ndarray] = {}
    for experiment_id in experiment_order:
        grid_values: list[list[list[float]]] = [
            [[] for _ in range(n_columns)] for _ in range(n_rows)
        ]
        for row in rows:
            if row["experiment_id"] != experiment_id:
                continue
            grid_values[row["grid_row"] - 1][row["grid_column"] - 1].append(
                float(row[value_column])
            )
        grid = np.full((n_rows, n_columns), np.nan, dtype=float)
        for row_index in range(n_rows):
            for column_index in range(n_columns):
                values = grid_values[row_index][column_index]
                if values:
                    grid[row_index, column_index] = float(np.mean(values))
        output[experiment_id] = grid
    return output


def plot_mean_spatial_grids(
    figure,
    grid_spec,
    rows: list[dict],
    *,
    metric: str,
    experiment_order: Sequence[str],
) -> None:
    grids = mean_spatial_grids(
        rows,
        metric=metric,
        experiment_order=experiment_order,
    )
    finite_values = np.concatenate(
        [grid[np.isfinite(grid)] for grid in grids.values()]
    )
    if metric == "fraction":
        vmin = 0.0
        vmax = float(np.ceil(np.percentile(finite_values, 99) * 100) / 100)
    else:
        vmin = 0.0
        vmax = float(np.ceil(np.percentile(finite_values, 99)))
    norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1)

    subgrid = grid_spec.subgridspec(
        2,
        len(experiment_order),
        height_ratios=(0.15, 1.0),
        wspace=0.18,
        hspace=0.10,
    )
    header_axis = figure.add_subplot(subgrid[0, :])
    header_axis.axis("off")
    header_axis.text(
        0.0,
        0.75,
        "A",
        transform=header_axis.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="center",
    )
    header_axis.text(
        0.08,
        0.75,
        (
            "Across-rat spatial mean stack fraction"
            if metric == "fraction"
            else "Across-rat spatial mean stack size"
        ),
        transform=header_axis.transAxes,
        fontsize=8.5,
        ha="left",
        va="center",
    )
    axes = []
    image = None
    for index, experiment_id in enumerate(experiment_order):
        axis = figure.add_subplot(subgrid[1, index])
        axes.append(axis)
        image = axis.imshow(
            grids[experiment_id],
            origin="lower",
            cmap="viridis",
            norm=norm,
            interpolation="nearest",
            aspect="equal",
            extent=(0, 1, 0, 1),
        )
        axis.set_title(environment_label(experiment_id), fontsize=8, pad=2)
        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])
        axis.set_xlabel("Normalized x")
        if index == 0:
            axis.set_ylabel("Normalized y")
        else:
            axis.set_yticklabels([])

    if image is not None:
        colorbar = figure.colorbar(
            image,
            ax=axes,
            fraction=0.040,
            pad=0.025,
        )
        colorbar.set_label("Fraction" if metric == "fraction" else "Count")
        colorbar.outline.set_linewidth(0.6)


def rat_environment_means(
    rows: Iterable[dict],
    *,
    metric: str,
    experiment_order: Sequence[str],
) -> dict[str, dict[str, float]]:
    value_column = METRIC_COLUMNS[metric]
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (row["rat_id"], row["experiment_id"])
        grouped.setdefault(key, []).append(float(row[value_column]))

    rat_ids = sorted({rat_id for rat_id, _ in grouped})
    output: dict[str, dict[str, float]] = {}
    for rat_id in rat_ids:
        values_by_environment = {}
        for experiment_id in experiment_order:
            values = grouped.get((rat_id, experiment_id), [])
            if values:
                values_by_environment[experiment_id] = float(np.mean(values))
        output[rat_id] = values_by_environment
    return output


def plot_rat_level_barplot(
    axis,
    rows: list[dict],
    *,
    metric: str,
    experiment_order: Sequence[str],
) -> None:
    rat_means = rat_environment_means(
        rows,
        metric=metric,
        experiment_order=experiment_order,
    )
    x = np.arange(len(experiment_order), dtype=float)
    group_means = []
    group_sds = []
    values_by_environment = []
    for experiment_id in experiment_order:
        values = np.asarray([
            rat_values[experiment_id]
            for rat_values in rat_means.values()
            if experiment_id in rat_values
        ], dtype=float)
        values_by_environment.append(values)
        group_means.append(float(np.mean(values)))
        group_sds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)

    colors = [
        ENVIRONMENT_COLORS.get(experiment_id, "#4C78A8")
        for experiment_id in experiment_order
    ]
    axis.bar(
        x,
        group_means,
        yerr=group_sds,
        width=0.68,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        capsize=3.5,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
        zorder=2,
    )
    for index, values in enumerate(values_by_environment):
        jitter = (
            np.linspace(-0.11, 0.11, len(values))
            if len(values) > 1
            else np.asarray([0.0])
        )
        axis.scatter(
            index + jitter,
            values,
            s=18,
            facecolor="white",
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )

    axis.set_xticks(x)
    axis.set_xticklabels(
        [environment_label(experiment_id) for experiment_id in experiment_order]
    )
    axis.set_xlabel("Environment")
    axis.set_ylabel(f"Mean {METRIC_LABELS[metric].lower()}\n(across grid positions)")
    axis.set_title(
        "Mean stack fraction across environments"
        if metric == "fraction"
        else "Mean stack size across environments"
    )
    axis.grid(axis="y", color="0.9", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_xlim(-0.45, len(experiment_order) - 0.55)
    axis.set_ylim(bottom=0)
    add_panel_label(axis, "D", x=-0.17, y=1.08)


def build_figure(
    result: dict,
    summary_rows: list[dict],
    *,
    example_bins: Sequence[tuple[int, int]],
    max_example_cells: int,
    stack_rotation_degrees: float,
    metric: str,
    experiment_order: Sequence[str],
):
    configure_plot_style()
    figure = plt.figure(figsize=(7.25, 6.9))
    outer = figure.add_gridspec(
        2,
        2,
        width_ratios=(0.88, 1.35),
        height_ratios=(1.0, 1.08),
        left=0.085,
        right=0.965,
        bottom=0.08,
        top=0.97,
        wspace=0.42,
        hspace=0.32,
    )

    spatial_axis = figure.add_subplot(outer[0, 0])
    plot_example_spatial_map(
        spatial_axis,
        result,
        example_bins,
        metric=metric,
    )
    plot_example_stacks(
        figure,
        outer[0, 1],
        result,
        example_bins,
        max_example_cells=max_example_cells,
        rotation_degrees=stack_rotation_degrees,
    )

    mosaic_axis = figure.add_subplot(outer[1, 0])
    plot_mean_stack_mosaic(mosaic_axis, result)
    barplot_axis = figure.add_subplot(outer[1, 1])
    plot_rat_level_barplot(
        barplot_axis,
        summary_rows,
        metric=metric,
        experiment_order=experiment_order,
    )
    return figure


def build_supplementary_figure(
    results_by_experiment: dict[str, dict],
    summary_rows: list[dict],
    *,
    metric: str,
    experiment_order: Sequence[str],
):
    """Build supplementary spatial summaries and all-environment stack mosaics."""
    configure_plot_style()
    figure = plt.figure(figsize=(7.25, 8.8))
    outer = figure.add_gridspec(
        3,
        2,
        height_ratios=(0.64, 1.0, 1.0),
        left=0.075,
        right=0.96,
        bottom=0.065,
        top=0.97,
        wspace=0.42,
        hspace=0.40,
    )
    plot_mean_spatial_grids(
        figure,
        outer[0, :],
        summary_rows,
        metric=metric,
        experiment_order=experiment_order,
    )

    panel_labels = ("B", "C", "D", "E")
    for index, (panel_label, experiment_id) in enumerate(
        zip(panel_labels, experiment_order)
    ):
        axis = figure.add_subplot(outer[index // 2 + 1, index % 2])
        plot_mean_stack_mosaic(
            axis,
            results_by_experiment[experiment_id],
            panel_label=panel_label,
            title=(
                f"Mean ratemap per stack: environment "
                f"{environment_label(experiment_id)}"
            ),
        )
    return figure


def parse_args() -> argparse.Namespace:
    default_results_dir = (
        stacks.REPO_ROOT / "MultiplePFAnalysis" / "results" / "stacks_with_ratemaps"
    )
    default_figure_dir = (
        stacks.REPO_ROOT
        / "MultiplePFAnalysis"
        / "results"
        / "paper_figures"
        / "1_place_cell_stacks"
        / "figure_1_1_place_cell_stacks"
    )
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=stacks.REPO_ROOT / "MultiplePFAnalysis" / "cache" / "preprocessed",
    )
    parser.add_argument("--results-dir", type=Path, default=default_results_dir)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help=(
            "CSV from analyze_place_cell_stacks.py. Defaults to "
            "<results-dir>/overall/linearized_stack_activity_all_rats.csv."
        ),
    )
    parser.add_argument("--rat-id", default="R2470")
    parser.add_argument("--experiment-id", default="exp_scales_a")
    parser.add_argument(
        "--experiment-order",
        nargs="+",
        default=list(DEFAULT_EXPERIMENT_ORDER),
    )
    parser.add_argument("--grid-rows", type=int, default=10)
    parser.add_argument("--grid-columns", type=int, default=10)
    parser.add_argument(
        "--example-bins",
        nargs=3,
        type=parse_example_bin,
        metavar="ROW,COLUMN",
        default=[(1, 1), (4, 4), (8, 8)],
        help=(
            "Three one-based grid positions for panel B. "
            "Default: 2,2 5,5 9,9."
        ),
    )
    parser.add_argument(
        "--max-example-cells",
        type=int,
        default=4,
        help="Number of highest-rate cells shown for each example stack.",
    )
    parser.add_argument(
        "--stack-rotation-degrees",
        type=float,
        default=0.0,
        help=(
            "In-plane rotation of the perspective ratemap layers in panel B. "
            "Negative values rotate clockwise. Default: 0 degrees."
        ),
    )
    parser.add_argument(
        "--threshold-hz",
        type=float,
        default=None,
        help="Defaults to the threshold recorded in the summary CSV.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_COLUMNS),
        default="fraction",
        help="Use normalized stack fractions (default) or raw active-cell counts.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=default_figure_dir / "figure_1_1_place_cell_stacks",
        help="Main-figure output path without extension.",
    )
    parser.add_argument(
        "--supplement-output-stem",
        type=Path,
        default=default_figure_dir / "supplementary_figure_1_1_place_cell_stacks",
        help="Supplementary-figure output path without extension. Use 'none' to skip.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "svg"),
        default=("png", "svg"),
    )
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def save_figure_formats(
    figure,
    output_stem: Path,
    *,
    output_formats: Sequence[str],
    dpi: int,
) -> None:
    for output_format in output_formats:
        format_dir = output_stem.parent / output_format
        format_dir.mkdir(parents=True, exist_ok=True)
        output_path = format_dir / f"{output_stem.name}.{output_format}"
        save_kwargs = {"bbox_inches": "tight"}
        if output_format == "png":
            save_kwargs["dpi"] = dpi
        figure.savefig(output_path, **save_kwargs)
        print(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir.expanduser()
    results_dir = args.results_dir.expanduser()
    summary_csv = (
        args.summary_csv.expanduser()
        if args.summary_csv is not None
        else results_dir / "overall" / "linearized_stack_activity_all_rats.csv"
    )
    threshold_hz = args.threshold_hz
    if threshold_hz is None:
        threshold_hz = threshold_from_csv(
            summary_csv,
            rat_id=args.rat_id,
            experiment_id=args.experiment_id,
        )

    summary_rows = read_summary_rows(
        summary_csv,
        threshold_hz=threshold_hz,
        experiment_order=args.experiment_order,
    )

    print(f"Loading cached experiment for {args.rat_id} ...")
    experiments = stacks.load_cached_experiments(cache_dir, args.rat_id)
    selected_experiments = stacks.select_experiments(
        experiments,
        args.experiment_order,
    )
    results_by_experiment = {}
    for experiment in selected_experiments:
        experiment_id = experiment.info.get("experiment_id", experiment.session)
        print(f"Computing stack summaries for {experiment_id} ...")
        results_by_experiment[experiment_id] = stacks.build_environment_result(
            experiment,
            n_rows=args.grid_rows,
            n_columns=args.grid_columns,
            threshold_fraction=0.0,
            min_threshold_hz=threshold_hz,
            show_progress=False,
        )
    if args.experiment_id not in results_by_experiment:
        raise ValueError(
            f"Example environment {args.experiment_id} is not in --experiment-order"
        )
    example_result = results_by_experiment[args.experiment_id]
    validate_example_bins(args.example_bins, example_result)

    figure = build_figure(
        example_result,
        summary_rows,
        example_bins=args.example_bins,
        max_example_cells=args.max_example_cells,
        stack_rotation_degrees=args.stack_rotation_degrees,
        metric=args.metric,
        experiment_order=args.experiment_order,
    )

    output_stem = args.output_stem.expanduser()
    if not output_stem.is_absolute():
        output_stem = stacks.REPO_ROOT / output_stem
    save_figure_formats(
        figure,
        output_stem,
        output_formats=args.formats,
        dpi=args.dpi,
    )
    plt.close(figure)

    if str(args.supplement_output_stem).lower() != "none":
        supplementary_figure = build_supplementary_figure(
            results_by_experiment,
            summary_rows,
            metric=args.metric,
            experiment_order=args.experiment_order,
        )
        supplement_output_stem = args.supplement_output_stem.expanduser()
        if not supplement_output_stem.is_absolute():
            supplement_output_stem = stacks.REPO_ROOT / supplement_output_stem
        save_figure_formats(
            supplementary_figure,
            supplement_output_stem,
            output_formats=args.formats,
            dpi=args.dpi,
        )
        plt.close(supplementary_figure)


if __name__ == "__main__":
    main()
