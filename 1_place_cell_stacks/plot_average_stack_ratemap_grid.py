#!/usr/bin/env python3
"""Plot a grid of mean stack ratemaps, one figure per rat and environment.

For every non-empty place-cell stack on the 10 x 10 (default) grid, this
script averages the ratemaps of the active cells in that stack and draws the
resulting mean ratemap as a small tile positioned at roughly the stack's
location in the arena, with a gap between neighbouring tiles.

Examples:
    python MultiplePFAnalysis/1_place_cell_stacks/plot_average_stack_ratemap_grid.py

    python MultiplePFAnalysis/1_place_cell_stacks/plot_average_stack_ratemap_grid.py \
        --rat-ids R2470 --experiment-ids exp_scales_a exp_scales_b

Outputs are written under
``<results-dir>/<rat_id>/<experiment_id>/average_ratemap_grid.png`` (plus a
``_no_ticks_text`` variant) by default.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_place_cell_stacks as stacks  # noqa: E402


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


def compute_mean_stack_ratemaps(result: dict) -> dict[tuple[int, int], np.ndarray]:
    """Mean ratemap of active cells for each non-empty stack, keyed by (row, column)."""
    mean_ratemaps: dict[tuple[int, int], np.ndarray] = {}
    n_rows = len(result["y_centers"])
    n_columns = len(result["x_centers"])
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
            mean_ratemaps[(row_index, column_index)] = np.nanmean(ratemaps, axis=0)
    return mean_ratemaps


def plot_ratemap_grid(
    result: dict,
    mean_ratemaps: dict[tuple[int, int], np.ndarray],
    *,
    tile_shrink: float,
    shared_scale: bool,
    title: str | None,
    dpi: int,
    output_path: Path,
    clean: bool,
) -> None:
    x_centers = result["x_centers"]
    y_centers = result["y_centers"]
    x_edges = result["x_edges"]
    y_edges = result["y_edges"]
    tile_half_width = (x_edges[1] - x_edges[0]) / 2.0 * tile_shrink
    tile_half_height = (y_edges[1] - y_edges[0]) / 2.0 * tile_shrink

    if shared_scale:
        vmax = max(float(np.nanmax(tile)) for tile in mean_ratemaps.values())
        norm = Normalize(vmin=0.0, vmax=vmax if vmax > 0 else 1.0)
    else:
        norm = None

    figsize = (6.5, 6.5) if clean else (7.6, 6.8)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    image = None
    for (row_index, column_index), mean_ratemap in mean_ratemaps.items():
        x_position = x_centers[column_index]
        y_position = y_centers[row_index]
        extent = (
            x_position - tile_half_width,
            x_position + tile_half_width,
            y_position - tile_half_height,
            y_position + tile_half_height,
        )
        image = ax.imshow(
            mean_ratemap,
            origin="lower",
            cmap=stacks.RATE_CMAP,
            extent=extent,
            aspect="equal",
            norm=norm,
        )

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_aspect("equal")

    if clean:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
    else:
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        if title:
            ax.set_title(title)
        if image is not None and shared_scale:
            colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
            colorbar.set_label("Mean firing rate (Hz)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.02 if clean else 0.1,
    )
    plt.close(fig)


def plot_ratemap_grid_per_tile_colorbar(
    result: dict,
    mean_ratemaps: dict[tuple[int, int], np.ndarray],
    *,
    title: str | None,
    dpi: int,
    output_path: Path,
) -> None:
    """One subplot per stack, positioned on a uniform grid, each with its own colorbar."""
    n_rows = len(result["y_centers"])
    n_columns = len(result["x_centers"])

    fig = plt.figure(figsize=(2.3 * n_columns, 2.1 * n_rows))
    grid_spec = fig.add_gridspec(n_rows, n_columns, wspace=0.65, hspace=0.35)

    for (row_index, column_index), mean_ratemap in mean_ratemaps.items():
        # Grid row 0 (top of figure) corresponds to the highest y_center.
        display_row = n_rows - 1 - row_index
        axis = fig.add_subplot(grid_spec[display_row, column_index])
        image = axis.imshow(
            mean_ratemap,
            origin="lower",
            cmap=stacks.RATE_CMAP,
            aspect="equal",
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f"r{row_index + 1:02d}c{column_index + 1:02d}", fontsize=6)

        divider = make_axes_locatable(axis)
        colorbar_axis = divider.append_axes("right", size="8%", pad=0.06)
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.ax.tick_params(labelsize=4)

    if title:
        fig.suptitle(title, fontsize=13)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--rat-ids",
        nargs="+",
        default=None,
        help="Rat IDs to plot. Defaults to all *_experiments.npy caches in --cache-dir.",
    )
    parser.add_argument(
        "--experiment-ids",
        nargs="+",
        default=list(stacks.DEFAULT_EXPERIMENT_IDS),
        help="Environment IDs to plot. Defaults to A, B, C, and D.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=stacks.REPO_ROOT / "MultiplePFAnalysis" / "cache" / "preprocessed",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=stacks.REPO_ROOT / "MultiplePFAnalysis" / "results" / "stacks_with_ratemaps",
        help="Used both as the output directory and to look up saved thresholds.",
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
    parser.add_argument(
        "--tile-shrink",
        type=float,
        default=0.82,
        help="Fraction of each grid cell's width/height occupied by a tile (creates the gaps).",
    )
    parser.add_argument(
        "--per-tile-scale",
        action="store_true",
        help="Normalize each tile's colors independently instead of sharing one scale across the grid.",
    )
    parser.add_argument(
        "--per-tile-colorbar",
        action="store_true",
        help=(
            "Also write a variant with one subplot and one colorbar per stack, "
            "each tile normalized to its own scale (average_ratemap_grid_per_tile_colorbar.png)."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir.expanduser()
    results_dir = args.results_dir.expanduser()
    show_progress = not args.no_progress

    rat_ids = args.rat_ids or stacks.discover_cached_rat_ids(cache_dir)

    rat_iterator = tqdm(rat_ids, desc="Rats", unit="rat", disable=not show_progress)
    for rat_id in rat_iterator:
        rat_iterator.set_postfix_str(rat_id, refresh=True)
        experiments = stacks.load_cached_experiments(cache_dir, rat_id)
        selected_experiments = stacks.select_experiments(experiments, args.experiment_ids)

        environment_iterator = tqdm(
            selected_experiments,
            desc=f"{rat_id}: environments",
            unit="environment",
            leave=False,
            disable=not show_progress,
        )
        for experiment in environment_iterator:
            experiment_id = experiment.info.get("experiment_id", experiment.session)
            environment_iterator.set_postfix_str(experiment_id, refresh=True)

            threshold_hz = args.threshold_hz
            if threshold_hz is None:
                threshold_hz = threshold_from_summary_csv(results_dir, rat_id, experiment_id)
            if threshold_hz is None:
                threshold_fraction = args.threshold_fraction
                min_threshold_hz = args.min_threshold_hz
            else:
                threshold_fraction = 0.0
                min_threshold_hz = threshold_hz

            result = stacks.build_environment_result(
                experiment,
                n_rows=args.grid_rows,
                n_columns=args.grid_columns,
                threshold_fraction=threshold_fraction,
                min_threshold_hz=min_threshold_hz,
                show_progress=False,
            )
            mean_ratemaps = compute_mean_stack_ratemaps(result)
            if not mean_ratemaps:
                tqdm.write(f"{rat_id} {experiment_id}: no non-empty stacks, skipping")
                del result
                continue

            environment_dir = results_dir / rat_id / experiment_id
            output_path = environment_dir / "average_ratemap_grid.png"
            clean_output_path = environment_dir / "average_ratemap_grid_no_ticks_text.png"
            title = (
                f"{rat_id} {result['arena_label']}: mean ratemap per stack\n"
                f"{len(mean_ratemaps)} stacks, threshold {result['threshold_hz']:.3f} Hz"
            )

            plot_ratemap_grid(
                result,
                mean_ratemaps,
                tile_shrink=args.tile_shrink,
                shared_scale=not args.per_tile_scale,
                title=title,
                dpi=args.dpi,
                output_path=output_path,
                clean=False,
            )
            plot_ratemap_grid(
                result,
                mean_ratemaps,
                tile_shrink=args.tile_shrink,
                shared_scale=not args.per_tile_scale,
                title=None,
                dpi=args.dpi,
                output_path=clean_output_path,
                clean=True,
            )
            tqdm.write(f"Wrote {output_path}")
            tqdm.write(f"Wrote {clean_output_path}")

            if args.per_tile_colorbar:
                per_tile_output_path = environment_dir / "average_ratemap_grid_per_tile_colorbar.png"
                plot_ratemap_grid_per_tile_colorbar(
                    result,
                    mean_ratemaps,
                    title=title,
                    dpi=args.dpi,
                    output_path=per_tile_output_path,
                )
                tqdm.write(f"Wrote {per_tile_output_path}")
            del result

        del selected_experiments
        del experiments


if __name__ == "__main__":
    main()
