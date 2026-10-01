
import argparse
import csv
import gc
import os
import pathlib
import platform
import subprocess
import sys
import time
from pathlib import Path

if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath


def find_repo_root(start=None):
    if start is None:
        start = Path.cwd()
    else:
        start = Path(start)

    paths_to_check = [start]
    for parent in start.parents:
        paths_to_check.append(parent)

    for path in paths_to_check:
        if (path / "MultiplePFAnalysis" / "__init__.py").is_file():
            return path
    raise FileNotFoundError("Could not find repository root containing MultiplePFAnalysis")


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CACHE_DIR = Path(
    os.environ.get(
        "MPF_PREPROCESSED_CACHE",
        REPO_ROOT / "MultiplePFAnalysis" / "cache" / "preprocessed",
    )
).expanduser()

DEFAULT_OUTPUT_DIR = REPO_ROOT / "MultiplePFAnalysis" / "results" / "rotation_remapping_correlations"

ALL_ROTATIONS = (0, 90, 180, 270)

CANONICAL_EXPERIMENT_ORDER = [
    "exp_scales_a",
    "exp_scales_b",
    "exp_scales_c",
    "exp_scales_d",
    "exp_scales_a2",
]


def sort_experiments_by_canonical_order(experiments):
    def position_in_canonical_order(experiment):
        experiment_id = experiment.info["experiment_id"]
        if experiment_id in CANONICAL_EXPERIMENT_ORDER:
            return CANONICAL_EXPERIMENT_ORDER.index(experiment_id)
        return len(CANONICAL_EXPERIMENT_ORDER)

    return sorted(experiments, key=position_in_canonical_order)


def discover_cached_rat_ids(cache_dir):
    rat_ids = []
    for cache_file in sorted(cache_dir.glob("*_experiments.npy")):
        if cache_file.name.startswith("._"):
            continue
        rat_ids.append(cache_file.name.replace("_experiments.npy", ""))
    if not rat_ids:
        raise FileNotFoundError(f"No *_experiments.npy files found in {cache_dir}")
    return rat_ids


def build_env_2_units(experiment_2):
    import numpy as np

    env_2_units = {}
    for unit in experiment_2.units:
        if unit.analysis.get("category") != "place_cell":
            continue

        unit_id_2 = (unit.tetrode_nr, unit.cluster_id)
        spatial = unit.analysis["spatial_ratemaps"]
        ratemap_2 = np.asarray(spatial["spike_rates_smoothed"], dtype=float)

        env_2_units[unit_id_2] = {
            "ratemap_2_flat": ratemap_2.flatten(),
        }
    return env_2_units


def build_env_1_units(experiment_1, target_shape):
    import numpy as np
    from scipy.ndimage import zoom

    env_1_units = {}

    for unit in experiment_1.units:
        if unit.analysis.get("category") != "place_cell":
            continue

        unit_id_1 = (unit.tetrode_nr, unit.cluster_id)
        spatial = unit.analysis["spatial_ratemaps"]
        ratemap_1 = np.asarray(spatial["spike_rates_smoothed"], dtype=float)

        rotated_0 = ratemap_1
        rotated_90 = np.flipud(ratemap_1.T)
        rotated_180 = np.flipud(rotated_90.T)
        rotated_270 = np.flipud(rotated_180.T)

        zf_0 = (target_shape[0] / rotated_0.shape[0], target_shape[1] / rotated_0.shape[1])
        zf_90 = (target_shape[0] / rotated_90.shape[0], target_shape[1] / rotated_90.shape[1])
        zf_180 = (target_shape[0] / rotated_180.shape[0], target_shape[1] / rotated_180.shape[1])
        zf_270 = (target_shape[0] / rotated_270.shape[0], target_shape[1] / rotated_270.shape[1])

        env_1_units[unit_id_1] = {
            "flat_0": zoom(rotated_0, zf_0, order=1).flatten(),
            "flat_90": zoom(rotated_90, zf_90, order=1).flatten(),
            "flat_180": zoom(rotated_180, zf_180, order=1).flatten(),
            "flat_270": zoom(rotated_270, zf_270, order=1).flatten(),
        }
    return env_1_units


def per_unit_remapping_correlation_different_rotations(all_unit_ids, env_1_units, env_2_units, rotation):
    import numpy as np

    results_units = {}

    for unit_id in all_unit_ids:
        if unit_id not in env_1_units or unit_id not in env_2_units:
            continue

        ratemap_1_flat = env_1_units[unit_id][f"flat_{rotation}"]
        ratemap_2_flat = env_2_units[unit_id]["ratemap_2_flat"]

        valid = np.isfinite(ratemap_1_flat) & np.isfinite(ratemap_2_flat)

        max_val_1 = np.nanmax(ratemap_1_flat)
        max_val_2 = np.nanmax(ratemap_2_flat)
        passes = np.sum(valid) > 3 and max_val_1 > 0 and max_val_2 > 0

        if not passes:
            continue

        r = np.corrcoef(ratemap_1_flat[valid], ratemap_2_flat[valid])[0, 1]
        results_units[unit_id] = r

    return results_units


PAIR_INDICES_AND_BASELINE_ROTATION = [
    (0, 1, 90),
    (1, 2, 90),
    (2, 3, 90),
    (0, 2, 180),
    (1, 3, 180),
]


def run_worker(rat_id, cache_dir, output_dir):
    import numpy as np
    import MultiplePFAnalysis

    cache_file = cache_dir / f"{rat_id}_experiments.npy"
    print(f"Loading {rat_id}: {cache_file}")

    experiments = np.load(cache_file, allow_pickle=True).tolist()
    print(f"Loaded {len(experiments)} experiments for {rat_id}")

    experiments = sort_experiments_by_canonical_order(experiments)

    for experiment in experiments:
        experiment.position = None
        experiment.tetrode_spikes = []
        for unit in experiment.units:
            unit.timestamps = None
            unit.waveforms = None
    gc.collect()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{rat_id}_rotation_correlations.csv"

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rat_id", "exp_id_a", "exp_id_b", "baseline_rotation",
                "tetrode_nr", "cluster_id", "rotation", "r",
            ],
        )
        writer.writeheader()

        for index_a, index_b, baseline_rotation in PAIR_INDICES_AND_BASELINE_ROTATION:
            if index_a >= len(experiments) or index_b >= len(experiments):
                print(f"  Pair ({index_a}, {index_b}): not enough experiments, skipping")
                continue

            experiment_1 = experiments[index_a]
            experiment_2 = experiments[index_b]
            exp_id_a = experiment_1.info["experiment_id"]
            exp_id_b = experiment_2.info["experiment_id"]

            env_2_units = build_env_2_units(experiment_2)
            if not env_2_units:
                print(f"  {exp_id_a} -> {exp_id_b}: no units, skipping")
                continue
            first_unit_2 = experiment_2.units[0]
            target_shape = np.asarray(
                first_unit_2.analysis["spatial_ratemaps"]["spike_rates_smoothed"]
            ).shape

            env_1_units = build_env_1_units(experiment_1, target_shape)

            all_unit_ids = set(env_1_units.keys()) | set(env_2_units.keys())

            n_rows_written = 0
            for rotation in ALL_ROTATIONS:
                results_units = per_unit_remapping_correlation_different_rotations(
                    all_unit_ids, env_1_units, env_2_units, rotation
                )
                for unit_id, r in results_units.items():
                    tetrode_nr, cluster_id = unit_id
                    writer.writerow({
                        "rat_id": rat_id,
                        "exp_id_a": exp_id_a,
                        "exp_id_b": exp_id_b,
                        "baseline_rotation": baseline_rotation,
                        "tetrode_nr": tetrode_nr,
                        "cluster_id": cluster_id,
                        "rotation": rotation,
                        "r": r,
                    })
                    n_rows_written += 1

            print(f"  {exp_id_a} -> {exp_id_b}: wrote {n_rows_written} rows (4 rotations)")

            del env_1_units, env_2_units
            gc.collect()

    print(f"Wrote {output_path}")

    del experiments
    gc.collect()


MAX_ATTEMPTS_PER_RAT = 8
SECONDS_BETWEEN_RETRIES = 30


def run_one_rat_with_retries(rat_id, cache_dir, output_dir):
    for attempt in range(1, MAX_ATTEMPTS_PER_RAT + 1):
        print(f"\n=== {rat_id}: attempt {attempt} of {MAX_ATTEMPTS_PER_RAT} ===")
        result = subprocess.run(
            [
                sys.executable,
                __file__,
                "--rat-id", rat_id,
            ]
        )
        if result.returncode == 0:
            return True

        print(f"{rat_id}: attempt {attempt} exited with code {result.returncode}")
        if attempt < MAX_ATTEMPTS_PER_RAT:
            print(f"  Waiting {SECONDS_BETWEEN_RETRIES}s before retrying, to let memory clear up...")
            gc.collect()
            time.sleep(SECONDS_BETWEEN_RETRIES)

    return False


def run_orchestrator(rat_ids, cache_dir, output_dir):
    print(f"Orchestrating {len(rat_ids)} rats, one subprocess each: {', '.join(rat_ids)}")
    failed_rats = []

    for rat_id in rat_ids:
        succeeded = run_one_rat_with_retries(rat_id, cache_dir, output_dir)
        if not succeeded:
            failed_rats.append(rat_id)

    if failed_rats:
        print(
            f"\n{len(failed_rats)} of {len(rat_ids)} rats did NOT complete, even after "
            f"{MAX_ATTEMPTS_PER_RAT} attempts each: {', '.join(failed_rats)}"
        )
        raise RuntimeError(
            f"{len(failed_rats)} rat(s) could not be loaded, most likely due to memory "
            f"pressure: {', '.join(failed_rats)}. Refusing to continue with incomplete "
            f"data. Close other programs/browser tabs to free memory and re-run just "
            f"these rats with --rat-ids, or increase the Windows page file (virtual "
            f"memory) size, then re-run this script."
        )

    print(f"\nAll {len(rat_ids)} rats completed successfully.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    rat_group = parser.add_mutually_exclusive_group()
    rat_group.add_argument(
        "--rat-id",
        help="WORKER MODE: process exactly this one rat directly.",
    )
    rat_group.add_argument(
        "--rat-ids",
        nargs="+",
        help="ORCHESTRATOR MODE: process each of these rats as a separate subprocess.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cache_dir = DEFAULT_CACHE_DIR.expanduser()
    output_dir = DEFAULT_OUTPUT_DIR.expanduser()

    if args.rat_id:
        run_worker(args.rat_id, cache_dir, output_dir)
    else:
        if args.rat_ids:
            rat_ids = args.rat_ids
        else:
            rat_ids = discover_cached_rat_ids(cache_dir)
        run_orchestrator(rat_ids, cache_dir, output_dir)


if __name__ == "__main__":
    main()
