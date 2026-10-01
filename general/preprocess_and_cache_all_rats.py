#!/usr/bin/env python
"""Preprocess Barry lab ExpScales recordings and cache them as NumPy files.

This script keeps the preprocessing path used by the notebooks:

1. load recordings with the original Barry lab `Recordings` class
2. run `MultiplePFAnalysis.barrylab_pipeline.preprocess_recordings_unit_analysis`
3. convert the result to `ExperimentData` dataclasses
4. save one `.npy` object array per rat

Load cached experiments later with:

    experiments = np.load("MultiplePFAnalysis/cache/preprocessed/R2470_experiments.npy", allow_pickle=True).tolist()
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
import traceback

import numpy as np


def find_repo_root() -> Path:
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    for candidate in candidates:
        if (candidate / "MultiplePFAnalysis" / "barrylab_pipeline.py").exists():
            return candidate
    raise RuntimeError("Could not find repo root containing MultiplePFAnalysis/barrylab_pipeline.py")


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MultiplePFAnalysis import barrylab_pipeline as blp  # noqa: E402


def discover_rat_ids(data_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in data_root.iterdir()
        if path.is_dir() and path.name.startswith("R")
    )


def strip_raw_waveforms(experiments) -> None:
    """Drop raw waveforms while keeping computed waveform analysis results."""
    for experiment in experiments:
        for unit in experiment.units:
            unit.waveforms = None
        for tetrode in experiment.tetrode_spikes:
            tetrode.waveforms = None


def save_experiments(experiments, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.asarray(experiments, dtype=object), allow_pickle=True)


def write_metadata(
    metadata_path: Path,
    *,
    rat_id: str,
    data_root: Path,
    recording_paths: list[Path],
    output_path: Path,
    recompute: bool,
    include_waveforms: bool,
    compute_autocorrelations: bool,
) -> None:
    metadata = {
        "rat_id": rat_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "recording_paths": [str(path) for path in recording_paths],
        "output_path": str(output_path),
        "recompute": recompute,
        "include_waveforms": include_waveforms,
        "compute_autocorrelations": compute_autocorrelations,
        "loader": "MultiplePFAnalysis.barrylab_pipeline",
        "format": "np.save object array of ExperimentData; load with np.load(..., allow_pickle=True).tolist()",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))


def preprocess_and_cache_rat(
    rat_id: str,
    *,
    data_root: Path,
    output_dir: Path,
    recompute: bool,
    overwrite: bool,
    include_waveforms: bool,
    compute_autocorrelations: bool,
    verbose: bool,
) -> Path:
    output_path = output_dir / f"{rat_id}_experiments.npy"
    metadata_path = output_dir / f"{rat_id}_experiments.json"

    if output_path.exists() and not overwrite:
        print(f"[skip] {rat_id}: {output_path} already exists")
        return output_path

    recording_paths = blp.get_paths_to_rat_recordings_on_first_day(data_root, rat_id)
    print(f"[load] {rat_id}: {len(recording_paths)} recordings")
    recordings = blp.load_barrylab_recordings(
        recording_paths,
        load_waveforms=True,
        continuous_data_type=None,
        correct_repeated_a_id=True,
        verbose=verbose,
    )

    print(f"[preprocess] {rat_id}")
    blp.preprocess_recordings_unit_analysis(
        recordings,
        recompute=recompute,
        compute_autocorrelations=compute_autocorrelations,
        verbose=verbose,
    )

    print(f"[convert] {rat_id}")
    experiments = blp.recordings_to_experiments(recordings)
    if not include_waveforms:
        strip_raw_waveforms(experiments)

    print(f"[save] {rat_id}: {output_path}")
    save_experiments(experiments, output_path)
    write_metadata(
        metadata_path,
        rat_id=rat_id,
        data_root=data_root,
        recording_paths=recording_paths,
        output_path=output_path,
        recompute=recompute,
        include_waveforms=include_waveforms,
        compute_autocorrelations=compute_autocorrelations,
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess all requested rats and cache ExperimentData lists as .npy files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "Paper_ExpScales_NoPreProcessing",
        help="Root folder containing rat folders such as R2470.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "MultiplePFAnalysis" / "cache" / "preprocessed",
        help="Directory where per-rat .npy cache files are written.",
    )
    parser.add_argument(
        "--rats",
        nargs="+",
        default=None,
        help="Rat IDs to preprocess. Defaults to all R* folders under --data-root.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute analysis even if it is already available in loaded recordings.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-rat cache files.",
    )
    parser.add_argument(
        "--include-waveforms",
        action="store_true",
        help="Keep raw waveforms in the cached dataclasses. This can make files very large.",
    )
    parser.add_argument(
        "--no-autocorrelations",
        action="store_true",
        help="Skip spike-time autocorrelation preprocessing.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later rats if one rat fails.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass verbose=True into the original Barry lab loader/preprocessing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    rat_ids = args.rats if args.rats is not None else discover_rat_ids(data_root)

    print(f"data_root: {data_root}")
    print(f"output_dir: {output_dir}")
    print(f"rats: {', '.join(rat_ids)}")

    failures = []
    for rat_id in rat_ids:
        try:
            preprocess_and_cache_rat(
                rat_id,
                data_root=data_root,
                output_dir=output_dir,
                recompute=args.recompute,
                overwrite=args.overwrite,
                include_waveforms=args.include_waveforms,
                compute_autocorrelations=not args.no_autocorrelations,
                verbose=args.verbose,
            )
        except Exception as exc:
            failures.append((rat_id, exc))
            print(f"[error] {rat_id}: {exc}")
            traceback.print_exc()
            if not args.continue_on_error:
                return 1

    if failures:
        print("Completed with failures:")
        for rat_id, exc in failures:
            print(f"  {rat_id}: {exc}")
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
