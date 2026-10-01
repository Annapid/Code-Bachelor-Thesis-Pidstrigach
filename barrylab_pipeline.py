from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .datatypes import ExperimentData, PositionData, TetrodeSpikeData, UnitData
from .utils import parse_experiment_identity


def _ensure_barrylab_ephys_analysis_on_path() -> None:
    package_root = Path(__file__).resolve().parent
    repo_root = package_root.parent
    ephys_root = repo_root / "electrophysiology_analysis"
    if ephys_root.exists() and str(ephys_root) not in sys.path:
        sys.path.insert(0, str(ephys_root))


def _rename_last_recording_a2(recordings) -> None:
    if recordings[-1].info["experiment_id"] == "exp_scales_a":
        recordings[-1].edit_info()["experiment_id"] = "exp_scales_a2"
    elif recordings[-1].info["experiment_id"] != "exp_scales_a2":
        print(
            "Final recording not exp_scales_a, skipping renaming {} {}".format(
                recordings[-1].info["animal"],
                recordings[-1].info["rec_datetime"],
            )
        )


def _import_barrylab_modules(*, include_preprocess: bool = False) -> dict[str, Any]:
    _ensure_barrylab_ephys_analysis_on_path()
    try:
        from barrylab_ephys_analysis.recording_io import Recordings
        from barrylab_ephys_analysis.scripts.exp_scales.params import Params
        if include_preprocess:
            from barrylab_ephys_analysis.scripts.exp_scales import paper_preprocess
        else:
            paper_preprocess = None
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import the original barrylab_ephys_analysis stack. "
            "Activate/install the environment that provides its dependencies "
            "such as openEPhys_DACQ and elephant."
        ) from exc

    return {
        "Recordings": Recordings,
        "Params": Params,
        "paper_preprocess": paper_preprocess,
    }


def get_paths_to_rat_recordings_on_first_day(root: str | Path, rat_id: str) -> list[Path]:
    """Return recording folders for one rat, matching the original first-day convention."""
    rat_root = Path(root).expanduser().resolve() / rat_id
    day_roots = sorted(path for path in rat_root.iterdir() if path.is_dir())
    if not day_roots:
        raise FileNotFoundError(f"No recording-day folders found under {rat_root}")

    first_day = day_roots[0]
    recording_paths = sorted(path for path in first_day.iterdir() if (path / "experiment_1.nwb").is_file())
    if not recording_paths:
        raise FileNotFoundError(f"No experiment_1.nwb recordings found under {first_day}")
    return recording_paths


def load_barrylab_recordings(
    fpaths: list[str | Path],
    *,
    clustering_name: str | None = None,
    load_waveforms: bool = True,
    continuous_data_type: str | None = None,
    load_analysis: bool = False,
    correct_repeated_a_id: bool = True,
    verbose: bool = False,
    **recordings_kwargs: Any,
):
    """Load data with the original `barrylab_ephys_analysis.recording_io.Recordings`."""
    modules = _import_barrylab_modules()
    Params = modules["Params"]
    Recordings = modules["Recordings"]

    if clustering_name is None:
        clustering_name = Params.clustering_name

    recordings = Recordings(
        [str(path) for path in fpaths],
        clustering_name=clustering_name,
        no_waveforms=not load_waveforms,
        continuous_data_type=continuous_data_type,
        verbose=verbose,
        **recordings_kwargs,
    )
    if correct_repeated_a_id:
        _rename_last_recording_a2(recordings)
    if load_analysis:
        recordings.load_analysis()
    return recordings


def load_barrylab_recordings_for_rat(
    root: str | Path,
    rat_id: str,
    **kwargs: Any,
):
    """Load all recordings from the first day of one rat using the original loader."""
    return load_barrylab_recordings(
        get_paths_to_rat_recordings_on_first_day(root, rat_id),
        **kwargs,
    )


def preprocess_recordings_unit_analysis(
    recordings,
    *,
    recompute: bool = True,
    compute_autocorrelations: bool = True,
    verbose: bool = False,
):
    """Run only original Barry lab preprocessing functions for unit/position analysis.

    This intentionally delegates analysis work to
    `barrylab_ephys_analysis.scripts.exp_scales.paper_preprocess`.
    It does not run LFP spectral overview, theta signals, or Bayes decoding.

    Note: the original `assign_unit_categories_if_not_available` includes
    duplicate place-cell removal via `set_duplicate_category_to_noise`.
    """
    paper_preprocess = _import_barrylab_modules(include_preprocess=True)["paper_preprocess"]

    if compute_autocorrelations:
        paper_preprocess.compute_unit_autocorrelations_if_not_available(
            recordings=recordings,
            recompute=recompute,
            verbose=verbose,
        )
    paper_preprocess.compute_waveform_properties_and_sorting_quality_if_not_available(
        recordings=recordings,
        recompute=recompute,
        verbose=verbose,
    )
    paper_preprocess.compute_ratemap_speed_mask_if_not_available(
        recordings=recordings,
        recompute=recompute,
        verbose=verbose,
    )
    paper_preprocess.compute_ratemaps_if_not_available(
        recordings=recordings,
        recompute=recompute,
        verbose=verbose,
    )
    paper_preprocess.compute_ratemap_stability_if_not_available(
        recordings=recordings,
        recompute=recompute,
        verbose=verbose,
    )
    paper_preprocess.detect_fields_if_not_available(
        recordings=recordings,
        recompute=recompute,
        verbose=verbose,
    )
    paper_preprocess.assign_unit_categories_if_not_available(
        recordings=recordings,
        recompute=recompute,
        verbose=verbose,
    )
    return recordings


def load_preprocessed_experiments_for_rat(
    root: str | Path,
    rat_id: str,
    *,
    recompute: bool = True,
    compute_autocorrelations: bool = True,
    verbose: bool = False,
    **recordings_kwargs: Any,
) -> list[ExperimentData]:
    """Load with original Barry lab code, preprocess there, then convert to dataclasses."""
    recordings = load_barrylab_recordings_for_rat(
        root,
        rat_id,
        load_waveforms=True,
        verbose=verbose,
        **recordings_kwargs,
    )
    preprocess_recordings_unit_analysis(
        recordings,
        recompute=recompute,
        compute_autocorrelations=compute_autocorrelations,
        verbose=verbose,
    )
    return recordings_to_experiments(recordings)


def recordings_to_experiments(recordings) -> list[ExperimentData]:
    """Convert original `Recordings` output into MultiplePFAnalysis dataclasses."""
    return [_recording_to_experiment(recording) for recording in recordings]


def _recording_to_experiment(recording) -> ExperimentData:
    rat_id, day, session = parse_experiment_identity(Path(recording.fpath))
    units = [_unit_to_dataclass(unit) for unit in recording.units]
    return ExperimentData(
        path=Path(recording.fpath),
        rat_id=rat_id,
        day=day,
        session=session,
        info=deepcopy(recording.info),
        position=_position_to_dataclass(recording.position),
        tetrode_spikes=_tetrode_spikes_to_dataclasses(recording),
        units=units,
        unit_lookup=_unit_lookup_to_dataclass(recording.unit_lookup_table),
        analysis=deepcopy(recording.analysis),
    )


def _position_to_dataclass(position: dict[str, Any] | None) -> PositionData | None:
    if position is None:
        return None
    return PositionData(
        timestamps=np.asarray(position["timestamps"]),
        xy=np.asarray(position["xy"]),
        speed=np.asarray(position["speed"]),
        head_direction=None,
        movement_direction=np.asarray(position["movement_direction"]),
        sampling_rate=int(position["sampling_rate"]),
        second_led_xy=None,
        analysis=deepcopy(position.get("analysis", {})),
    )


def _unit_to_dataclass(unit: dict[str, Any]) -> UnitData:
    return UnitData(
        tetrode_nr=int(unit["tetrode_nr"]),
        cluster_id=int(unit["tetrode_cluster_id"]),
        channel_group=unit.get("channel_group"),
        timestamps=np.asarray(unit["timestamps"]),
        waveforms=None if unit.get("waveforms") is None else np.asarray(unit["waveforms"]),
        sampling_rate=float(unit["sampling_rate"]),
        analysis=deepcopy(unit.get("analysis", {})),
    )


def _unit_lookup_to_dataclass(unit_lookup_table: dict[Any, dict[Any, Any]]) -> dict[int, dict[int, int]]:
    return {
        int(tetrode_nr): {int(cluster_id): int(unit_index) for cluster_id, unit_index in cluster_lookup.items()}
        for tetrode_nr, cluster_lookup in unit_lookup_table.items()
    }


def _tetrode_spikes_to_dataclasses(recording) -> list[TetrodeSpikeData]:
    units_by_tetrode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for unit in recording.units:
        units_by_tetrode[int(unit["tetrode_nr"])].append(unit)

    tetrode_spikes = []
    for tetrode_nr in sorted(units_by_tetrode):
        units = units_by_tetrode[tetrode_nr]
        timestamps = np.concatenate([np.asarray(unit["timestamps"]) for unit in units])
        cluster_ids = np.concatenate(
            [
                np.full(len(unit["timestamps"]), int(unit["tetrode_cluster_id"]), dtype=np.int16)
                for unit in units
            ]
        )
        waveforms = None
        if all(unit.get("waveforms") is not None for unit in units):
            waveforms = np.concatenate([np.asarray(unit["waveforms"]) for unit in units], axis=0)

        order = np.argsort(timestamps)
        timestamps = timestamps[order]
        cluster_ids = cluster_ids[order]
        if waveforms is not None:
            waveforms = waveforms[order]

        tetrode_spikes.append(
            TetrodeSpikeData(
                tetrode_nr=tetrode_nr,
                channel_group=units[0].get("channel_group"),
                timestamps=timestamps,
                waveforms=waveforms,
                cluster_ids=cluster_ids,
                idx_keep=None,
            )
        )

    return tetrode_spikes
