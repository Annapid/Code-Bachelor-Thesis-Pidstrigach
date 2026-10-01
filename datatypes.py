from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PositionData:
    timestamps: np.ndarray
    xy: np.ndarray
    speed: np.ndarray
    head_direction: np.ndarray | None
    movement_direction: np.ndarray
    sampling_rate: int
    second_led_xy: np.ndarray | None = None
    analysis: dict[str, Any] = field(default_factory=dict)


@dataclass
class TetrodeSpikeData:
    tetrode_nr: int
    channel_group: str | None
    timestamps: np.ndarray
    waveforms: np.ndarray | None
    cluster_ids: np.ndarray | None
    idx_keep: np.ndarray | None


@dataclass
class UnitData:
    tetrode_nr: int
    cluster_id: int
    channel_group: str | None
    timestamps: np.ndarray
    waveforms: np.ndarray | None
    sampling_rate: float
    analysis: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentData:
    path: Path
    rat_id: str
    day: str
    session: str
    info: dict[str, Any]
    position: PositionData | None
    tetrode_spikes: list[TetrodeSpikeData]
    units: list[UnitData]
    unit_lookup: dict[int, dict[int, int]] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)

    @property
    def experiment_key(self) -> str:
        return f"{self.day}/{self.session}"

    def get_unit(self, tetrode_nr: int, cluster_id: int) -> UnitData | None:
        unit_index = self.unit_lookup.get(tetrode_nr, {}).get(cluster_id)
        if unit_index is None:
            return None
        return self.units[unit_index]
