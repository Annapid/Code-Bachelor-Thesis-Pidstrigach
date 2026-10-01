from __future__ import annotations

from pathlib import Path


def parse_experiment_identity(path: Path) -> tuple[str, str, str]:
    if len(path.parents) < 3:
        raise ValueError(f"Path does not match rat/day/session layout: {path}")
    rat_id = path.parents[2].name
    day = path.parents[1].name
    session = path.parents[0].name
    return rat_id, day, session
