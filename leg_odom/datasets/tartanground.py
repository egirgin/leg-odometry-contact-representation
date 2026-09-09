"""TartanGround sequences: imu.csv + one *_bag.csv per trajectory directory.

This is a data product (dataset.kind: tartanground), not a robot model — kinematics are
selected separately via robot.kinematics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from leg_odom.datasets.single_sequence import CachedSingleSequenceDataset
from leg_odom.io.split_imu_bag import load_prepared_split_sequence


class TartangroundDataset(CachedSingleSequenceDataset):
    def __init__(
        self,
        root: str | Path,
        *,
        verbose: bool = False,
        sanitize_imu: bool = True,
        validate: bool = True,
        extra_meta: dict[str, Any] | None = None,
        preload: bool = True,
    ) -> None:
        em = dict(extra_meta or {})
        em.setdefault("dataset", "tartanground")
        super().__init__(
            root, verbose=verbose, sanitize_imu=sanitize_imu, validate=validate, extra_meta=em, preload=preload
        )

    def _require_sequence_directory(self, root: Path) -> Path:
        root = root.expanduser().resolve()
        if not (root / "imu.csv").is_file():
            raise FileNotFoundError(f"dataset.sequence_dir must contain imu.csv, got {root}")
        return root

    def _load_prepared(self):
        df, hz, gt, accel_gc = load_prepared_split_sequence(
            self._sequence_dir, verbose=self._verbose, sanitize_imu=self._sanitize_imu
        )
        return df, hz, gt, accel_gc, {}
