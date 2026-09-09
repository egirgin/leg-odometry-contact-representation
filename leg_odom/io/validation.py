"""Sanity checks on a prepared sequence frame before it's used."""

from __future__ import annotations

import numpy as np
import pandas as pd

from leg_odom.io.columns import IMU_CORE_COLS, motor_position_cols


def validate_prepared_split_dataframe(dataframe: pd.DataFrame, *, require_joint_positions: bool = True) -> None:
    for col in ("t_abs", "dt", *IMU_CORE_COLS):
        if col not in dataframe.columns:
            raise KeyError(f"missing column {col!r}")

    if require_joint_positions:
        for col in motor_position_cols():
            if col not in dataframe.columns:
                raise KeyError(f"missing joint column {col!r}")

    t = dataframe["t_abs"].to_numpy(dtype=float)
    if not np.all(np.diff(t) >= 0):
        raise ValueError("t_abs is not non-decreasing")

    dt = dataframe["dt"].to_numpy(dtype=float)
    if not np.isfinite(dt).all() or (dt <= 0).any():
        raise ValueError("dt must be finite and positive")

    for col in IMU_CORE_COLS:
        if not np.isfinite(dataframe[col].to_numpy(dtype=float)).all():
            raise ValueError(f"non-finite values in {col}")
