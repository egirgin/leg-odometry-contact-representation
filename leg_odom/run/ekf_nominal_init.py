"""Seeds the EKF's nominal state from the first fully-observed row of the merged timeline.

Only used when ekf.initialize_nominal_from_data is true; otherwise the filter starts at
zero position/velocity and identity attitude.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from leg_odom.filters.esekf import ErrorStateEkf
from leg_odom.io.columns import IMU_BODY_QUAT_COLS

_POS_COL_GROUPS = (("pos_x", "pos_y", "pos_z"), ("p_x", "p_y", "p_z"))
_VEL_COL_GROUPS = (("vel_x", "vel_y", "vel_z"), ("v_x", "v_y", "v_z"))
_BIAS_ACCEL_COLS = ("bax", "bay", "baz")
_BIAS_GYRO_COLS = ("bgx", "bgy", "bgz")


def _pick_xyz_group(columns, groups, label: str) -> tuple[str, str, str]:
    colset = set(columns)
    for g in groups:
        if all(c in colset for c in g):
            return g
    raise ValueError(f"ekf.initialize_nominal_from_data: merged timeline lacks {label} (expected one of {list(groups)})")


def _row_all_finite(row: pd.Series, names: tuple[str, ...]) -> bool:
    return all(math.isfinite(float(row[c])) for c in names)


def _first_valid_row_index(df: pd.DataFrame, needed: tuple[str, ...]) -> int:
    for i in range(len(df)):
        if _row_all_finite(df.iloc[i], needed):
            return i
    raise ValueError(f"ekf.initialize_nominal_from_data: no row has finite values for all of {needed}")


def apply_nominal_init_from_timeline(ekf: ErrorStateEkf, timeline: pd.DataFrame) -> None:
    pos = _pick_xyz_group(timeline.columns, _POS_COL_GROUPS, "world position (m)")
    vel = _pick_xyz_group(timeline.columns, _VEL_COL_GROUPS, "world velocity (m/s)")
    for c in IMU_BODY_QUAT_COLS:
        if c not in timeline.columns:
            raise ValueError(f"ekf.initialize_nominal_from_data requires body quaternion columns {IMU_BODY_QUAT_COLS}")

    idx = _first_valid_row_index(timeline, pos + vel + IMU_BODY_QUAT_COLS)
    row = timeline.iloc[idx]

    p = np.array([row[c] for c in pos], dtype=np.float64)
    v = np.array([row[c] for c in vel], dtype=np.float64)
    q = np.array([row[c] for c in IMU_BODY_QUAT_COLS], dtype=np.float64)
    R = Rotation.from_quat(q).as_matrix()

    ba = np.array([row[c] for c in _BIAS_ACCEL_COLS]) if all(c in timeline.columns for c in _BIAS_ACCEL_COLS) and _row_all_finite(row, _BIAS_ACCEL_COLS) else None
    bg = np.array([row[c] for c in _BIAS_GYRO_COLS]) if all(c in timeline.columns for c in _BIAS_GYRO_COLS) and _row_all_finite(row, _BIAS_GYRO_COLS) else None

    ekf.seed_nominal_state(p=p, v=v, R=R, bias_accel=ba, bias_gyro=bg)


def ekf_initialize_nominal_from_data_enabled(cfg: Mapping[str, Any] | None) -> bool:
    if not isinstance(cfg, Mapping):
        return False
    return bool(cfg.get("ekf", {}).get("initialize_nominal_from_data", False))
