"""IMU sanity checks for FLU body-frame exports.

Expected default: the accelerometer reports specific force (gravity included), so it reads
about +9.81 m/s^2 on body +Z when the robot is level and stationary. The tolerated alternative
is gravity-compensated (pure linear) acceleration, in which case the log must also carry a body
orientation quaternion so initial attitude doesn't depend on gravity being in the accel channel.
No axis remapping is attempted; data that doesn't match either case raises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from leg_odom.io.columns import IMU_ACCEL_COLS, IMU_BODY_QUAT_COLS, IMU_CORE_COLS
from leg_odom.thresholds import (
    IMU_FLU_SPECIFIC_FORCE_MAG_MAX,
    IMU_FLU_SPECIFIC_FORCE_MAG_MIN,
    IMU_FLU_SPECIFIC_FORCE_MAX_TILT_DEG,
    IMU_GRAVITY_REMOVED_MEAN_MAG_THRESHOLD,
    IMU_GYRO_MEDIAN_NORM_DEG_S_HINT,
    IMU_VECTOR_NEAR_ZERO_NORM,
)


def _angle_to_body_plus_z_deg(v: np.ndarray) -> float:
    up = np.array([0.0, 0.0, 1.0])
    norm_v = float(np.linalg.norm(v))
    if norm_v < IMU_VECTOR_NEAR_ZERO_NORM:
        return float("nan")
    cosang = float(np.clip(np.dot(v / norm_v, up), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _has_body_orientation_quaternion(dataframe: pd.DataFrame) -> bool:
    return all(c in dataframe.columns for c in IMU_BODY_QUAT_COLS)


def infer_accel_gravity_compensated(dataframe: pd.DataFrame) -> bool:
    """If mean |accel| is far below g, gravity was likely already removed."""
    mag = np.linalg.norm(dataframe[list(IMU_ACCEL_COLS)].to_numpy(dtype=float), axis=1)
    return bool(float(np.mean(mag)) < IMU_GRAVITY_REMOVED_MEAN_MAG_THRESHOLD)


def _assert_flu_specific_force(dataframe: pd.DataFrame, *, verbose: bool) -> None:
    accel = dataframe[list(IMU_ACCEL_COLS)].to_numpy(dtype=float)
    a_mean = accel.mean(axis=0)
    norm = float(np.linalg.norm(a_mean))
    tilt = _angle_to_body_plus_z_deg(a_mean)

    if not np.isfinite(tilt):
        raise ValueError("IMU FLU check failed: mean acceleration has near-zero norm")

    if float(a_mean[2]) <= 0.0:
        raise ValueError(
            f"IMU FLU check failed: mean accel_z must be positive (got {float(a_mean[2]):.4f} m/s^2); "
            "data must be FLU with gravity along body +Z at rest"
        )

    if not (IMU_FLU_SPECIFIC_FORCE_MAG_MIN <= norm <= IMU_FLU_SPECIFIC_FORCE_MAG_MAX):
        raise ValueError(
            f"IMU FLU check failed: mean |accel| is {norm:.2f} m/s^2, expected roughly "
            f"{IMU_FLU_SPECIFIC_FORCE_MAG_MIN}-{IMU_FLU_SPECIFIC_FORCE_MAG_MAX} m/s^2 at rest "
            f"(use gravity-compensated accel + orientation columns {IMU_BODY_QUAT_COLS} instead)"
        )

    if tilt > IMU_FLU_SPECIFIC_FORCE_MAX_TILT_DEG:
        raise ValueError(
            f"IMU FLU check failed: mean acceleration tilts {tilt:.1f} deg from body +Z "
            f"(max {IMU_FLU_SPECIFIC_FORCE_MAX_TILT_DEG} deg)"
        )

    if verbose:
        print(f"[IMU] FLU specific-force check OK (mean |a|={norm:.2f} m/s^2, tilt={tilt:.1f} deg)")


def _assert_gravity_compensated_with_orientation(dataframe: pd.DataFrame, *, verbose: bool) -> None:
    if not _has_body_orientation_quaternion(dataframe):
        raise ValueError(
            "IMU appears gravity-compensated (mean |accel| is small) but is missing orientation "
            f"columns {IMU_BODY_QUAT_COLS}"
        )
    if verbose:
        print("[IMU] gravity-compensated acceleration detected, orientation columns present")


def sanitize_imu_dataframe(dataframe: pd.DataFrame, *, verbose: bool = False) -> tuple[pd.DataFrame, bool]:
    """Validates FLU semantics and converts gyro units to rad/s if they look like deg/s.

    Returns (dataframe, accel_gravity_compensated).
    """
    req = list(IMU_CORE_COLS)
    if not all(c in dataframe.columns for c in req):
        raise KeyError(f"CSV missing required IMU columns: {req}")

    gyro = dataframe[list(IMU_CORE_COLS[:3])].to_numpy(dtype=float)
    median_norm = float(np.median(np.linalg.norm(gyro, axis=1)))
    if median_norm > IMU_GYRO_MEDIAN_NORM_DEG_S_HINT:
        dataframe[list(IMU_CORE_COLS[:3])] *= np.pi / 180.0
        if verbose:
            print("[IMU] converted gyro units from deg/s to rad/s")

    accel_gravity_compensated = infer_accel_gravity_compensated(dataframe)
    if accel_gravity_compensated:
        _assert_gravity_compensated_with_orientation(dataframe, verbose=verbose)
    else:
        _assert_flu_specific_force(dataframe, verbose=verbose)

    return dataframe, accel_gravity_compensated
