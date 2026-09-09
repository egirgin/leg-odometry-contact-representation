"""Loads an OCELOT-style recording directory: lowstate.csv (+ optional groundtruth.csv, frames/).

lowstate.csv must carry IMU + joint columns; groundtruth.csv, if present, carries pos_x/y/z.
Foot-force columns are read opportunistically but never required — the detectors in this repo
are kinematics-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leg_odom.io.columns import (
    FOOT_FORCE_COLS,
    IMU_CORE_COLS,
    TIME_NANOSEC_COL,
    TIME_SEC_COL,
    motor_position_cols,
    motor_torque_cols,
    motor_velocity_cols,
)
from leg_odom.io.imu_sanitize import infer_accel_gravity_compensated, sanitize_imu_dataframe
from leg_odom.io.timebase import build_timebase, estimate_median_sample_rate_hz


def _required_ocelot_columns() -> tuple[str, ...]:
    return (
        TIME_SEC_COL,
        TIME_NANOSEC_COL,
        *IMU_CORE_COLS,
        *motor_position_cols(),
        *motor_velocity_cols(),
        *motor_torque_cols(),
    )


def _coerce_numeric(df: pd.DataFrame, columns: list[str], *, required: bool) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if required:
        bad = [c for c in columns if not np.isfinite(out[c].to_numpy(dtype=np.float64)).all()]
        if bad:
            raise ValueError("non-finite values in required columns: " + ", ".join(bad))
    return out


def discover_ocelot_csv_path(sequence_dir: str | Path) -> Path:
    root = Path(sequence_dir).expanduser().resolve()
    p = root / "lowstate.csv"
    if not p.is_file():
        raise FileNotFoundError(f"missing lowstate.csv under {root}")
    return p


def load_prepared_ocelot(
    sequence_dir: str | Path,
    *,
    verbose: bool = False,
    sanitize_imu: bool = True,
) -> tuple[pd.DataFrame, float, pd.DataFrame, bool, dict[str, Any]]:
    """Returns (frames_df, median_hz, gt_df, accel_gravity_compensated, meta)."""
    root = Path(sequence_dir).expanduser().resolve()
    csv_path = discover_ocelot_csv_path(root)
    gt_path = root / "groundtruth.csv"
    frames_dir = root / "frames"

    df = pd.read_csv(csv_path)
    required = list(_required_ocelot_columns())
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"lowstate.csv missing required columns: {', '.join(missing)}")
    df = _coerce_numeric(df, required, required=True)

    present_force_cols = [c for c in FOOT_FORCE_COLS if c in df.columns]
    if present_force_cols:
        df = _coerce_numeric(df, present_force_cols, required=False)

    df = df.sort_values([TIME_SEC_COL, TIME_NANOSEC_COL]).reset_index(drop=True)
    build_timebase(df)
    hz = estimate_median_sample_rate_hz(df["dt"])

    if sanitize_imu:
        df, accel_gc = sanitize_imu_dataframe(df, verbose=verbose)
    else:
        accel_gc = infer_accel_gravity_compensated(df)

    gt_df = pd.DataFrame()
    if gt_path.is_file():
        gt_raw = pd.read_csv(gt_path).rename(columns={"pos_x": "local_x", "pos_y": "local_y", "pos_z": "local_z"})
        if TIME_SEC_COL in gt_raw.columns and TIME_NANOSEC_COL in gt_raw.columns:
            gt_t = gt_raw[TIME_SEC_COL].astype(float) + gt_raw[TIME_NANOSEC_COL].astype(float) * 1e-9
            t0 = float(df[TIME_SEC_COL].iloc[0]) + float(df[TIME_NANOSEC_COL].iloc[0]) * 1e-9
            gt_raw["t_abs"] = gt_t - t0
            keep = [c for c in ("local_x", "local_y", "local_z", "t_abs") if c in gt_raw.columns]
            gt_df = gt_raw[keep]

    meta = {
        "has_groundtruth_csv": gt_path.is_file(),
        "has_frames_dir": frames_dir.is_dir(),
        "frames_png_count": len(list(frames_dir.glob("*.png"))) if frames_dir.is_dir() else 0,
    }
    if verbose:
        print(f"[io] ocelot recording: rows={len(df)}, hz~{hz:.2f}, gt={meta['has_groundtruth_csv']}")
    return df, hz, gt_df, accel_gc, meta
