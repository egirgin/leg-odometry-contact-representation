"""Loads a TartanGround-style split layout: imu.csv + one *_bag.csv (kinematics), asof-merged."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from leg_odom.io.columns import TIME_NANOSEC_COL, TIME_SEC_COL
from leg_odom.io.ground_truth import extract_position_ground_truth
from leg_odom.io.imu_sanitize import infer_accel_gravity_compensated, sanitize_imu_dataframe
from leg_odom.io.timebase import build_timebase, estimate_median_sample_rate_hz


def discover_bag_csv_path(sequence_dir: Path) -> Path:
    candidates = sorted(sequence_dir.glob("*_bag.csv"))
    if not candidates:
        raise FileNotFoundError(f"no '*_bag.csv' under {sequence_dir}")
    return candidates[0]


def merge_split_imu_bag(sequence_dir: str | Path, *, verbose: bool = False) -> pd.DataFrame:
    """Aligns both CSVs' time origins to zero and asof-merges kinematics onto IMU timestamps."""
    root = Path(sequence_dir).expanduser().resolve()
    imu_path = root / "imu.csv"
    if not imu_path.is_file():
        raise FileNotFoundError(f"missing imu.csv under {root}")
    kin_path = discover_bag_csv_path(root)
    if verbose:
        print(f"[io] split layout: {root}, kinematics file: {kin_path.name}")

    imu_df = pd.read_csv(imu_path)
    kin_df = pd.read_csv(kin_path)
    for name, df in (("imu.csv", imu_df), ("*_bag.csv", kin_df)):
        if TIME_SEC_COL not in df.columns or TIME_NANOSEC_COL not in df.columns:
            raise KeyError(f"{name} must contain '{TIME_SEC_COL}' and '{TIME_NANOSEC_COL}'")

    for df in (imu_df, kin_df):
        t = df[TIME_SEC_COL].astype(float) + df[TIME_NANOSEC_COL].astype(float) * 1e-9
        df["t_abs"] = t - float(t.iloc[0])
    imu_df = imu_df.sort_values("t_abs").reset_index(drop=True)
    kin_df = kin_df.sort_values("t_abs").reset_index(drop=True)

    drop_from_kin = [c for c in (TIME_SEC_COL, TIME_NANOSEC_COL, "time") if c in kin_df.columns]
    merged = pd.merge_asof(
        imu_df, kin_df.drop(columns=drop_from_kin, errors="ignore"), on="t_abs", direction="backward"
    )

    kin_only_cols = [c for c in kin_df.columns if c not in imu_df.columns and c != "t_abs"]
    for col in kin_only_cols:
        if col in merged.columns:
            merged[col] = merged[col].ffill().bfill()

    if verbose:
        print(f"[io] IMU rows={len(imu_df)}, kin rows={len(kin_df)}, merged={len(merged)}")
    return merged


def load_prepared_split_sequence(
    sequence_dir: str | Path, *, verbose: bool = False, sanitize_imu: bool = True
) -> tuple[pd.DataFrame, float, pd.DataFrame, bool]:
    """Returns (dataframe, median_hz, position_gt, accel_gravity_compensated)."""
    df = merge_split_imu_bag(sequence_dir, verbose=verbose)
    build_timebase(df)
    hz = estimate_median_sample_rate_hz(df["dt"])
    if sanitize_imu:
        df, accel_gc = sanitize_imu_dataframe(df, verbose=verbose)
    else:
        accel_gc = infer_accel_gravity_compensated(df)
    gt = extract_position_ground_truth(df)
    if verbose:
        print(f"[io] median sample rate ~ {hz:.2f} Hz, ground truth present={not gt.empty}")
    return df, hz, gt, accel_gc
