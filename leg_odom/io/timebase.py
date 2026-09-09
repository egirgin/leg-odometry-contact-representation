"""Builds t_abs (seconds since first sample) and dt on a merged log frame."""

from __future__ import annotations

import numpy as np
import pandas as pd

from leg_odom.io.columns import TIME_NANOSEC_COL, TIME_SEC_COL
from leg_odom.thresholds import (
    TIMEBASE_DT_CLIP_MAX_S,
    TIMEBASE_DT_CLIP_MIN_S,
    TIMEBASE_MIN_POSITIVE_DT_SAMPLES,
    TIMEBASE_RATE_FALLBACK_HZ,
)


def estimate_median_sample_rate_hz(dt_column: pd.Series) -> float:
    dt = dt_column.to_numpy(dtype=float)
    dt = dt[dt > 0.0]
    if len(dt) <= TIMEBASE_MIN_POSITIVE_DT_SAMPLES:
        return TIMEBASE_RATE_FALLBACK_HZ
    return 1.0 / float(np.median(dt))


def build_timebase(dataframe: pd.DataFrame) -> None:
    """Mutates dataframe in place, adding t_abs and dt from 'sec'/'nanosec' columns."""
    if TIME_SEC_COL not in dataframe.columns or TIME_NANOSEC_COL not in dataframe.columns:
        raise KeyError(f"expected columns '{TIME_SEC_COL}' and '{TIME_NANOSEC_COL}'")

    s = pd.to_numeric(dataframe[TIME_SEC_COL], errors="coerce").astype(float)
    ns = pd.to_numeric(dataframe[TIME_NANOSEC_COL], errors="coerce").astype(float)
    t = s + ns * 1e-9
    dataframe["t_abs"] = t - float(t.iloc[0])

    dt = dataframe["t_abs"].diff().fillna(0.0).to_numpy(dtype=np.float64, copy=True)
    dt[dt <= 0.0] = np.nan
    fallback_dt = 1.0 / TIMEBASE_RATE_FALLBACK_HZ
    median_dt = float(np.nanmedian(dt[np.isfinite(dt)])) if np.isfinite(dt).any() else fallback_dt
    dt = np.nan_to_num(dt, nan=median_dt)
    dataframe["dt"] = np.clip(dt, TIMEBASE_DT_CLIP_MIN_S, TIMEBASE_DT_CLIP_MAX_S)
