"""A single sanity-check figure per run: estimated vs. ground-truth trajectory."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from leg_odom.io.columns import TIME_NANOSEC_COL, TIME_SEC_COL


def _hist_time(hist: pd.DataFrame) -> np.ndarray:
    if "t_abs" in hist.columns:
        return hist["t_abs"].to_numpy(dtype=np.float64)
    return hist[TIME_SEC_COL].to_numpy(dtype=np.float64) + hist[TIME_NANOSEC_COL].to_numpy(dtype=np.float64) * 1e-9


def _gt_time(gt_df: pd.DataFrame) -> np.ndarray:
    if "t_abs" in gt_df.columns:
        return gt_df["t_abs"].to_numpy(dtype=np.float64)
    return gt_df[TIME_SEC_COL].to_numpy(dtype=np.float64) + gt_df[TIME_NANOSEC_COL].to_numpy(dtype=np.float64) * 1e-9


def plot_trajectory(hist: pd.DataFrame, gt_df: pd.DataFrame | None, out_path: Path) -> None:
    """Top-down XY path plus vertical position over time, both relative to the start."""
    if hist.empty:
        return
    t = _hist_time(hist)
    px, py, pz = (hist[c].to_numpy(dtype=np.float64) for c in ("p_x", "p_y", "p_z"))
    ex, ey = px - px[0], py - py[0]

    fig, (ax_xy, ax_z) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax_xy.plot(ex, ey, label="Estimated", color="C0", linewidth=2)
    ax_xy.plot(0, 0, "ko", markersize=8, label="Start")

    has_gt = gt_df is not None and not gt_df.empty and {"local_x", "local_y"}.issubset(gt_df.columns)
    if has_gt:
        gx, gy = gt_df["local_x"].to_numpy(dtype=np.float64), gt_df["local_y"].to_numpy(dtype=np.float64)
        gxs, gys = gx - gx[0], gy - gy[0]
        ax_xy.plot(gxs, gys, label="Ground truth", color="black", linestyle="--", linewidth=2)

    ax_xy.set_xlabel("X [m]")
    ax_xy.set_ylabel("Y [m]")
    ax_xy.set_title("Top-down trajectory")
    ax_xy.grid(True, alpha=0.4)
    ax_xy.legend(loc="best")
    ax_xy.set_aspect("equal")

    ax_z.plot(t, pz, label="Estimated z", color="C0", linewidth=1.5)
    if has_gt and "local_z" in gt_df.columns:
        gt_t = _gt_time(gt_df)
        gz_i = np.interp(t, gt_t, gt_df["local_z"].to_numpy(dtype=np.float64), left=np.nan, right=np.nan)
        ax_z.plot(t, gz_i, label="Ground truth z", color="black", linestyle="--", linewidth=1.5, alpha=0.85)
    ax_z.set_xlabel("Time [s]")
    ax_z.set_ylabel("z [m]")
    ax_z.set_title("Vertical position")
    ax_z.grid(True, alpha=0.4)
    ax_z.legend(loc="best")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
