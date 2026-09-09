"""Trajectory metrics against embedded ground truth: ATE, AHE, RPE, FPE, discrete Frechet.

Evaluation time base prefers t_abs on the EKF history when present, falling back to
sec+nanosec, matching how ground truth is exposed by leg_odom.io.ground_truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from leg_odom.io.columns import TIME_NANOSEC_COL, TIME_SEC_COL

EVALUATION_CSV_COLUMNS: tuple[str, ...] = (
    "sequence_name", "skipped",
    "ate_m", "ate_x_m", "ate_y_m", "ate_z_m",
    "ahe_deg", "rpe_trans_pct", "rpe_rot_deg_per_m",
    "fpe_m", "drift_pct", "length_err", "frechet_m", "gt_length_m", "est_length_m",
)


def compute_path_dist(data: np.ndarray) -> np.ndarray:
    diffs = np.diff(data[:, :2], axis=0)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    return np.concatenate(([0], np.cumsum(dists)))


def extract_spatial_headings(pos: np.ndarray, min_dist: float = 0.1) -> np.ndarray:
    """Headings from chords spaced at least min_dist apart -- avoids atan2 noise blowup at low speed."""
    dists = compute_path_dist(pos)
    headings = np.zeros(len(pos))
    last_idx = 0
    for i in range(1, len(pos)):
        if dists[i] - dists[last_idx] >= min_dist:
            dx, dy = pos[i, 0] - pos[last_idx, 0], pos[i, 1] - pos[last_idx, 1]
            headings[last_idx:i] = np.arctan2(dy, dx)
            last_idx = i
    if last_idx < len(pos):
        headings[last_idx:] = headings[last_idx - 1] if last_idx > 0 else 0.0
    return np.unwrap(headings)


def calculate_absolute_heading_error(gt_h: np.ndarray, est_h: np.ndarray) -> float:
    if len(gt_h) < 2 or len(est_h) < 2:
        return 0.0
    diffs = (est_h - gt_h + np.pi) % (2 * np.pi) - np.pi
    return float(np.degrees(np.sqrt(np.mean(diffs**2))))


def match_predictions_to_gt(gt_t: np.ndarray, gt_pos: np.ndarray, est_t: np.ndarray, est_pos: np.ndarray) -> np.ndarray:
    interp_func = interp1d(est_t, est_pos, axis=0, kind="linear", fill_value="extrapolate")
    return np.asarray(interp_func(gt_t), dtype=np.float64)


def calculate_ate_rmse_synced(gt_pos: np.ndarray, matched_est_pos: np.ndarray) -> float:
    errors = np.linalg.norm(gt_pos - matched_est_pos, axis=1)
    return float(np.sqrt(np.mean(errors**2)))


def calculate_per_axis_rmse(gt_pos: np.ndarray, matched_est_pos: np.ndarray) -> tuple[float, float, float]:
    """z is NaN when positions are 2D."""
    ex, ey = gt_pos[:, 0] - matched_est_pos[:, 0], gt_pos[:, 1] - matched_est_pos[:, 1]
    rx, ry = float(np.sqrt(np.mean(ex**2))), float(np.sqrt(np.mean(ey**2)))
    if matched_est_pos.shape[1] >= 3 and gt_pos.shape[1] >= 3:
        ez = gt_pos[:, 2] - matched_est_pos[:, 2]
        return rx, ry, float(np.sqrt(np.mean(ez**2)))
    return rx, ry, float("nan")


def calculate_ate_norm_rmse_3d(gt_pos: np.ndarray, matched_est_pos: np.ndarray) -> float:
    if gt_pos.shape[1] < 3 or matched_est_pos.shape[1] < 3:
        return float("nan")
    err = gt_pos[:, :3] - matched_est_pos[:, :3]
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def calculate_rpe_metrics_synced(
    gt_pos: np.ndarray, matched_est_pos: np.ndarray, gt_h: np.ndarray, est_h: np.ndarray, window_m: float = 1.0
) -> tuple[float, float]:
    """RPE over a fixed traveled-distance window (not a fixed number of samples)."""
    gt_dist = compute_path_dist(gt_pos)
    trans_errors_sq: list[float] = []
    rot_errors_sq: list[float] = []

    for i in range(len(gt_pos)):
        candidates = np.where(gt_dist > (gt_dist[i] + window_m))[0]
        if len(candidates) == 0:
            break
        j = int(candidates[0])
        actual_dist = gt_dist[j] - gt_dist[i]
        if actual_dist < 1e-3:
            continue
        trans_error = float(np.linalg.norm((matched_est_pos[j] - matched_est_pos[i]) - (gt_pos[j] - gt_pos[i])))
        trans_errors_sq.append(((trans_error / actual_dist) * 100) ** 2)
        rot_err = (est_h[j] - est_h[i] - (gt_h[j] - gt_h[i]) + np.pi) % (2 * np.pi) - np.pi
        rot_errors_sq.append((np.degrees(np.abs(rot_err)) / actual_dist) ** 2)

    if not trans_errors_sq:
        return 0.0, 0.0
    return float(np.sqrt(np.mean(trans_errors_sq))), float(np.sqrt(np.mean(rot_errors_sq)))


def discrete_frechet(P: np.ndarray, Q: np.ndarray) -> float:
    n, m = len(P), len(Q)
    if n == 0 or m == 0:
        return 0.0
    ca = np.full((n, m), -1.0)
    dist_matrix = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
    ca[0, 0] = dist_matrix[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], dist_matrix[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], dist_matrix[0, j])
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]), dist_matrix[i, j])
    return float(ca[n - 1, m - 1])


def calculate_shape_metrics(gt_resampled: np.ndarray, est_resampled: np.ndarray, gt_total_len: float, est_total_len: float) -> tuple[float, float]:
    ratio = est_total_len / gt_total_len if gt_total_len > 0 else 0.0
    return abs(1.0 - ratio), discrete_frechet(gt_resampled, est_resampled)


def resample_spatially(data: np.ndarray, step: float = 0.1) -> tuple[np.ndarray, float]:
    """Resamples a path at fixed arc-length intervals, so Frechet distance isn't skewed by
    stretches where the robot moved slowly and left many closely-spaced samples."""
    dists = compute_path_dist(data)
    total_dist = float(dists[-1])
    if total_dist == 0:
        return data, 0.0
    _, unique_idx = np.unique(dists, return_index=True)
    if len(unique_idx) < 2:
        return data, total_dist
    new_dists = np.arange(0, total_dist, step)
    interp_func = interp1d(dists[unique_idx], data[unique_idx, :2], axis=0, fill_value="extrapolate")
    return np.asarray(interp_func(new_dists), dtype=np.float64), total_dist


def _est_time_seconds(hist: pd.DataFrame) -> np.ndarray:
    if "t_abs" in hist.columns:
        return hist["t_abs"].to_numpy(dtype=np.float64)
    if TIME_SEC_COL in hist.columns and TIME_NANOSEC_COL in hist.columns:
        return hist[TIME_SEC_COL].to_numpy(dtype=np.float64) + hist[TIME_NANOSEC_COL].to_numpy(dtype=np.float64) * 1e-9
    raise ValueError(f"EKF history needs t_abs or {TIME_SEC_COL!r}/{TIME_NANOSEC_COL!r}")


def _gt_time_seconds(gt_df: pd.DataFrame) -> np.ndarray:
    if "t_abs" in gt_df.columns:
        return gt_df["t_abs"].to_numpy(dtype=np.float64)
    if TIME_SEC_COL in gt_df.columns and TIME_NANOSEC_COL in gt_df.columns:
        return gt_df[TIME_SEC_COL].to_numpy(dtype=np.float64) + gt_df[TIME_NANOSEC_COL].to_numpy(dtype=np.float64) * 1e-9
    raise ValueError(f"ground truth needs t_abs or {TIME_SEC_COL!r}/{TIME_NANOSEC_COL!r}")


def _sort_est_timeseries_for_interp(est_t: np.ndarray, est_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(est_t, kind="mergesort")
    t_s, p_s = est_t[order], est_pos[order]
    if len(t_s) < 2:
        return t_s, p_s
    keep = np.append(np.abs(np.diff(t_s)) > 1e-15, True)  # collapse duplicate timestamps, keep the last
    return t_s[keep], p_s[keep]


def time_alignment_report(hist: pd.DataFrame, gt_df: pd.DataFrame) -> dict[str, Any]:
    """Read-only diagnostic: which time columns were used, their ranges, and overlap size."""
    out: dict[str, Any] = {"error": None}
    try:
        est_t, gt_t = _est_time_seconds(hist), _gt_time_seconds(gt_df)
    except ValueError as e:
        out["error"] = str(e)
        return out
    est_min, est_max = float(np.nanmin(est_t)), float(np.nanmax(est_t))
    gt_min, gt_max = float(np.nanmin(gt_t)), float(np.nanmax(gt_t))
    out.update(
        est_t_min=est_min, est_t_max=est_max, gt_t_min=gt_min, gt_t_max=gt_max,
        overlap_duration_s=max(0.0, min(est_max, gt_max) - max(est_min, gt_min)),
        n_gt_in_overlap=int(np.sum((gt_t >= est_min) & (gt_t <= est_max))),
    )
    return out


def _nan_row(sequence_name: str, skipped: str) -> dict[str, Any]:
    row = {c: np.nan for c in EVALUATION_CSV_COLUMNS}
    row["sequence_name"], row["skipped"] = sequence_name, skipped
    return row


class TrajectoryEvaluator:
    @staticmethod
    def write_metrics_csv(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = [{c: r.get(c, np.nan) for c in EVALUATION_CSV_COLUMNS} for r in rows]
        pd.DataFrame(normalized, columns=list(EVALUATION_CSV_COLUMNS)).to_csv(path, index=False)

    def evaluate(self, hist: pd.DataFrame, gt_df: pd.DataFrame, *, sequence_name: str = "", print_report: bool = True) -> dict[str, Any]:
        if gt_df is None or gt_df.empty:
            return self._skip(sequence_name, "no_ground_truth", print_report)
        if hist is None or hist.empty:
            return self._skip(sequence_name, "empty_history", print_report)
        if "local_x" not in gt_df.columns or "local_y" not in gt_df.columns:
            return self._skip(sequence_name, "missing_local_xy", print_report)

        try:
            est_t, gt_t = _est_time_seconds(hist), _gt_time_seconds(gt_df)
        except ValueError as e:
            return self._skip(sequence_name, str(e), print_report)

        use_3d = "local_z" in gt_df.columns and "p_z" in hist.columns and pd.api.types.is_numeric_dtype(gt_df["local_z"])
        est_pos = hist[["p_x", "p_y", "p_z"]].to_numpy(dtype=np.float64) if use_3d else hist[["p_x", "p_y"]].to_numpy(dtype=np.float64)
        gt_pos = gt_df[["local_x", "local_y", "local_z"]].to_numpy(dtype=np.float64) if use_3d else gt_df[["local_x", "local_y"]].to_numpy(dtype=np.float64)

        valid = (gt_t >= est_t.min()) & (gt_t <= est_t.max())
        gt_t, gt_pos = gt_t[valid], gt_pos[valid]
        if len(gt_t) < 2:
            return self._skip(sequence_name, "insufficient_overlap", print_report)

        est_t_i, est_pos_i = _sort_est_timeseries_for_interp(est_t, est_pos)
        if len(est_t_i) < 2:
            return self._skip(sequence_name, "insufficient_est_samples", print_report)

        matched = match_predictions_to_gt(gt_t, gt_pos, est_t_i, est_pos_i)
        ate_x, ate_y, ate_z = calculate_per_axis_rmse(gt_pos, matched)
        ate = calculate_ate_norm_rmse_3d(gt_pos, matched) if (use_3d and np.isfinite(ate_z)) else calculate_ate_rmse_synced(gt_pos[:, :2], matched[:, :2])

        gt_xy, matched_xy = gt_pos[:, :2], matched[:, :2]
        gt_h, est_h = extract_spatial_headings(gt_xy, 0.1), extract_spatial_headings(matched_xy, 0.1)
        ahe_deg = calculate_absolute_heading_error(gt_h, est_h)
        rpe_trans, rpe_rot = calculate_rpe_metrics_synced(gt_xy, matched_xy, gt_h, est_h, window_m=1.0)
        fpe = float(np.linalg.norm(gt_xy[-1] - matched_xy[-1]))

        gt_resampled, gt_len = resample_spatially(gt_xy, step=0.5)
        est_resampled, est_len = resample_spatially(matched_xy, step=0.5)
        len_err, frechet = calculate_shape_metrics(gt_resampled, est_resampled, gt_len, est_len)
        drift_pct = (fpe / gt_len) * 100 if gt_len > 0 else 0.0

        out = {
            "sequence_name": sequence_name, "skipped": "",
            "ate_m": ate, "ate_x_m": ate_x, "ate_y_m": ate_y, "ate_z_m": ate_z if use_3d else float("nan"),
            "ahe_deg": ahe_deg, "rpe_trans_pct": rpe_trans, "rpe_rot_deg_per_m": rpe_rot,
            "fpe_m": fpe, "drift_pct": drift_pct, "length_err": len_err, "frechet_m": frechet,
            "gt_length_m": gt_len, "est_length_m": est_len,
        }
        if print_report:
            print(
                "--- EVALUATION METRICS ---\n"
                f"ATE [m]: {ate:.4f}  AHE [deg]: {ahe_deg:.4f}\n"
                f"RPE trans [%]: {rpe_trans:.4f}  RPE rot [deg/m]: {rpe_rot:.4f}\n"
                f"FPE [m]: {fpe:.4f}  Frechet [m]: {frechet:.4f}\n"
                "--------------------------"
            )
        return out

    @staticmethod
    def _skip(sequence_name: str, reason: str, print_report: bool) -> dict[str, Any]:
        if print_report:
            print(f"[eval] skipped ({reason})")
        return _nan_row(sequence_name, reason)
