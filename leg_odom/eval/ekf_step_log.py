"""Per-timestep EKF + contact state, streamed to CSV. Feeds leg_odom.eval.trajectory_eval."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial.transform import Rotation

from leg_odom.filters.esekf import ErrorStateEkf
from leg_odom.io.columns import TIME_NANOSEC_COL, TIME_SEC_COL

_NLEGS = 4
_P_DIAG_NAMES = ("P_dp_x", "P_dp_y", "P_dp_z")  # position-error variances, world frame


def empty_zupt_info() -> dict[str, Any]:
    return {"per_foot": [], "accepted": 0, "nis": float("nan"), "dof": 0, "nis_lo": float("nan"), "nis_hi": float("nan")}


def build_ekf_step_log_row(
    timeline_row: pd.Series,
    ekf: ErrorStateEkf,
    *,
    gyro_raw: npt.NDArray[np.floating],
    foot_kin: list[tuple[npt.NDArray, npt.NDArray, npt.NDArray]],
    stance: list[bool],
    contact_score: list[float],
    contact_zupt_var: list[float],
    zupt_info: Mapping[str, Any],
    n_legs: int = _NLEGS,
) -> dict[str, Any]:
    """foot_kin holds (p_foot_body, J, qdot) per leg; the rest are per-leg detector outputs."""
    eul = Rotation.from_matrix(ekf.R).as_euler("zyx", degrees=True)
    row: dict[str, Any] = {
        TIME_SEC_COL: float(timeline_row.get(TIME_SEC_COL, float("nan"))),
        TIME_NANOSEC_COL: float(timeline_row.get(TIME_NANOSEC_COL, float("nan"))),
        "t_abs": float(timeline_row["t_abs"]),
        "p_x": float(ekf.p[0]), "p_y": float(ekf.p[1]), "p_z": float(ekf.p[2]),
        "v_x": float(ekf.v[0]), "v_y": float(ekf.v[1]), "v_z": float(ekf.v[2]),
        "roll_deg": float(eul[2]), "pitch_deg": float(eul[1]), "yaw_deg": float(eul[0]),
        "bgx": float(ekf.bias_gyro[0]), "bgy": float(ekf.bias_gyro[1]), "bgz": float(ekf.bias_gyro[2]),
        "bax": float(ekf.bias_accel[0]), "bay": float(ekf.bias_accel[1]), "baz": float(ekf.bias_accel[2]),
        "zupt_n_feet_accepted": int(zupt_info.get("accepted", 0)),
        "zupt_nis": float(zupt_info.get("nis", float("nan"))),
        "zupt_nis_dof": int(zupt_info.get("dof", 0)),
        "zupt_nis_lo": float(zupt_info.get("nis_lo", float("nan"))),
        "zupt_nis_hi": float(zupt_info.get("nis_hi", float("nan"))),
    }
    p_diag = np.diag(ekf.P)
    for i, name in enumerate(_P_DIAG_NAMES):
        row[name] = float(p_diag[i])

    zupt_by_leg = {int(d["leg_id"]): d for d in zupt_info.get("per_foot", []) if "leg_id" in d}
    g = np.asarray(gyro_raw, dtype=np.float64).reshape(3)

    for i in range(n_legs):
        row[f"leg{i}_stance"] = int(bool(stance[i]))
        row[f"leg{i}_contact_score"] = float(contact_score[i])
        row[f"leg{i}_zupt_meas_var"] = float(contact_zupt_var[i]) if np.isfinite(contact_zupt_var[i]) else float("nan")
        zd = zupt_by_leg.get(i)
        if zd is not None:
            row[f"leg{i}_zupt_mahal"] = float(zd.get("mahal", float("nan")))
            row[f"leg{i}_zupt_accepted"] = float(int(bool(zd.get("accepted"))))
            row[f"leg{i}_zupt_innov_vx"] = float(zd.get("v_pred_x", float("nan")))
            row[f"leg{i}_zupt_innov_vy"] = float(zd.get("v_pred_y", float("nan")))
            row[f"leg{i}_zupt_innov_vz"] = float(zd.get("v_pred_z", float("nan")))
        else:
            for suffix in ("mahal", "accepted", "innov_vx", "innov_vy", "innov_vz"):
                row[f"leg{i}_zupt_{suffix}"] = float("nan")

        pb, jj, qd = foot_kin[i]
        vw = ekf.foot_velocity_world(g, pb, jj, qd)
        row[f"leg{i}_v_wx"], row[f"leg{i}_v_wy"], row[f"leg{i}_v_wz"] = float(vw[0]), float(vw[1]), float(vw[2])

    return row


def ekf_step_log_columns(n_legs: int = _NLEGS) -> tuple[str, ...]:
    base = (
        TIME_SEC_COL, TIME_NANOSEC_COL, "t_abs",
        "p_x", "p_y", "p_z", "v_x", "v_y", "v_z", "roll_deg", "pitch_deg", "yaw_deg",
        "bgx", "bgy", "bgz", "bax", "bay", "baz",
        *_P_DIAG_NAMES,
        "zupt_n_feet_accepted", "zupt_nis", "zupt_nis_dof", "zupt_nis_lo", "zupt_nis_hi",
    )
    per_leg = []
    for i in range(n_legs):
        per_leg += [
            f"leg{i}_stance", f"leg{i}_contact_score", f"leg{i}_zupt_meas_var",
            f"leg{i}_zupt_mahal", f"leg{i}_zupt_accepted",
            f"leg{i}_zupt_innov_vx", f"leg{i}_zupt_innov_vy", f"leg{i}_zupt_innov_vz",
            f"leg{i}_v_wx", f"leg{i}_v_wy", f"leg{i}_v_wz",
        ]
    return base + tuple(per_leg)


EKF_STEP_LOG_COLUMNS = ekf_step_log_columns()


def _csv_cell(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return ""
    return str(x)


class EkfStepLogWriter:
    """Streams rows to CSV without holding the whole run in memory."""

    def __init__(self, path: Path, *, n_legs: int = _NLEGS) -> None:
        self.path = Path(path)
        self._fieldnames = ekf_step_log_columns(n_legs)
        self._f = self.path.open("w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=self._fieldnames, extrasaction="ignore")
        self._w.writeheader()

    def write_row(self, row: Mapping[str, Any]) -> None:
        self._w.writerow({k: _csv_cell(row.get(k, "")) for k in self._fieldnames})

    def close(self) -> None:
        self._f.close()


def sanitize_sequence_slug(sequence_name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in sequence_name.strip())[:200] or "recording"
