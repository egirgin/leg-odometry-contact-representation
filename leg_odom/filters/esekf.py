"""Error-state EKF: IMU propagation plus a per-foot zero-velocity (ZUPT) correction.

The nominal state (position, velocity, rotation, IMU biases) lives on SO(3) x R^12 and is
updated by injecting a locally linear 15-dim error state -- standard trick to avoid
linearizing the rotation directly. See Sola, "Quaternion kinematics for the error-state
Kalman filter" for the derivation this follows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
import scipy.stats as st
import yaml
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.transform import Rotation

_DEFAULT_P0_DIAG = np.array(
    [0.01**2] * 3 + [0.5**2] * 3 + [np.deg2rad(5.0) ** 2] * 3 + [0.5**2] * 3 + [np.deg2rad(1.0) ** 2] * 3
)

_DEFAULT_IMU_NOISE: dict[str, float] = {
    "accel_std": 0.5,
    "gyro_std": float(np.deg2rad(1.0)),
    "accel_bias_std": 0.01,
    "gyro_bias_std": float(np.deg2rad(0.01)),
}

GRAVITY_WORLD_FLU = np.array([0.0, 0.0, 9.81])


class ErrorStateEkf:
    """Nominal state (p, v, R, bias_accel, bias_gyro) plus a 15x15 error covariance."""

    def __init__(self, *, P0: npt.NDArray[np.floating] | None = None, imu_noise: dict[str, float] | None = None) -> None:
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.R = np.eye(3)
        self.bias_accel = np.zeros(3)
        self.bias_gyro = np.zeros(3)
        self.P = np.diag(_DEFAULT_P0_DIAG).copy() if P0 is None else np.asarray(P0, dtype=np.float64).reshape(15, 15).copy()

        self.imu_noise = dict(_DEFAULT_IMU_NOISE)
        if imu_noise:
            self.imu_noise.update(imu_noise)
        ast, gst = self.imu_noise["accel_std"], self.imu_noise["gyro_std"]
        ab, gb = self.imu_noise["accel_bias_std"], self.imu_noise["gyro_bias_std"]
        self.Q = np.diag([ast**2] * 3 + [gst**2] * 3 + [ab**2] * 3 + [gb**2] * 3)

        self._p0, self._v0, self._R0 = self.p.copy(), self.v.copy(), self.R.copy()
        self._ba0, self._bg0, self._P0 = self.bias_accel.copy(), self.bias_gyro.copy(), self.P.copy()

    def reset(self) -> None:
        self.p, self.v, self.R = self._p0.copy(), self._v0.copy(), self._R0.copy()
        self.bias_accel, self.bias_gyro, self.P = self._ba0.copy(), self._bg0.copy(), self._P0.copy()

    def seed_nominal_state(
        self,
        *,
        p: npt.NDArray[np.floating],
        v: npt.NDArray[np.floating],
        R: npt.NDArray[np.floating],
        bias_accel: npt.NDArray[np.floating] | None = None,
        bias_gyro: npt.NDArray[np.floating] | None = None,
    ) -> None:
        """Overrides the initial nominal state, e.g. from ground truth at t=0."""
        self.p = np.asarray(p, dtype=np.float64).reshape(3).copy()
        self.v = np.asarray(v, dtype=np.float64).reshape(3).copy()
        self.R = np.asarray(R, dtype=np.float64).reshape(3, 3).copy()
        if bias_accel is not None:
            self.bias_accel = np.asarray(bias_accel, dtype=np.float64).reshape(3).copy()
        if bias_gyro is not None:
            self.bias_gyro = np.asarray(bias_gyro, dtype=np.float64).reshape(3).copy()
        self._p0, self._v0, self._R0 = self.p.copy(), self.v.copy(), self.R.copy()
        self._ba0, self._bg0 = self.bias_accel.copy(), self.bias_gyro.copy()

    @staticmethod
    def skew(v: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        v = np.asarray(v, dtype=np.float64).reshape(3)
        return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])

    def predict(
        self, accel_raw: npt.NDArray[np.floating], gyro_raw: npt.NDArray[np.floating], dt: float, *, accel_gravity_compensated: bool = False
    ) -> None:
        dt = float(dt)
        gyro_corrected = np.asarray(gyro_raw, dtype=np.float64).reshape(3) - self.bias_gyro
        accel_body_corrected = np.asarray(accel_raw, dtype=np.float64).reshape(3) - self.bias_accel

        accel_world = self.R @ accel_body_corrected
        if not accel_gravity_compensated:
            accel_world = accel_world - GRAVITY_WORLD_FLU

        self.p += self.v * dt + 0.5 * accel_world * dt**2
        self.v += accel_world * dt
        self.R = self.R @ Rotation.from_rotvec(gyro_corrected * dt).as_matrix()

        F = np.zeros((15, 15))
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 6:9] = -self.R @ self.skew(accel_body_corrected)
        F[3:6, 9:12] = -self.R
        F[6:9, 6:9] = -self.skew(gyro_corrected)
        F[6:9, 12:15] = -np.eye(3)
        stm = np.eye(15) + F * dt

        G = np.zeros((15, 12))
        G[3:6, 0:3] = -self.R
        G[6:9, 3:6] = -np.eye(3)
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)

        self.P = stm @ self.P @ stm.T + G @ self.Q @ G.T * dt

    def imu_predict(
        self, dt_s: float, gyro_rad_s: npt.NDArray[np.floating], accel_m_s2: npt.NDArray[np.floating], *, accel_gravity_compensated: bool = False
    ) -> None:
        self.predict(accel_m_s2, gyro_rad_s, dt_s, accel_gravity_compensated=accel_gravity_compensated)

    def foot_velocity_world(
        self, gyro_raw: npt.NDArray[np.floating], p_foot_body: npt.NDArray[np.floating], J: npt.NDArray[np.floating], qdot: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.float64]:
        w = np.asarray(gyro_raw, dtype=np.float64).reshape(3)
        pb = np.asarray(p_foot_body, dtype=np.float64).reshape(3)
        jac = np.asarray(J, dtype=np.float64).reshape(3, -1)
        qd = np.asarray(qdot, dtype=np.float64).reshape(jac.shape[1])
        v_rel_body = np.cross(w - self.bias_gyro, pb) + jac @ qd
        return self.v + self.R @ v_rel_body

    def update_zupt(self, stance_legs: list[dict[str, Any]], gyro_raw: npt.NDArray[np.floating]) -> dict[str, Any]:
        """Stacks one 3D velocity pseudo-measurement per stance leg, gates each by NIS, updates."""
        empty = {"per_foot": [], "accepted": 0, "nis": np.nan, "dof": 0, "nis_lo": np.nan, "nis_hi": np.nan}
        if not stance_legs:
            return empty

        n = len(stance_legs)
        H = np.zeros((3 * n, 15))
        R_block = np.zeros((3 * n, 3 * n))
        innov = np.zeros(3 * n)
        per_foot_info: list[dict[str, Any]] = []
        gyro_raw = np.asarray(gyro_raw, dtype=np.float64).reshape(3)
        nis_threshold = st.chi2.ppf(0.95, df=3)

        for i, leg in enumerate(stance_legs):
            p_foot_body = np.asarray(leg["p_foot_body"], dtype=np.float64).reshape(3)
            jacobian = np.asarray(leg["J"], dtype=np.float64)
            qdot = np.asarray(leg["qdot"], dtype=np.float64)
            r_foot = np.asarray(leg["R_foot"], dtype=np.float64).reshape(3, 3)

            v_foot_pred = self.foot_velocity_world(gyro_raw, p_foot_body, jacobian, qdot)
            v_foot_rel_body = self.R.T @ (v_foot_pred - self.v)

            idx = slice(3 * i, 3 * (i + 1))
            innov[idx] = -v_foot_pred
            Hi = np.zeros((3, 15))
            Hi[:, 3:6] = np.eye(3)
            Hi[:, 6:9] = -self.R @ self.skew(v_foot_rel_body)
            Hi[:, 12:15] = self.R @ self.skew(p_foot_body)
            H[idx, :] = Hi
            R_block[idx, idx] = r_foot
            per_foot_info.append(
                {
                    "leg_id": leg["leg_id"],
                    "qscore": float(leg.get("qscore", 1.0)),
                    "speed_world": float(np.linalg.norm(v_foot_pred)),
                    "v_pred_x": float(v_foot_pred[0]),
                    "v_pred_y": float(v_foot_pred[1]),
                    "v_pred_z": float(v_foot_pred[2]),
                }
            )

        accepted: list[int] = []
        for i in range(n):
            idx = slice(3 * i, 3 * (i + 1))
            S_i = H[idx, :] @ self.P @ H[idx, :].T + R_block[idx, idx]
            try:
                cS, lower = cho_factor(S_i, lower=True, check_finite=False)
                nis_i = float(innov[idx].T @ cho_solve((cS, lower), innov[idx], check_finite=False))
            except np.linalg.LinAlgError:
                s_reg = 0.5 * (S_i + S_i.T) + 1e-9 * np.eye(3)
                nis_i = float(innov[idx].T @ np.linalg.pinv(s_reg) @ innov[idx])
            per_foot_info[i]["mahal"] = nis_i
            per_foot_info[i]["accepted"] = bool(nis_i < nis_threshold)
            if nis_i < nis_threshold:
                accepted.append(i)

        if not accepted:
            return {**empty, "per_foot": per_foot_info}

        idx_arr = np.concatenate([np.arange(3 * i, 3 * i + 3) for i in accepted])
        H_acc, innov_acc, R_acc = H[idx_arr, :], innov[idx_arr], R_block[np.ix_(idx_arr, idx_arr)]
        S = H_acc @ self.P @ H_acc.T + R_acc
        try:
            cS, lower = cho_factor(S, lower=True, check_finite=False)
            K = cho_solve((cS, lower), H_acc @ self.P).T
            nis = float(innov_acc.T @ cho_solve((cS, lower), innov_acc, check_finite=False))
        except (np.linalg.LinAlgError, ValueError):
            s_inv = np.linalg.pinv(0.5 * (S + S.T) + 1e-9 * np.eye(S.shape[0]))
            K = self.P @ H_acc.T @ s_inv
            nis = float(innov_acc.T @ s_inv @ innov_acc)

        dx = K @ innov_acc
        self.p += dx[0:3]
        self.v += dx[3:6]
        self.R = self.R @ Rotation.from_rotvec(dx[6:9]).as_matrix()
        self.bias_accel += dx[9:12]
        self.bias_gyro += dx[12:15]

        # Joseph form: numerically stable even when K isn't exactly the Kalman-optimal gain.
        I = np.eye(15)
        self.P = (I - K @ H_acc) @ self.P @ (I - K @ H_acc).T + K @ R_acc @ K.T

        dof = len(idx_arr)
        return {
            "per_foot": per_foot_info,
            "accepted": len(accepted),
            "nis": nis,
            "dof": dof,
            "nis_lo": float(st.chi2.ppf(0.05, dof)),
            "nis_hi": float(st.chi2.ppf(0.95, dof)),
        }


def _resolve_noise_config_path(raw: str, workspace_root: Path | None) -> Path:
    p = Path(raw).expanduser()
    return p.resolve() if p.is_absolute() else (Path(workspace_root) / p).resolve() if workspace_root else p.resolve()


def _apply_ekf_noise_mapping(block: Mapping[str, Any], noise: dict[str, float], p0: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    imu_block = block.get("imu_noise")
    if isinstance(imu_block, Mapping):
        for k in noise:
            if k in imu_block:
                noise[k] = float(imu_block[k])
    p0d = block.get("P0_diagonal")
    if isinstance(p0d, (list, tuple)) and len(p0d) == 15:
        p0 = np.diag(np.asarray(p0d, dtype=np.float64))
    return noise, p0


def build_error_state_ekf(resolved_cfg: Mapping[str, Any] | None = None, *, workspace_root: Path | None = None) -> ErrorStateEkf:
    """Resolution order: code defaults -> ekf.noise_config YAML -> inline ekf.* overrides."""
    block: Mapping[str, Any] = (resolved_cfg or {}).get("ekf") or {}
    noise = dict(_DEFAULT_IMU_NOISE)
    p0 = np.diag(_DEFAULT_P0_DIAG)

    noise_config = block.get("noise_config")
    if noise_config:
        path = _resolve_noise_config_path(str(noise_config), workspace_root)
        if not path.is_file():
            raise FileNotFoundError(f"ekf.noise_config: not a file: {path}")
        file_data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(file_data, Mapping):
            noise, p0 = _apply_ekf_noise_mapping(file_data, noise, p0)

    noise, p0 = _apply_ekf_noise_mapping(block, noise, p0)
    return ErrorStateEkf(P0=p0, imu_noise=noise)
