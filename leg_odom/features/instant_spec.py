"""The scalar instant feature vector shared by both contact detectors.

Both the HMM+GMM baseline and the DAE+GMM detector consume the same 5D per-timestep,
per-leg vector: vertical foot position, 3D foot velocity, and calf torque (see the paper's
feature-selection ablation). This module builds that vector from a ContactDetectorStepInput
(online, inside the EKF loop) or from a whole recorded sequence (offline, for training).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from leg_odom.contact.base import ContactDetectorStepInput
from leg_odom.io.columns import IMU_ACCEL_COLS, IMU_GYRO_COLS, motor_position_cols, motor_torque_cols, motor_velocity_cols
from leg_odom.kinematics.base import BaseKinematics

INSTANT_FEATURE_SPEC_VERSION = 1

ALLOWED_INSTANT_FEATURE_FIELDS: frozenset[str] = frozenset(
    {
        "p_foot_body_x", "p_foot_body_y", "p_foot_body_z",
        "v_foot_body_x", "v_foot_body_y", "v_foot_body_z",
        "est_tau_hip", "est_tau_thigh", "est_tau_calf",
        "gyro_x", "gyro_y", "gyro_z",
        "accel_x", "accel_y", "accel_z",
    }
)

DEFAULT_INSTANT_FEATURE_FIELDS: tuple[str, ...] = (
    "est_tau_calf",
    "v_foot_body_x",
    "v_foot_body_y",
    "v_foot_body_z",
    "p_foot_body_z",
)


@dataclass(frozen=True, slots=True)
class InstantFeatureSpec:
    """An ordered, validated list of instant feature names."""

    fields: tuple[str, ...]
    stance_height_index: int  # index of p_foot_body_z within `fields`; lower mean => stance

    @property
    def instant_dim(self) -> int:
        return len(self.fields)


def parse_instant_feature_fields(names: Sequence[str]) -> InstantFeatureSpec:
    """Validates and wraps a feature-name list. p_foot_body_z must be included:

    it's what both detectors use to decide which GMM cluster is "stance" (lower mean
    foot height wins), since neither is trained against any external label.
    """
    if not names:
        raise ValueError("instant feature fields list is empty")
    fields = tuple(str(x).strip() for x in names)
    for f in fields:
        if f not in ALLOWED_INSTANT_FEATURE_FIELDS:
            raise ValueError(f"unknown instant feature field {f!r}")
    if "p_foot_body_z" not in fields:
        raise ValueError("feature_fields must include p_foot_body_z (used for stance/swing ordering)")
    return InstantFeatureSpec(fields=fields, stance_height_index=fields.index("p_foot_body_z"))


def instant_vector_from_step(step: ContactDetectorStepInput, spec: InstantFeatureSpec) -> npt.NDArray[np.float64]:
    """Maps one ContactDetectorStepInput to a (instant_dim,) vector, in feature order."""
    tau = np.asarray(step.tau_leg, dtype=np.float64).reshape(-1)
    vb = np.asarray(step.v_foot_body, dtype=np.float64).reshape(3)
    pb = np.asarray(step.p_foot_body, dtype=np.float64).reshape(3)
    gy = np.asarray(step.gyro_body_corrected, dtype=np.float64).reshape(3)
    ac = np.asarray(step.accel_body_corrected, dtype=np.float64).reshape(3)

    values = {
        "p_foot_body_x": pb[0], "p_foot_body_y": pb[1], "p_foot_body_z": pb[2],
        "v_foot_body_x": vb[0], "v_foot_body_y": vb[1], "v_foot_body_z": vb[2],
        "est_tau_hip": tau[0] if tau.size > 0 else 0.0,
        "est_tau_thigh": tau[1] if tau.size > 1 else 0.0,
        "est_tau_calf": tau[2] if tau.size > 2 else (tau[-1] if tau.size else 0.0),
        "gyro_x": gy[0], "gyro_y": gy[1], "gyro_z": gy[2],
        "accel_x": ac[0], "accel_y": ac[1], "accel_z": ac[2],
    }
    return np.array([float(values[name]) for name in spec.fields], dtype=np.float64)


def flatten_history_window(rows: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """(N, d) oldest-first -> (N * d,) row-major."""
    r = np.asarray(rows, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(f"expected (N, d) window, got shape {r.shape}")
    return r.reshape(-1, order="C")


def sliding_windows_flat(instants: npt.NDArray[np.float64], history_length: int) -> npt.NDArray[np.float64]:
    """(T, d) -> (T - N + 1, N * d); the first N-1 rows are dropped (no padding)."""
    x = np.asarray(instants, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (T, d), got {x.shape}")
    t, d = x.shape
    n = int(history_length)
    if n < 1:
        raise ValueError("history_length must be >= 1")
    if t < n:
        return np.zeros((0, n * d), dtype=np.float64)
    out = np.empty((t - n + 1, n * d), dtype=np.float64)
    for k in range(n - 1, t):
        out[k - (n - 1), :] = flatten_history_window(x[k - n + 1 : k + 1, :])
    return out


def build_timeline_features_for_leg(
    frames: pd.DataFrame, kin_model: BaseKinematics, leg_index: int, spec: InstantFeatureSpec
) -> npt.NDArray[np.float64]:
    """Offline (T, instant_dim) for one leg, using the same FK/velocity as the EKF's contact step.

    Uses raw gyro/accel from the merged frame (no EKF bias subtraction) to match what pretraining
    should see: kinematics as logged, not filter-internal state.
    """
    motor_cols = list(motor_position_cols())
    vel_cols = list(motor_velocity_cols())
    tau_cols = list(motor_torque_cols())
    gyro_cols = list(IMU_GYRO_COLS)
    accel_cols = list(IMU_ACCEL_COLS)

    n_legs, jpl = kin_model.n_legs, kin_model.joints_per_leg
    if leg_index < 0 or leg_index >= n_legs:
        raise ValueError(f"leg_index must be in [0, {n_legs - 1}]")

    t_rows = len(frames)
    out = np.zeros((t_rows, spec.instant_dim), dtype=np.float64)
    sl = slice(leg_index * jpl, (leg_index + 1) * jpl)

    for k in range(t_rows):
        row = frames.iloc[k]
        gyro = row[gyro_cols].to_numpy(dtype=np.float64)
        accel = row[accel_cols].to_numpy(dtype=np.float64)
        q_leg = row[motor_cols].to_numpy(dtype=np.float64)[sl]
        dq_leg = row.reindex(vel_cols, fill_value=0.0).to_numpy(dtype=np.float64)[sl]
        tau_leg = row.reindex(tau_cols, fill_value=0.0).to_numpy(dtype=np.float64)[sl]

        p_fb = np.asarray(kin_model.fk(leg_index, q_leg), dtype=np.float64).reshape(3)
        jac = np.asarray(kin_model.J_analytical(leg_index, q_leg), dtype=np.float64).reshape(3, jpl)
        v_foot_body = np.cross(gyro, p_fb) + jac @ dq_leg

        step = ContactDetectorStepInput(
            p_foot_body=p_fb,
            v_foot_body=v_foot_body,
            q_leg=np.ascontiguousarray(q_leg),
            dq_leg=np.ascontiguousarray(dq_leg),
            tau_leg=np.ascontiguousarray(tau_leg),
            gyro_body_corrected=np.ascontiguousarray(gyro),
            accel_body_corrected=np.ascontiguousarray(accel),
        )
        out[k, :] = instant_vector_from_step(step, spec)
    return out
