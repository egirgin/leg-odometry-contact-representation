"""ANYmal-C kinematics: URDF-derived homogeneous transforms and an analytic foot Jacobian.

Leg order: 0=left front, 1=right front, 2=left hind, 3=right hind.
Joint order per leg: [q_haa, q_hfe, q_kfe] (hip abduction, hip flexion, knee flexion), radians.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from leg_odom.kinematics.base import BaseKinematics


def _ht_from_xyz_rpy(xyz, rpy) -> npt.NDArray[np.float64]:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = Rotation.from_euler("XYZ", np.asarray(rpy, dtype=np.float64)).as_matrix()
    t[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return t


def _ht_rot_x(angle: float, axis_sign: float) -> npt.NDArray[np.float64]:
    return _ht_from_xyz_rpy([0.0, 0.0, 0.0], [axis_sign * angle, 0.0, 0.0])


# URDF-derived (xyz, rpy) offsets per leg. All revolute joints are local +/-X rotations.
_URDF_PARAMS: dict[int, dict[str, Any]] = {
    0: {
        "base_haa": ([0.2999, 0.104, 0.0], [2.61799387799, 0, 0]),
        "haa_axis": 1,
        "hip_fixed": ([0, 0, 0], [-2.61799387799, 0, 0]),
        "hfe_fixed": ([0.0599, 0.08381, 0.0], [0, 0, 1.57079632679]),
        "hfe_axis": 1,
        "thigh_fixed": ([0, 0, 0], [0, 0, -1.57079632679]),
        "kfe_fixed": ([0.0, 0.1003, -0.285], [0, 0, 1.57079632679]),
        "kfe_axis": 1,
        "shank_fixed": ([0, 0, 0], [0, 0, -1.57079632679]),
        "foot_fixed": ([0.08795, 0.01305, -0.33797], [0, 0, 0]),
    },
    1: {
        "base_haa": ([0.2999, -0.104, 0.0], [-2.61799387799, 0, 0]),
        "haa_axis": 1,
        "hip_fixed": ([0, 0, 0], [2.61799387799, 0, 0]),
        "hfe_fixed": ([0.0599, -0.08381, 0.0], [0, 0, -1.57079632679]),
        "hfe_axis": -1,
        "thigh_fixed": ([0, 0, 0], [0, 0, 1.57079632679]),
        "kfe_fixed": ([0.0, -0.1003, -0.285], [0, 0, -1.57079632679]),
        "kfe_axis": -1,
        "shank_fixed": ([0, 0, 0], [0, 0, 1.57079632679]),
        "foot_fixed": ([0.08795, -0.01305, -0.33797], [0, 0, 0]),
    },
    2: {
        "base_haa": ([-0.2999, 0.104, 0.0], [-2.61799387799, 0, -3.14159265359]),
        "haa_axis": -1,
        "hip_fixed": ([0, 0, 0], [-2.61799387799, 0, -3.14159265359]),
        "hfe_fixed": ([-0.0599, 0.08381, 0.0], [0, 0, 1.57079632679]),
        "hfe_axis": 1,
        "thigh_fixed": ([0, 0, 0], [0, 0, -1.57079632679]),
        "kfe_fixed": ([-0.0, 0.1003, -0.285], [0, 0, 1.57079632679]),
        "kfe_axis": 1,
        "shank_fixed": ([0, 0, 0], [0, 0, -1.57079632679]),
        "foot_fixed": ([-0.08795, 0.01305, -0.33797], [0, 0, 0]),
    },
    3: {
        "base_haa": ([-0.2999, -0.104, 0.0], [2.61799387799, 0, -3.14159265359]),
        "haa_axis": -1,
        "hip_fixed": ([0, 0, 0], [2.61799387799, 0, -3.14159265359]),
        "hfe_fixed": ([-0.0599, -0.08381, 0.0], [0, 0, -1.57079632679]),
        "hfe_axis": -1,
        "thigh_fixed": ([0, 0, 0], [0, 0, 1.57079632679]),
        "kfe_fixed": ([-0.0, -0.1003, -0.285], [0, 0, -1.57079632679]),
        "kfe_axis": -1,
        "shank_fixed": ([0, 0, 0], [0, 0, -1.57079632679]),
        "foot_fixed": ([-0.08795, -0.01305, -0.33797], [0, 0, 0]),
    },
}


class AnymalKinematics(BaseKinematics):
    """Precomputes the fixed chain segments per leg; live joints add X rotations."""

    def __init__(self) -> None:
        self._t_base_haa: dict[int, npt.NDArray[np.float64]] = {}
        self._t_hip_hfe: dict[int, npt.NDArray[np.float64]] = {}
        self._t_thigh_kfe: dict[int, npt.NDArray[np.float64]] = {}
        self._t_shank_foot: dict[int, npt.NDArray[np.float64]] = {}
        self._axes: dict[int, tuple[int, int, int]] = {}

        for i, p in _URDF_PARAMS.items():
            self._t_base_haa[i] = _ht_from_xyz_rpy(*p["base_haa"])
            self._t_hip_hfe[i] = _ht_from_xyz_rpy(*p["hip_fixed"]) @ _ht_from_xyz_rpy(*p["hfe_fixed"])
            self._t_thigh_kfe[i] = _ht_from_xyz_rpy(*p["thigh_fixed"]) @ _ht_from_xyz_rpy(*p["kfe_fixed"])
            self._t_shank_foot[i] = _ht_from_xyz_rpy(*p["shank_fixed"]) @ _ht_from_xyz_rpy(*p["foot_fixed"])
            self._axes[i] = (p["haa_axis"], p["hfe_axis"], p["kfe_axis"])

    def fk(self, leg_id: int, q: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        qv = self._validate_leg_and_q(leg_id, q)
        ax = self._axes[leg_id]
        t_haa, t_hfe, t_kfe = (_ht_rot_x(float(qv[i]), float(ax[i])) for i in range(3))

        t_foot = (
            self._t_base_haa[leg_id]
            @ t_haa
            @ self._t_hip_hfe[leg_id]
            @ t_hfe
            @ self._t_thigh_kfe[leg_id]
            @ t_kfe
            @ self._t_shank_foot[leg_id]
        )
        return np.asarray(t_foot[:3, 3], dtype=np.float64).copy()

    def J_analytical(self, leg_id: int, q: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        """Geometric Jacobian: each column is omega_j x (p_foot - p_j) for a revolute joint."""
        qv = self._validate_leg_and_q(leg_id, q)
        ax = self._axes[leg_id]
        t_haa, t_hfe, t_kfe = (_ht_rot_x(float(qv[i]), float(ax[i])) for i in range(3))

        t0 = self._t_base_haa[leg_id]
        t1 = t0 @ t_haa @ self._t_hip_hfe[leg_id]
        t2 = t1 @ t_hfe @ self._t_thigh_kfe[leg_id]
        t_foot = t2 @ t_kfe @ self._t_shank_foot[leg_id]

        p0, p1, p2, p_foot = t0[:3, 3], t1[:3, 3], t2[:3, 3], t_foot[:3, 3]
        w0 = t0[:3, :3] @ np.array([ax[0], 0.0, 0.0])
        w1 = t1[:3, :3] @ np.array([ax[1], 0.0, 0.0])
        w2 = t2[:3, :3] @ np.array([ax[2], 0.0, 0.0])

        jac = np.zeros((3, 3), dtype=np.float64)
        jac[:, 0] = np.cross(w0, p_foot - p0)
        jac[:, 1] = np.cross(w1, p_foot - p1)
        jac[:, 2] = np.cross(w2, p_foot - p2)
        return jac
