"""Unitree Go2 kinematics: closed-form FK, numerical Jacobian.

Leg order: 0=front-left, 1=front-right, 2=rear-left, 3=rear-right.
Joint order per leg: [q_abad, q_hip, q_knee], radians.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from leg_odom.kinematics.base import BaseKinematics


class Go2Kinematics(BaseKinematics):
    """Hip offsets in the body frame plus a serial abad-hip-knee chain."""

    def __init__(self) -> None:
        self._hip_off: dict[int, npt.NDArray[np.float64]] = {
            0: np.array([0.247, 0.050, 0.0]),
            1: np.array([0.247, -0.050, 0.0]),
            2: np.array([-0.247, 0.050, 0.0]),
            3: np.array([-0.247, -0.050, 0.0]),
        }
        self._thigh_length = 0.210
        self._calf_length = 0.210
        self._abad_offset = 0.083

    def fk(self, leg_id: int, q: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        qv = self._validate_leg_and_q(leg_id, q)
        q_abad, q_hip, q_knee = float(qv[0]), float(qv[1]), float(qv[2])
        side = 1.0 if leg_id in (0, 2) else -1.0  # left legs vs. right legs mirror the abad offset

        r_abad = Rotation.from_euler("x", q_abad).as_matrix()
        r_hip = Rotation.from_euler("y", q_hip).as_matrix()
        r_knee = Rotation.from_euler("y", q_knee).as_matrix()

        p_ab = np.array([0.0, side * self._abad_offset, 0.0])
        p_th = np.array([0.0, 0.0, -self._thigh_length])
        p_cf = np.array([0.0, 0.0, -self._calf_length])

        foot = self._hip_off[leg_id] + r_abad @ (p_ab + r_hip @ (p_th + r_knee @ p_cf))
        return foot.astype(np.float64, copy=False)

    def J_analytical(self, leg_id: int, q: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        """No closed-form Jacobian for Go2; fall back to central differences."""
        return self.jacobian_numerical(leg_id, q)
