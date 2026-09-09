"""Foot forward kinematics and Jacobian, expressed in the body frame (FLU: +X fwd, +Y left, +Z up)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from leg_odom.thresholds import KINEMATICS_NUMERICAL_JACOBIAN_STEP


class BaseKinematics(ABC):
    """Maps joint angles/rates to foot position/velocity in the body frame."""

    n_legs: ClassVar[int] = 4
    joints_per_leg: ClassVar[int] = 3

    @staticmethod
    def _validate_leg_and_q(leg_id: int, q: npt.NDArray[np.floating], *, n_legs: int = 4) -> npt.NDArray[np.float64]:
        if leg_id < 0 or leg_id >= n_legs:
            raise ValueError(f"leg_id must be in [0, {n_legs - 1}], got {leg_id}")
        qv = np.asarray(q, dtype=np.float64).reshape(-1)
        if qv.shape != (3,):
            raise ValueError(f"joint vector must have shape (3,), got {qv.shape}")
        return qv

    @abstractmethod
    def fk(self, leg_id: int, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Foot origin position in the body frame, shape (3,)."""

    @abstractmethod
    def J_analytical(self, leg_id: int, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Jacobian J such that v_foot_body = J @ qdot, shape (3, 3)."""

    def jacobian_numerical(
        self, leg_id: int, q: npt.NDArray[np.floating], *, h: float | None = None
    ) -> npt.NDArray[np.float64]:
        """Central-difference fallback for robots without a closed-form Jacobian."""
        step = KINEMATICS_NUMERICAL_JACOBIAN_STEP if h is None else float(h)
        qv = self._validate_leg_and_q(leg_id, q, n_legs=self.n_legs)
        f0 = np.asarray(self.fk(leg_id, qv), dtype=np.float64).reshape(3)
        jac = np.zeros((3, 3), dtype=np.float64)
        for j in range(3):
            dq = np.zeros(3, dtype=np.float64)
            dq[j] = step
            f1 = np.asarray(self.fk(leg_id, qv + dq), dtype=np.float64).reshape(3)
            jac[:, j] = (f1 - f0) / step
        return jac
