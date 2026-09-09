"""Contact detector interface shared by HMM+GMM and DAE+GMM.

ContactDetectorStepInput carries per-foot kinematics for one timestep; a detector reads
only the fields it needs. The Jacobian itself stays out of this struct — it's an EKF/FK
concern, not a detector one. ZUPT measurement covariance is derived from p_stance in
leg_odom.filters.zupt_measurement, not here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class ContactDetectorStepInput:
    p_foot_body: npt.NDArray[np.float64]  # foot origin, body frame, shape (3,)
    v_foot_body: npt.NDArray[np.float64]  # foot velocity relative to the body, body frame, shape (3,)
    q_leg: npt.NDArray[np.float64]  # joint positions (rad), shape (n_joints,)
    dq_leg: npt.NDArray[np.float64]  # joint rates (rad/s)
    tau_leg: npt.NDArray[np.float64]  # estimated joint torques (Nm)
    gyro_body_corrected: npt.NDArray[np.float64]  # bias-corrected gyro, shape (3,)
    accel_body_corrected: npt.NDArray[np.float64]  # bias-corrected specific force, shape (3,)


class ContactEstimate(NamedTuple):
    stance: bool
    p_stance: float


class BaseContactDetector(ABC):
    """Shared interface for stance estimators (HMM+GMM, DAE+GMM)."""

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Length of the flattened feature vector this detector consumes."""

    @property
    @abstractmethod
    def history_length(self) -> int:
        """Window size N (the current row is index -1 when N > 1)."""

    @abstractmethod
    def update(self, step: ContactDetectorStepInput) -> ContactEstimate:
        """Consume the latest step and return a stance flag and stance probability."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state at sequence boundaries."""
