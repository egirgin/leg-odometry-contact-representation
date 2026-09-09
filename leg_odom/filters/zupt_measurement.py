"""Maps a contact detector's stance probability to a ZUPT measurement covariance."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

_P_STANCE_FLOOR = 1e-9


def zupt_isotropic_meas_from_p_stance(p_stance: float) -> tuple[float, npt.NDArray[np.float64]]:
    """sigma_sq = 0.5 / max(p_stance, floor); low stance belief inflates R and lets NIS gating reject it."""
    sigma_sq = 0.5 / max(float(p_stance), _P_STANCE_FLOOR)
    return sigma_sq, np.eye(3) * sigma_sq
