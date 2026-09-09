"""Two-state HMM with multivariate Gaussian emissions (state 0 = swing, state 1 = stance)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import multivariate_normal


class TwoStateGaussianHMM:
    """Fixed 2x2 transition matrix; emissions come from an external GMM fit."""

    def __init__(self, trans_stay: float) -> None:
        ts = float(trans_stay)
        if not (0.0 < ts < 1.0):
            raise ValueError("trans_stay must be in (0, 1)")
        switch = 1.0 - ts
        self._trans_mat = np.array([[ts, switch], [switch, ts]])
        self.belief = np.array([0.5, 0.5])
        self._dist_swing: multivariate_normal | None = None
        self._dist_stance: multivariate_normal | None = None

    def update_dists(
        self,
        mu_swing: npt.NDArray[np.floating],
        cov_swing: npt.NDArray[np.floating],
        mu_stance: npt.NDArray[np.floating],
        cov_stance: npt.NDArray[np.floating],
        *,
        ridge: float = 1e-6,
    ) -> None:
        d = int(np.asarray(mu_swing).reshape(-1).shape[0])
        reg = ridge * np.eye(d)
        self._dist_swing = multivariate_normal(
            mean=np.asarray(mu_swing, dtype=np.float64).reshape(d),
            cov=np.asarray(cov_swing, dtype=np.float64).reshape(d, d) + reg,
            allow_singular=True,
        )
        self._dist_stance = multivariate_normal(
            mean=np.asarray(mu_stance, dtype=np.float64).reshape(d),
            cov=np.asarray(cov_stance, dtype=np.float64).reshape(d, d) + reg,
            allow_singular=True,
        )

    def reset_belief(self) -> None:
        self.belief = np.array([0.5, 0.5])

    def update(self, x: npt.NDArray[np.floating]) -> tuple[float, bool]:
        if self._dist_swing is None or self._dist_stance is None:
            raise RuntimeError("call update_dists before update")
        xv = np.asarray(x, dtype=np.float64).reshape(-1)
        predicted = self.belief @ self._trans_mat
        likelihoods = np.array([self._dist_swing.pdf(xv), self._dist_stance.pdf(xv)])
        unnorm = predicted * likelihoods
        total = float(np.sum(unnorm))
        self.belief = unnorm / total if total >= 1e-12 else predicted
        p_stance = float(self.belief[1])
        return p_stance, p_stance > float(self.belief[0])
