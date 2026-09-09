"""HMM+GMM contact detector: 2-component GMM emissions feeding a 2-state Gaussian HMM.

Two modes:
  offline - GMM is fit once on the whole sequence before the EKF runs. Simple and
            self-contained, but needs the full sequence up front (history_length is
            forced to 1: instant emissions only).
  online  - starts from a pretrained .npz and periodically refits the GMM on a sliding
            window, for real deployment where the whole sequence isn't available yet.
            Reverts to the last good (or pretrained) emissions if a refit looks degenerate.
"""

from __future__ import annotations

import warnings
from collections import deque
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import numpy.typing as npt
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from leg_odom.contact.base import BaseContactDetector, ContactDetectorStepInput, ContactEstimate
from leg_odom.contact.hmm_gmm.fitting import fit_gmm_ordered, fit_offline_per_leg, load_pretrained_gmm_npz
from leg_odom.contact.hmm_gmm.hmm_gaussian import TwoStateGaussianHMM
from leg_odom.datasets.types import LegOdometrySequence
from leg_odom.features import DEFAULT_INSTANT_FEATURE_FIELDS, InstantFeatureSpec, flatten_history_window, instant_vector_from_step, parse_instant_feature_fields
from leg_odom.kinematics.base import BaseKinematics


class GmmHmmContactDetector(BaseContactDetector):
    def __init__(
        self,
        *,
        feature_fields: tuple[str, ...] | None = None,
        history_length: int = 1,
        trans_stay: float = 0.99,
        mode: Literal["offline", "online"] = "offline",
        pretrained_path: str | Path | None = None,
        initial_means: npt.NDArray[np.floating] | None = None,
        initial_covariances: npt.NDArray[np.floating] | None = None,
        fit_interval: int = 250,
        window_size: int = 500,
        degeneracy_max_weight: float = 0.80,
        random_state: int = 42,
    ) -> None:
        self._spec = parse_instant_feature_fields(feature_fields or DEFAULT_INSTANT_FEATURE_FIELDS)
        self._N = 1 if mode == "offline" else max(1, int(history_length))
        self._D = self._N * self._spec.instant_dim
        self._trans_stay = float(trans_stay)
        self._mode: Literal["offline", "online"] = mode
        self._fit_interval = int(fit_interval)
        self._window_size = int(window_size)
        self._degen_w = float(degeneracy_max_weight)
        self._rng = int(random_state)

        self._instant_buf: deque[npt.NDArray[np.float64]] = deque(maxlen=self._N)
        self._flat_window: deque[npt.NDArray[np.float64]] = deque(maxlen=self._window_size)
        self._hmm = TwoStateGaussianHMM(self._trans_stay)
        self._clock = 0

        self._fallback_means: npt.NDArray[np.float64] | None = None
        self._fallback_covs: npt.NDArray[np.float64] | None = None
        self._last_good_means: npt.NDArray[np.float64] | None = None
        self._last_good_covs: npt.NDArray[np.float64] | None = None

        if mode == "offline":
            if initial_means is None or initial_covariances is None:
                raise ValueError("offline mode requires initial_means/initial_covariances from pre-fitting")
            m, c = np.asarray(initial_means, dtype=np.float64), np.asarray(initial_covariances, dtype=np.float64)
            if m.shape != (2, self._D) or c.shape != (2, self._D, self._D):
                raise ValueError(f"offline GMM shape mismatch: means {m.shape}, covs {c.shape}")
            self._fallback_means, self._fallback_covs = m.copy(), c.copy()
            self._apply_emission_params(m, c)
        else:
            if pretrained_path is None:
                raise ValueError("online mode requires pretrained_path")
            m, c = load_pretrained_gmm_npz(
                Path(pretrained_path),
                expected_feature_dim=self._D,
                expected_history_length=self._N,
                expected_instant_dim=self._spec.instant_dim,
            )
            self._fallback_means, self._fallback_covs = m.copy(), c.copy()
            self._apply_emission_params(m, c)

    @property
    def feature_dim(self) -> int:
        return self._D

    @property
    def history_length(self) -> int:
        return self._N

    def _apply_emission_params(self, means: npt.NDArray[np.float64], covs: npt.NDArray[np.float64]) -> None:
        self._hmm.update_dists(mu_swing=means[1], cov_swing=covs[1], mu_stance=means[0], cov_stance=covs[0])

    def _revert_emissions(self) -> None:
        m, c = self._last_good_means, self._last_good_covs
        if m is None:
            m, c = self._fallback_means, self._fallback_covs
        if m is not None and c is not None:
            self._apply_emission_params(m, c)

    def _maybe_refit_online(self, flat_x: npt.NDArray[np.float64]) -> None:
        if self._mode != "online":
            return
        self._flat_window.append(flat_x.copy())
        self._clock += 1
        if len(self._flat_window) < self._window_size:
            return
        if self._fit_interval <= 0 or (self._clock % self._fit_interval) != 0:
            return

        X = np.stack(list(self._flat_window))
        mo, co, bad = fit_gmm_ordered(X, self._spec, self._N, random_state=self._rng)
        if bad or mo.shape[1] != self._D:
            self._revert_emissions()
            return

        # A single dominant component means the window looks unimodal (e.g. foot stayed
        # planted the whole time) -- the fit isn't trustworthy as a stance/swing split.
        probe = GaussianMixture(n_components=2, covariance_type="full", random_state=self._rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            try:
                probe.fit(X)
            except (ValueError, np.linalg.LinAlgError):
                self._revert_emissions()
                return
        if float(np.max(probe.weights_)) >= self._degen_w:
            self._revert_emissions()
            return

        self._apply_emission_params(mo, co)
        self._last_good_means, self._last_good_covs = mo.copy(), co.copy()

    def update(self, step: ContactDetectorStepInput) -> ContactEstimate:
        inst = instant_vector_from_step(step, self._spec)
        self._instant_buf.append(inst)

        if self._mode == "online" and len(self._instant_buf) < self._N:
            return ContactEstimate(stance=True, p_stance=1.0)

        flat_x = flatten_history_window(np.stack(list(self._instant_buf)))
        self._maybe_refit_online(flat_x)
        p_stance, stance = self._hmm.update(flat_x)
        return ContactEstimate(stance=stance, p_stance=float(p_stance))

    def reset(self) -> None:
        self._instant_buf.clear()
        self._flat_window.clear()
        self._hmm.reset_belief()
        self._clock = 0
        if self._fallback_means is not None and self._fallback_covs is not None:
            self._apply_emission_params(self._fallback_means, self._fallback_covs)


def build_gmm_hmm_detectors_from_cfg(
    cfg: Mapping[str, Any], *, recording: LegOdometrySequence | None = None, kin_model: BaseKinematics | None = None
) -> list[GmmHmmContactDetector]:
    """Builds one GmmHmmContactDetector per leg from the contact.gmm config block."""
    gmm_cfg = cfg["contact"]["gmm"]
    feature_fields = tuple(str(x) for x in gmm_cfg.get("feature_fields", DEFAULT_INSTANT_FEATURE_FIELDS))
    mode = str(gmm_cfg.get("mode", "offline")).lower()
    if mode not in ("offline", "online"):
        raise ValueError(f"contact.gmm.mode must be offline|online, got {mode!r}")
    history_length = 1 if mode == "offline" else int(gmm_cfg.get("history_length", 1))

    common_kwargs: dict[str, Any] = dict(
        feature_fields=feature_fields,
        history_length=history_length,
        trans_stay=float(gmm_cfg.get("trans_stay", 0.99)),
        mode=mode,
        fit_interval=int(gmm_cfg.get("fit_interval", 250)),
        window_size=int(gmm_cfg.get("window_size", 500)),
        degeneracy_max_weight=float(gmm_cfg.get("degeneracy_max_weight", 0.80)),
        random_state=int(gmm_cfg.get("random_state", 42)),
    )

    n_legs = int(kin_model.n_legs) if kin_model is not None else 4
    if mode == "offline":
        if recording is None or kin_model is None:
            raise ValueError("offline GMM requires recording and kin_model")
        per_leg_params = fit_offline_per_leg(
            recording, kin_model, feature_fields=feature_fields, history_length=1, random_state=common_kwargs["random_state"]
        )
        return [
            GmmHmmContactDetector(**common_kwargs, initial_means=m, initial_covariances=c) for m, c in per_leg_params
        ]

    pretrained_path = gmm_cfg.get("pretrained_path")
    if not pretrained_path:
        raise ValueError("contact.gmm.pretrained_path is required for online mode")
    return [GmmHmmContactDetector(**common_kwargs, pretrained_path=pretrained_path) for _ in range(n_legs)]
