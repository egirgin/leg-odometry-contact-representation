"""2-component GMM fitting for the HMM's emissions, with pretrained .npz I/O."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from leg_odom.datasets.types import LegOdometrySequence
from leg_odom.features import DEFAULT_INSTANT_FEATURE_FIELDS, InstantFeatureSpec, build_timeline_features_for_leg, parse_instant_feature_fields, sliding_windows_flat
from leg_odom.kinematics.base import BaseKinematics


def flat_ordering_component_index(spec: InstantFeatureSpec, history_length: int) -> int:
    """Column in the flattened N*d emission that holds p_foot_body_z, for stance/swing ordering."""
    return (int(history_length) - 1) * spec.instant_dim + spec.stance_height_index


def order_gmm_components(
    means: npt.NDArray[np.float64], covariances: npt.NDArray[np.float64], spec: InstantFeatureSpec, history_length: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Permutes sklearn's two components so row 0 is stance (lower mean foot height), row 1 swing."""
    j = flat_ordering_component_index(spec, history_length)
    stance_idx = int(np.argmin(means[:, j]))
    swing_idx = 1 - stance_idx
    return np.stack([means[stance_idx], means[swing_idx]]), np.stack([covariances[stance_idx], covariances[swing_idx]])


def fit_gmm_ordered(
    X: npt.NDArray[np.float64], spec: InstantFeatureSpec, history_length: int, *, random_state: int = 42
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], bool]:
    """Fits a full-covariance 2-GMM. Returns (means (2,D), covariances (2,D,D), degenerate)."""
    X = np.asarray(X, dtype=np.float64)
    d = int(X.shape[1]) if X.ndim == 2 else 0
    if X.ndim != 2 or X.shape[0] < 4:
        return np.zeros((2, d)), np.zeros((2, d, d)), True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=random_state, max_iter=200)
        try:
            gmm.fit(X)
        except (ValueError, np.linalg.LinAlgError):
            return np.zeros((2, d)), np.zeros((2, d, d)), True

    if float(np.max(gmm.weights_)) >= 0.999:
        return gmm.means_, gmm.covariances_, True
    mo, co = order_gmm_components(gmm.means_, gmm.covariances_, spec, history_length)
    return mo, co, False


def load_pretrained_gmm_npz(
    path: Path, *, expected_feature_dim: int, expected_history_length: int | None = None, expected_instant_dim: int | None = None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"pretrained GMM file not found: {path}")
    data = np.load(path, allow_pickle=True)
    means = np.asarray(data["means"], dtype=np.float64)
    covs = np.asarray(data["covariances"], dtype=np.float64)
    if means.shape != (2, expected_feature_dim):
        raise ValueError(f"pretrained GMM means shape {means.shape}, expected (2, {expected_feature_dim})")
    if covs.shape != (2, expected_feature_dim, expected_feature_dim):
        raise ValueError(f"pretrained GMM covariances shape {covs.shape}")
    if expected_history_length is not None and "history_length" in data.files:
        if int(data["history_length"]) != expected_history_length:
            raise ValueError("pretrained history_length does not match detector configuration")
    if expected_instant_dim is not None and "instant_dim" in data.files:
        if int(data["instant_dim"]) != expected_instant_dim:
            raise ValueError("pretrained instant_dim does not match detector configuration")
    return means, covs


def fit_offline_per_leg(
    recording: LegOdometrySequence, kin_model: BaseKinematics, *, feature_fields: tuple[str, ...] | None, history_length: int, random_state: int = 42
) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Whole-sequence GMM per leg, used by mode: offline before the EKF loop starts."""
    spec = parse_instant_feature_fields(feature_fields or DEFAULT_INSTANT_FEATURE_FIELDS)
    n = int(history_length)
    out = []
    for leg in range(kin_model.n_legs):
        inst = build_timeline_features_for_leg(recording.frames, kin_model, leg, spec)
        X = sliding_windows_flat(inst, n)
        if X.shape[0] < 4:
            raise ValueError(f"leg {leg}: need at least {n + 3} frames for offline GMM, got {len(inst)}")
        mo, co, bad = fit_gmm_ordered(X, spec, n, random_state=random_state + leg)
        if bad:
            raise RuntimeError(f"leg {leg}: offline GMM fit degenerated; try more data or online mode with a fallback .npz")
        out.append((mo, co))
    return out
