"""DAE+GMM contact detector: a denoising autoencoder's latent code, clustered with a 2-GMM.

The encoder (CNN or GRU, see leg_odom.training.autoencoder.models) is trained purely to
reconstruct kinematic windows -- no contact labels involved. Because the reconstruction
objective forces the latent code to separate stance from swing dynamics, a 2-component GMM
fit on that latent space recovers a usable stance/swing split. Which of the two components
means "stance" is decided the same way as the HMM+GMM baseline: the component with the lower
mean foot height wins (see leg_odom.contact.hmm_gmm).

Two modes: offline precomputes p_stance for the whole sequence up front (GMM either fit fresh
on this sequence, or loaded from a pretrained checkpoint); online refits the latent GMM on a
sliding window as new steps arrive.
"""

from __future__ import annotations

import json
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

from leg_odom.contact.base import BaseContactDetector, ContactDetectorStepInput, ContactEstimate
from leg_odom.datasets.types import LegOdometrySequence
from leg_odom.features import InstantFeatureSpec, build_timeline_features_for_leg, instant_vector_from_step, parse_instant_feature_fields
from leg_odom.kinematics.base import BaseKinematics
from leg_odom.training.autoencoder.models import AutoencoderModel

try:
    import torch
except ImportError as exc:
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError("DAE contact detection requires PyTorch") from _TORCH_IMPORT_ERROR


def _pick_device(name: str | None) -> "torch.device":
    _require_torch()
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _artifact_paths(model_dir: Path) -> tuple[Path, Path, Path, Path]:
    md = Path(model_dir).expanduser().resolve()
    ckpt = md / "contact_autoencoder.pt"
    for p in (ckpt, md / "contact_autoencoder_meta.json", md / "contact_autoencoder_scaler.npz"):
        if not p.is_file():
            raise FileNotFoundError(f"missing DAE artifact: {p}")
    return ckpt, md / "contact_autoencoder_meta.json", md / "contact_autoencoder_scaler.npz", md / "contact_autoencoder_gmm.npz"


def _scale(instants: npt.NDArray[np.float64], mean: npt.NDArray[np.float64], scale: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    return (instants - mean) / safe_scale


def _windows_for_leg(inst: npt.NDArray[np.float64], window_size: int, mean, scale) -> npt.NDArray[np.float32]:
    """Left-padded sliding windows over a (T, F) instant timeline -> (T, window_size, F)."""
    if inst.shape[0] == 0:
        return np.empty((0, window_size, inst.shape[1]), dtype=np.float32)
    pad_n = window_size - 1
    padded = np.concatenate([np.repeat(inst[0:1], pad_n, axis=0), inst]) if pad_n > 0 else inst
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window_size, axis=0).transpose(0, 2, 1)
    scaled = _scale(windows.reshape(-1, inst.shape[1]), mean, scale).astype(np.float32)
    return scaled.reshape(windows.shape[0], window_size, inst.shape[1])


def _map_clusters_to_stance(cluster_ids: npt.NDArray[np.int64], ordering_signal: npt.NDArray[np.float64]) -> dict[int, int]:
    """Cluster with the lower mean foot height (ordering_signal) is stance."""
    ids = np.asarray(cluster_ids)
    means = {int(c): float(np.mean(ordering_signal[ids == c])) for c in np.unique(ids)}
    if not means:
        return {}
    stance_id = min(means, key=means.get)
    return {c: int(c == stance_id) for c in means}


def _load_pretrained_gmm(path: Path, expected_dim: int) -> GaussianMixture:
    data = np.load(path, allow_pickle=False)
    means = np.asarray(data["means"], dtype=np.float64)
    covs = np.asarray(data["covariances"], dtype=np.float64)
    weights = np.asarray(data["weights"], dtype=np.float64)
    if means.shape[1] != expected_dim:
        raise ValueError(f"pretrained GMM latent dim {means.shape[1]} != model latent dim {expected_dim}")
    gmm = GaussianMixture(n_components=means.shape[0], covariance_type="full")
    gmm.weights_, gmm.means_, gmm.covariances_ = weights, means, covs
    gmm.precisions_cholesky_ = _compute_precision_cholesky(covs, "full")
    gmm.converged_, gmm.n_iter_ = True, 0
    return gmm


class DaeSharedRuntime:
    """A loaded DAE checkpoint + input scaler + feature spec, shared across all four legs."""

    def __init__(self, model_dir: Path, *, device: "torch.device", encode_batch_size: int) -> None:
        _require_torch()
        ckpt, meta_path, scaler_path, self.gmm_path = _artifact_paths(model_dir)

        meta = json.loads(meta_path.read_text())
        self.spec: InstantFeatureSpec = parse_instant_feature_fields(tuple(meta["feature_fields"]))
        self.window_size = int(meta["history_length"])

        z = np.load(scaler_path)
        self.mean, self.scale = z["mean"].reshape(-1), z["scale"].reshape(-1)

        model_cfg = meta["autoencoder_model"]
        self.model = AutoencoderModel(
            input_dim=self.spec.instant_dim,
            window_size=self.window_size,
            latent_dim=int(model_cfg["latent_dim"]),
            encoder_type=str(model_cfg["encoder_type"]),
            cnn_cfg=model_cfg.get("cnn", {}),
            gru_cfg=model_cfg.get("gru", {}),
            decoder_cfg=model_cfg.get("decoder", {}),
        ).to(device)
        bundle = torch.load(ckpt, map_location=device, weights_only=False)
        self.model.load_state_dict(bundle["state_dict"])
        self.model.eval()

        self.device = device
        self.encode_batch_size = int(encode_batch_size)

    def encode(self, windows: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        if windows.shape[0] == 0:
            return np.empty((0, self.model.latent_dim))
        out = []
        with torch.no_grad():
            for i in range(0, windows.shape[0], self.encode_batch_size):
                batch = torch.tensor(windows[i : i + self.encode_batch_size], dtype=torch.float32, device=self.device)
                out.append(self.model.encode(batch).cpu().numpy())
        return np.concatenate(out, axis=0)


class DaeOfflineContactDetector(BaseContactDetector):
    """Replays a precomputed p_stance array, one value per timestep."""

    def __init__(self, *, p_stance: npt.NDArray[np.float64], history_length: int, feature_dim: int, stance_probability_threshold: float = 0.5) -> None:
        self._p = np.asarray(p_stance, dtype=np.float64).reshape(-1)
        self._thr = float(stance_probability_threshold)
        self._k = 0
        self._hist = int(history_length)
        self._dim = int(feature_dim)

    @property
    def feature_dim(self) -> int:
        return self._dim

    @property
    def history_length(self) -> int:
        return self._hist

    def reset(self) -> None:
        self._k = 0

    def update(self, step: ContactDetectorStepInput) -> ContactEstimate:
        if self._p.size == 0:
            return ContactEstimate(stance=False, p_stance=0.5)
        p = float(self._p[min(self._k, self._p.size - 1)])
        self._k += 1
        return ContactEstimate(stance=p >= self._thr, p_stance=p)


def _offline_p_stance_per_leg(
    recording: LegOdometrySequence, kin_model: BaseKinematics, runtime: DaeSharedRuntime, *, gmm_path: Path | None
) -> list[npt.NDArray[np.float64]]:
    n_legs = kin_model.n_legs
    per_leg_z: list[npt.NDArray[np.float64]] = []
    per_leg_ordering: list[npt.NDArray[np.float64]] = []
    for leg in range(n_legs):
        inst = build_timeline_features_for_leg(recording.frames, kin_model, leg, runtime.spec)
        windows = _windows_for_leg(inst, runtime.window_size, runtime.mean, runtime.scale)
        per_leg_z.append(runtime.encode(windows))
        per_leg_ordering.append(inst[:, runtime.spec.stance_height_index])

    pooled_z = np.concatenate(per_leg_z, axis=0)
    pooled_ordering = np.concatenate(per_leg_ordering, axis=0)

    if gmm_path is not None:
        gmm = _load_pretrained_gmm(gmm_path, expected_dim=pooled_z.shape[1])
    else:
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
        gmm.fit(pooled_z)

    proba, labels = gmm.predict_proba(pooled_z), gmm.predict(pooled_z)
    mapping = _map_clusters_to_stance(labels, pooled_ordering)
    stance_ids = [c for c, is_stance in mapping.items() if is_stance]
    p_stance_pooled = np.sum(proba[:, stance_ids], axis=1) if stance_ids else np.full(pooled_z.shape[0], 0.5)

    out, offset = [], 0
    for leg in range(n_legs):
        n = per_leg_z[leg].shape[0]
        out.append(p_stance_pooled[offset : offset + n])
        offset += n
    return out


class DaeOnlineContactDetector(BaseContactDetector):
    """Encodes each new window and refits the latent GMM on a sliding buffer."""

    def __init__(self, *, runtime: DaeSharedRuntime, stance_probability_threshold: float, gmm_window_size: int, gmm_fit_interval: int) -> None:
        self._runtime = runtime
        self._thr = float(stance_probability_threshold)
        self._gmm_window_size = int(gmm_window_size)
        self._gmm_fit_interval = int(gmm_fit_interval)
        self._step = 0
        self._inst_buf: deque[npt.NDArray[np.float64]] = deque(maxlen=runtime.window_size)
        self._ordering_buf: deque[float] = deque(maxlen=self._gmm_window_size)
        self._z_buf: deque[npt.NDArray[np.float64]] = deque(maxlen=self._gmm_window_size)
        self._gmm: GaussianMixture | None = None
        self._gmm_mapping: dict[int, int] | None = None

    @property
    def feature_dim(self) -> int:
        return self._runtime.window_size * self._runtime.spec.instant_dim

    @property
    def history_length(self) -> int:
        return self._runtime.window_size

    def reset(self) -> None:
        self._step = 0
        self._inst_buf.clear()
        self._ordering_buf.clear()
        self._z_buf.clear()
        self._gmm, self._gmm_mapping = None, None

    def _padded_window(self) -> npt.NDArray[np.float64]:
        if not self._inst_buf:
            return np.zeros((self._runtime.window_size, self._runtime.spec.instant_dim))
        pad_n = max(0, self._runtime.window_size - len(self._inst_buf))
        tail = np.stack(self._inst_buf)
        if pad_n == 0:
            return tail
        return np.concatenate([np.repeat(tail[0:1], pad_n, axis=0), tail])

    def _maybe_refit(self) -> None:
        if self._gmm_fit_interval < 1 or self._step % self._gmm_fit_interval != 0:
            return
        if len(self._z_buf) < self._gmm_window_size:
            return
        z = np.stack(self._z_buf)
        try:
            gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
            gmm.fit(z)
        except ValueError:
            return
        self._gmm = gmm
        labels = gmm.predict(z)
        self._gmm_mapping = _map_clusters_to_stance(labels, np.array(self._ordering_buf))

    def update(self, step: ContactDetectorStepInput) -> ContactEstimate:
        inst = instant_vector_from_step(step, self._runtime.spec)
        self._inst_buf.append(inst)
        ordering_val = float(inst[self._runtime.spec.stance_height_index])
        self._ordering_buf.append(ordering_val)

        scaled = _scale(self._padded_window(), self._runtime.mean, self._runtime.scale)
        x = torch.tensor(scaled, dtype=torch.float32, device=self._runtime.device).unsqueeze(0)
        with torch.no_grad():
            z = self._runtime.model.encode(x).cpu().numpy().reshape(-1)
        self._z_buf.append(z)

        self._step += 1
        self._maybe_refit()
        if self._gmm is None:
            return ContactEstimate(stance=False, p_stance=0.5)

        proba = self._gmm.predict_proba(z.reshape(1, -1))
        mapping = self._gmm_mapping or _map_clusters_to_stance(self._gmm.predict(z.reshape(1, -1)), np.array([ordering_val]))
        stance_ids = [c for c, is_stance in mapping.items() if is_stance]
        p_stance = float(np.sum(proba[:, stance_ids])) if stance_ids else 0.5
        return ContactEstimate(stance=p_stance >= self._thr, p_stance=p_stance)


def build_dae_gmm_detectors_from_cfg(
    cfg: Mapping[str, Any], *, recording: Any, kin_model: Any, workspace_root: Path | None = None
) -> list[BaseContactDetector]:
    ae_cfg = cfg["contact"]["autoencoder"]
    model_dir = Path(ae_cfg["model_dir"]).expanduser()
    if not model_dir.is_absolute():
        model_dir = (workspace_root or Path.cwd()) / model_dir

    device = _pick_device(ae_cfg.get("device"))
    runtime = DaeSharedRuntime(model_dir, device=device, encode_batch_size=int(ae_cfg.get("encode_batch_size", 256)))
    stance_thr = float(ae_cfg.get("stance_probability_threshold", 0.5))
    mode = str(ae_cfg.get("mode", "offline")).lower()
    n_legs = int(kin_model.n_legs)

    if mode == "offline":
        gmm_cfg = ae_cfg.get("gmm", {})
        gmm_path = runtime.gmm_path if gmm_cfg.get("pretrained", True) else None
        if gmm_path is not None and not gmm_path.is_file():
            raise FileNotFoundError(f"pretrained DAE GMM not found: {gmm_path}")
        per_leg_p = _offline_p_stance_per_leg(recording, kin_model, runtime, gmm_path=gmm_path)
        feature_dim = runtime.window_size * runtime.spec.instant_dim
        return [
            DaeOfflineContactDetector(p_stance=per_leg_p[leg], history_length=runtime.window_size, feature_dim=feature_dim, stance_probability_threshold=stance_thr)
            for leg in range(n_legs)
        ]

    if mode != "online":
        raise ValueError(f"contact.autoencoder.mode must be offline or online, got {mode!r}")
    gmm_cfg = ae_cfg.get("gmm", {})
    return [
        DaeOnlineContactDetector(
            runtime=runtime,
            stance_probability_threshold=stance_thr,
            gmm_window_size=int(gmm_cfg.get("window_size", 500)),
            gmm_fit_interval=int(gmm_cfg.get("fit_interval", 250)),
        )
        for _ in range(n_legs)
    ]
