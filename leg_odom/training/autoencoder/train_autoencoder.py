"""Trains the denoising autoencoder and fits its latent-space GMM.

Run leg_odom.features.precompute_contact_instants first, then:

    python -m leg_odom.training.autoencoder.train_autoencoder \\
      --config leg_odom/training/autoencoder/default_autoencoder_config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from leg_odom.features.precomputed_io import discover_precomputed_instants_npz, load_precomputed_sequence_npz
from leg_odom.run.kinematics_factory import build_kinematics_by_name
from leg_odom.training.autoencoder.config import load_autoencoder_train_config
from leg_odom.training.autoencoder.models import AutoencoderModel


def _pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _split_paths(paths: list[Path], train_ratio: float, val_ratio: float, seed: int) -> tuple[list[Path], list[Path], list[Path]]:
    rng = np.random.default_rng(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, round(n * train_ratio))
    n_val = round(n * val_ratio)
    return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]


class WindowDataset(Dataset):
    """Sliding windows over one leg's instant timeline, scaled by a fitted StandardScaler."""

    def __init__(self, instants: np.ndarray, window_size: int, scaler: StandardScaler) -> None:
        self.window_size = window_size
        pad = np.repeat(instants[0:1], window_size - 1, axis=0) if window_size > 1 else instants[:0]
        padded = np.concatenate([pad, instants], axis=0)
        self.windows = scaler.transform(padded.astype(np.float64)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.windows) - self.window_size + 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.windows[idx : idx + self.window_size])


def _apply_augmentation(x: torch.Tensor, aug_cfg: Mapping) -> torch.Tensor:
    if not aug_cfg.get("enabled", True):
        return x
    noise_std = float(aug_cfg.get("gaussian_noise_std", 0.0))
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std
    scale_min, scale_max = float(aug_cfg.get("scale_min", 1.0)), float(aug_cfg.get("scale_max", 1.0))
    if scale_min != 1.0 or scale_max != 1.0:
        scale = torch.empty((x.size(0), 1, 1), device=x.device).uniform_(scale_min, scale_max)
        x = x * scale
    return x


def _run_epoch(model: AutoencoderModel, loader: DataLoader, device: torch.device, optimizer, aug_cfg: Mapping | None) -> float:
    train = optimizer is not None
    model.train(train)
    losses = []
    for batch in loader:
        x = batch.to(device)
        x_in = _apply_augmentation(x, aug_cfg) if aug_cfg is not None else x
        with torch.set_grad_enabled(train):
            recon, _ = model(x_in)
            loss = F.mse_loss(recon, x)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def _encode_all(model: AutoencoderModel, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            out.append(model.encode(batch.to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)


def run_training(cfg: dict) -> None:
    device = _pick_device(cfg.get("training", {}).get("device"))
    kin = build_kinematics_by_name(cfg["robot"]["kinematics"])
    n_legs = kin.n_legs
    seed = int(cfg["training"]["seed"])
    torch.manual_seed(seed)

    paths = discover_precomputed_instants_npz(cfg["dataset"]["precomputed_root"])
    train_paths, val_paths, _test_paths = _split_paths(
        paths, float(cfg["training"]["train_ratio"]), float(cfg["training"]["val_ratio"]), seed
    )
    bundles = {p: load_precomputed_sequence_npz(p, n_legs=n_legs) for p in paths}
    feature_fields = bundles[paths[0]].feature_fields

    train_instants = np.concatenate([bundles[p].instants_by_leg[leg] for p in train_paths for leg in range(n_legs)], axis=0)
    scaler = StandardScaler().fit(train_instants)

    window_size = int(cfg["model"]["window_size"])
    train_ds = torch.utils.data.ConcatDataset(
        [WindowDataset(bundles[p].instants_by_leg[leg], window_size, scaler) for p in train_paths for leg in range(n_legs)]
    )
    val_ds = torch.utils.data.ConcatDataset(
        [WindowDataset(bundles[p].instants_by_leg[leg], window_size, scaler) for p in val_paths for leg in range(n_legs)]
    ) if val_paths else None

    batch_size = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds is not None else None

    model_cfg = cfg["model"]
    model = AutoencoderModel(
        input_dim=len(feature_fields), window_size=window_size, latent_dim=int(model_cfg["latent_dim"]),
        encoder_type=model_cfg["encoder_type"], cnn_cfg=model_cfg.get("cnn"), gru_cfg=model_cfg.get("gru"),
        decoder_cfg=model_cfg["decoder"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]))

    aug_cfg = cfg.get("augmentation", {"enabled": False})
    train_losses, val_losses = [], []
    best_val, best_state = float("inf"), None

    for epoch in range(int(cfg["training"]["epochs"])):
        train_loss = _run_epoch(model, train_loader, device, optimizer, aug_cfg)
        val_loss = _run_epoch(model, val_loader, device, None, None) if val_loader is not None else train_loss
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"[train_autoencoder] epoch {epoch + 1}: train_mse={train_loss:.6f} val_mse={val_loss:.6f}")
        if val_loss < best_val:
            best_val, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save({"state_dict": model.state_dict(), "encoder_type": model_cfg["encoder_type"]}, out_dir / "contact_autoencoder.pt")
    np.savez(out_dir / "contact_autoencoder_scaler.npz", mean=scaler.mean_, scale=scaler.scale_)

    latents = _encode_all(model, DataLoader(train_ds, batch_size=batch_size), device)
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=seed).fit(latents)
    np.savez(
        out_dir / "contact_autoencoder_gmm.npz", means=gmm.means_, covariances=gmm.covariances_,
        weights=gmm.weights_, n_components=2,
    )

    meta = {
        "feature_fields": list(feature_fields),
        "history_length": window_size,
        "instant_dim": len(feature_fields),
        "dataset_kind": cfg["dataset"]["kind"],
        "robot_kinematics": type(kin).__name__,
        "seed": seed,
        "autoencoder_model": model_cfg,
    }
    (out_dir / "contact_autoencoder_meta.json").write_text(json.dumps(meta, indent=2))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("reconstruction MSE")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curve.png", dpi=150)
    plt.close(fig)

    print(f"[train_autoencoder] wrote checkpoint, scaler, GMM, and meta under {out_dir.resolve()}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train the DAE+GMM contact detector's autoencoder.")
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()
    run_training(load_autoencoder_train_config(args.config))


if __name__ == "__main__":
    main()
