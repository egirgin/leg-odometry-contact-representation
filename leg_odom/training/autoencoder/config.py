"""Loads YAML config for leg_odom.training.autoencoder.train_autoencoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_autoencoder_train_config(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"autoencoder train config not found: {p}")
    cfg = dict(yaml.safe_load(p.read_text(encoding="utf-8")))
    cfg.setdefault("output", {}).setdefault("dir", "pretrained_autoencoder")
    return cfg
