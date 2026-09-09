"""Loads and validates the YAML config for precompute_contact_instants."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def load_precompute_config(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"precompute config not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("precompute config must be a YAML mapping")
    cfg = dict(raw)

    for key in ("dataset_root", "output_root", "dataset_kind", "robot"):
        if not str(cfg.get(key, "")).strip():
            raise ValueError(f"precompute config: {key!r} is required")
    if str(cfg["dataset_kind"]).lower() not in ("tartanground", "ocelot"):
        raise ValueError("dataset_kind must be tartanground or ocelot")
    if str(cfg["robot"]).lower() not in ("anymal", "go2"):
        raise ValueError("robot must be anymal or go2")
    cfg.setdefault("overwrite", False)
    cfg.setdefault("verbose", True)
    cfg.setdefault("max_sequences", None)
    return cfg
