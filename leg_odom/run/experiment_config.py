"""Loads and validates experiment YAML: robot, dataset, contact detector, EKF noise, output layout."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

EXPERIMENT_SCHEMA_VERSION = 1

ALLOWED_KINEMATICS = frozenset({"anymal", "go2"})
ALLOWED_DATASET_KINDS = frozenset({"tartanground", "ocelot"})
ALLOWED_CONTACT_DETECTORS = frozenset({"none", "gmm", "autoencoder"})

_GMM_DEFAULTS = {
    "mode": "offline",
    "pretrained_path": None,
    "history_length": 1,
    "trans_stay": 0.99,
    "fit_interval": 250,
    "window_size": 500,
    "degeneracy_max_weight": 0.98,
    "random_state": 42,
    "feature_fields": ["est_tau_calf", "v_foot_body_x", "v_foot_body_y", "v_foot_body_z", "p_foot_body_z"],
}

_AUTOENCODER_DEFAULTS = {
    "mode": "offline",
    "model_dir": None,
    "stance_probability_threshold": 0.5,
    "device": None,
    "encode_batch_size": 256,
    "gmm": {"pretrained": True, "window_size": 500, "fit_interval": 250},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else copy.deepcopy(v)
    return out


def _default_experiment_dict(detector: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "run": {"name": "unnamed_run", "debug": False},
        "robot": {"kinematics": "anymal"},
        "dataset": {"kind": "tartanground", "sequence_dir": ""},
        "contact": {"detector": "none"},
        "ekf": {"noise_config": None, "initialize_nominal_from_data": False},
        "output": {"base_dir": "output_leg_odom", "include_timestamp": True},
    }
    if detector == "gmm":
        base["contact"]["gmm"] = dict(_GMM_DEFAULTS)
    elif detector == "autoencoder":
        base["contact"]["autoencoder"] = dict(_AUTOENCODER_DEFAULTS)
    return base


def merge_experiment_defaults(loaded: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(loaded, Mapping):
        raise TypeError("experiment YAML root must be a mapping")
    detector = str(loaded.get("contact", {}).get("detector", "none")).lower()
    return _deep_merge(_default_experiment_dict(detector), dict(loaded))


def load_experiment_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"experiment config not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or "run" not in raw or "name" not in raw.get("run", {}):
        raise ValueError("experiment YAML must set run.name explicitly")
    if "sequence_dir" not in raw.get("dataset", {}):
        raise ValueError("experiment YAML must set dataset.sequence_dir explicitly")
    return merge_experiment_defaults(raw)


def validate_experiment_dict(cfg: Mapping[str, Any], *, strict_paths: bool = False, workspace_root: Path | None = None) -> None:
    ver = int(cfg.get("schema_version", -1))
    if ver != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {ver!r}; expected {EXPERIMENT_SCHEMA_VERSION}")

    kin = str(cfg["robot"]["kinematics"]).lower()
    if kin not in ALLOWED_KINEMATICS:
        raise ValueError(f"robot.kinematics must be one of {sorted(ALLOWED_KINEMATICS)}, got {kin!r}")

    dkind = str(cfg["dataset"]["kind"]).lower()
    if dkind not in ALLOWED_DATASET_KINDS:
        raise ValueError(f"dataset.kind must be one of {sorted(ALLOWED_DATASET_KINDS)}, got {dkind!r}")

    seq_raw = cfg["dataset"].get("sequence_dir")
    if not isinstance(seq_raw, str) or not seq_raw.strip():
        raise ValueError("dataset.sequence_dir must be a non-empty string")
    seq = Path(seq_raw).expanduser()
    if not seq.is_absolute():
        if workspace_root is None:
            raise ValueError(f"dataset.sequence_dir is relative and workspace_root was not given: {seq_raw!r}")
        cfg["dataset"]["sequence_dir"] = str((workspace_root / seq).resolve())

    det = str(cfg["contact"]["detector"]).lower()
    if det not in ALLOWED_CONTACT_DETECTORS:
        raise ValueError(f"contact.detector must be one of {sorted(ALLOWED_CONTACT_DETECTORS)}, got {det!r}")
    if det == "gmm" and not isinstance(cfg["contact"].get("gmm"), Mapping):
        raise ValueError("contact.detector is gmm but contact.gmm is missing")
    if det == "autoencoder":
        ae = cfg["contact"].get("autoencoder")
        if not isinstance(ae, Mapping) or not str(ae.get("model_dir", "")).strip():
            raise ValueError("contact.detector is autoencoder but contact.autoencoder.model_dir is missing")

    if not isinstance(cfg["run"]["name"], str) or not cfg["run"]["name"].strip():
        raise ValueError("run.name must be a non-empty string")
    if not isinstance(cfg["output"]["base_dir"], str) or not cfg["output"]["base_dir"].strip():
        raise ValueError("output.base_dir must be a non-empty string")

    if strict_paths:
        if workspace_root is None:
            raise ValueError("workspace_root is required when strict_paths=True")
        _validate_dataset_paths(cfg)
        _validate_noise_config_file(cfg, workspace_root)


def _validate_dataset_paths(cfg: Mapping[str, Any]) -> None:
    seq = Path(cfg["dataset"]["sequence_dir"]).expanduser().resolve()
    kind = str(cfg["dataset"]["kind"]).lower()
    if not seq.is_dir():
        raise ValueError(f"dataset.sequence_dir is not a directory: {seq}")
    required = "imu.csv" if kind == "tartanground" else "lowstate.csv"
    if not (seq / required).is_file():
        raise ValueError(f"dataset: missing {required} under {seq}")


def _validate_noise_config_file(cfg: Mapping[str, Any], workspace_root: Path) -> None:
    nc = cfg.get("ekf", {}).get("noise_config")
    if not nc:
        return
    p = Path(str(nc)).expanduser()
    p = p.resolve() if p.is_absolute() else (workspace_root / p).resolve()
    if not p.is_file():
        raise ValueError(f"ekf.noise_config must point to an existing file, got {p}")


def resolve_dataset_paths(cfg: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    seq = Path(out["dataset"]["sequence_dir"]).expanduser()
    out["dataset"]["sequence_dir"] = str(seq.resolve() if seq.is_absolute() else (workspace_root / seq).resolve())
    return out


def resolve_ekf_noise_config_path(cfg: dict[str, Any], workspace_root: Path) -> None:
    nc = cfg.get("ekf", {}).get("noise_config")
    if not nc:
        return
    p = Path(str(nc)).expanduser()
    cfg["ekf"]["noise_config"] = str(p.resolve() if p.is_absolute() else (workspace_root / p).resolve())


def resolve_contact_autoencoder_paths(cfg: dict[str, Any], workspace_root: Path) -> None:
    if str(cfg.get("contact", {}).get("detector", "")).lower() != "autoencoder":
        return
    ae = cfg["contact"]["autoencoder"]
    v = ae.get("model_dir")
    if not isinstance(v, str) or not v.strip():
        return
    p = Path(v.strip()).expanduser()
    ae["model_dir"] = str(p.resolve() if p.is_absolute() else (workspace_root / p).resolve())


def debug_enabled(cfg: Mapping[str, Any]) -> bool:
    return bool(cfg.get("run", {}).get("debug", False))
