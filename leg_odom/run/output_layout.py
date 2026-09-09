"""Creates a timestamped run directory and writes the resolved experiment YAML into it."""

from __future__ import annotations

import copy
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

from leg_odom.run.experiment_config import (
    resolve_contact_autoencoder_paths,
    resolve_dataset_paths,
    resolve_ekf_noise_config_path,
    validate_experiment_dict,
)


def prepare_run_output_dir(
    cfg: Mapping[str, Any], *, workspace_root: Path, source_config_path: Path | None = None, validate_paths: bool = True
) -> tuple[Path, dict[str, Any]]:
    """Run directory layout: output.base_dir/output_{run.name}/{dataset.kind}/{env}/{trajectory}."""
    cfg_dict = copy.deepcopy(dict(cfg))
    validate_experiment_dict(cfg_dict, strict_paths=validate_paths, workspace_root=workspace_root)

    resolved_cfg = resolve_dataset_paths(cfg_dict, workspace_root)
    resolve_ekf_noise_config_path(resolved_cfg, workspace_root)
    resolve_contact_autoencoder_paths(resolved_cfg, workspace_root)

    run_name = str(resolved_cfg["run"]["name"])
    if any(c in run_name for c in "/\\"):
        raise ValueError(f"run.name must not contain path separators, got {run_name!r}")

    base = Path(resolved_cfg["output"]["base_dir"]).expanduser()
    if not base.is_absolute():
        base = (workspace_root / base).resolve()

    seq_path = Path(resolved_cfg["dataset"]["sequence_dir"]).resolve()
    dataset_kind = str(resolved_cfg["dataset"]["kind"]).lower()
    env_name = seq_path.parent.name
    if not env_name or env_name == seq_path.anchor:
        env_name = "_"
        warnings.warn("dataset.sequence_dir has no usable parent folder name; using env_name='_'", stacklevel=2)

    traj_leaf = seq_path.name or "sequence"
    if resolved_cfg["output"]["include_timestamp"]:
        traj_leaf = f"{traj_leaf}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_dir = base / f"output_{run_name}" / dataset_kind / env_name / traj_leaf
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "experiment_resolved.yaml").open("w", encoding="utf-8") as f:
        f.write("# Validated experiment config, saved with absolute paths for reproducibility.\n")
        if source_config_path is not None:
            f.write(f"# input_yaml: {source_config_path.expanduser().resolve()}\n")
        f.write("\n")
        yaml.safe_dump(resolved_cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return run_dir, resolved_cfg
