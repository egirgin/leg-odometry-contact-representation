"""Run orchestration: experiment YAML loading, output directories, the EKF pipeline."""

from __future__ import annotations

from typing import Any

from leg_odom.run.experiment_config import (
    EXPERIMENT_SCHEMA_VERSION,
    debug_enabled,
    load_experiment_yaml,
    merge_experiment_defaults,
    validate_experiment_dict,
)


def __getattr__(name: str) -> Any:
    if name == "run_ekf_pipeline":
        from leg_odom.run.ekf_process import run_ekf_pipeline

        return run_ekf_pipeline
    if name == "prepare_run_output_dir":
        from leg_odom.run.output_layout import prepare_run_output_dir

        return prepare_run_output_dir
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EXPERIMENT_SCHEMA_VERSION",
    "debug_enabled",
    "load_experiment_yaml",
    "merge_experiment_defaults",
    "prepare_run_output_dir",
    "run_ekf_pipeline",
    "validate_experiment_dict",
]
