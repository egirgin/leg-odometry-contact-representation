"""Builds a dataset backend from cfg["dataset"]["kind"]."""

from __future__ import annotations

from typing import Any, Mapping

from leg_odom.datasets.base import BaseLegOdometryDataset
from leg_odom.datasets.ocelot import OcelotDataset
from leg_odom.datasets.tartanground import TartangroundDataset


def build_leg_odometry_dataset(
    cfg: Mapping[str, Any], *, verbose: bool = False, sanitize_imu: bool = True, validate: bool = True, preload: bool = True
) -> BaseLegOdometryDataset:
    kind = str(cfg["dataset"]["kind"]).lower()
    sequence_dir = cfg["dataset"]["sequence_dir"]
    kwargs = dict(verbose=verbose, sanitize_imu=sanitize_imu, validate=validate, preload=preload, extra_meta={"dataset_kind": kind})

    if kind == "tartanground":
        return TartangroundDataset(sequence_dir, **kwargs)
    if kind == "ocelot":
        return OcelotDataset(sequence_dir, **kwargs)
    raise ValueError(f"unsupported dataset.kind {kind!r}")
