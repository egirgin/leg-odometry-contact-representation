"""Builds per-foot contact detectors from cfg["contact"]["detector"]."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from leg_odom.contact.base import BaseContactDetector
from leg_odom.contact.hmm_gmm.detector import build_gmm_hmm_detectors_from_cfg
from leg_odom.datasets.types import LegOdometrySequence
from leg_odom.kinematics.base import BaseKinematics


@dataclass(frozen=True)
class ContactStack:
    detector_id: str
    per_foot: list[BaseContactDetector] | None


def build_contact_stack(
    cfg: Mapping[str, Any],
    *,
    recording: LegOdometrySequence | None = None,
    kin_model: BaseKinematics | None = None,
    workspace_root: Path | None = None,
) -> ContactStack:
    det = str(cfg.get("contact", {}).get("detector", "none")).lower()

    if det == "gmm":
        return ContactStack(det, build_gmm_hmm_detectors_from_cfg(cfg, recording=recording, kin_model=kin_model))

    if det == "autoencoder":
        from leg_odom.contact.dae_gmm import build_dae_gmm_detectors_from_cfg

        if recording is None or kin_model is None:
            raise ValueError("contact.detector autoencoder requires recording and kin_model")
        return ContactStack(det, build_dae_gmm_detectors_from_cfg(cfg, recording=recording, kin_model=kin_model, workspace_root=workspace_root))

    return ContactStack(det, None)
