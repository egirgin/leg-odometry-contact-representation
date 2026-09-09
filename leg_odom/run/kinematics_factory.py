"""Builds a kinematics backend from cfg["robot"]["kinematics"]."""

from __future__ import annotations

from typing import Any, Mapping

from leg_odom.kinematics.anymal import AnymalKinematics
from leg_odom.kinematics.base import BaseKinematics
from leg_odom.kinematics.go2 import Go2Kinematics


def build_kinematics_by_name(name: str) -> BaseKinematics:
    key = name.strip().lower()
    if key == "anymal":
        return AnymalKinematics()
    if key == "go2":
        return Go2Kinematics()
    raise ValueError(f"unsupported robot kinematics {name!r} (expected anymal or go2)")


def build_kinematics_backend(cfg: Mapping[str, Any]) -> BaseKinematics:
    return build_kinematics_by_name(str(cfg["robot"]["kinematics"]))
