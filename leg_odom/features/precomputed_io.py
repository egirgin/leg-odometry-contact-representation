"""Reads and writes precomputed_instants.npz -- one file per sequence, one array per leg."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

PRECOMPUTED_INSTANTS_FILENAME = "precomputed_instants.npz"


def precomputed_npz_relpath(dataset_root: Path, sequence_dir: Path) -> Path:
    """Mirrors sequence_dir under the precomputed tree; falls back to a hash if it's outside dataset_root."""
    dr, sd = Path(dataset_root).expanduser().resolve(), Path(sequence_dir).expanduser().resolve()
    try:
        return sd.relative_to(dr)
    except ValueError:
        return Path("_external") / hashlib.sha256(str(sd).encode()).hexdigest()[:16]


def discover_precomputed_instants_npz(precomputed_root: str | Path) -> list[Path]:
    root = Path(precomputed_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"precomputed_root is not a directory: {root}")
    paths = sorted({p.resolve() for p in root.rglob(PRECOMPUTED_INSTANTS_FILENAME)}, key=str)
    if not paths:
        raise FileNotFoundError(f"no {PRECOMPUTED_INSTANTS_FILENAME} under {root}; run precompute_contact_instants first")
    return paths


@dataclass(frozen=True, slots=True)
class PrecomputedSequenceBundle:
    npz_path: Path
    instants_by_leg: dict[int, npt.NDArray[np.float64]]
    feature_fields: tuple[str, ...]
    sequence_dir: str
    robot_kinematics: str


def save_sequence_npz(path: Path, *, instants_by_leg: dict[int, np.ndarray], feature_fields: tuple[str, ...], sequence_dir: str, robot_kinematics: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_kw = {f"instants_leg{leg}": arr.astype(np.float64, copy=False) for leg, arr in instants_by_leg.items()}
    save_kw["feature_fields_str"] = np.array(",".join(feature_fields))
    save_kw["sequence_dir"] = np.array(sequence_dir)
    save_kw["robot_kinematics"] = np.array(robot_kinematics)
    np.savez_compressed(path, **save_kw)


def load_precomputed_sequence_npz(npz_path: Path, *, n_legs: int) -> PrecomputedSequenceBundle:
    p = Path(npz_path).expanduser().resolve()
    with np.load(p, allow_pickle=False) as z:
        instants_by_leg = {leg: np.array(z[f"instants_leg{leg}"], dtype=np.float64, copy=True) for leg in range(n_legs)}
        feature_fields = tuple(str(z["feature_fields_str"]).split(","))
        sequence_dir = str(z["sequence_dir"])
        robot_kinematics = str(z["robot_kinematics"])
    return PrecomputedSequenceBundle(p, instants_by_leg, feature_fields, sequence_dir, robot_kinematics)
