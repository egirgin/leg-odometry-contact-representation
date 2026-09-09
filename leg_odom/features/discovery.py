"""Finds sequence directories under a dataset root, for precompute and training."""

from __future__ import annotations

from pathlib import Path


def is_valid_tartanground_sequence_dir(sequence_dir: Path) -> tuple[bool, str]:
    d = sequence_dir.expanduser().resolve()
    if not (d / "imu.csv").is_file():
        return False, "missing imu.csv"
    bags = sorted(d.glob("*_bag.csv"))
    if len(bags) != 1:
        return False, f"expected exactly one *_bag.csv, found {len(bags)}"
    return True, ""


def is_valid_ocelot_sequence_dir(sequence_dir: Path) -> tuple[bool, str]:
    return (sequence_dir.expanduser().resolve() / "lowstate.csv").is_file(), "missing lowstate.csv"


def _discover_under_marker(root: str | Path, *, marker: str, validate, empty_message: str) -> list[Path]:
    root_p = Path(root).expanduser().resolve()
    if not root_p.is_dir():
        raise NotADirectoryError(f"dataset root is not a directory: {root_p}")

    seen: set[Path] = set()
    valid: list[Path] = []
    for path in root_p.rglob(marker):
        parent = path.parent.resolve()
        if parent in seen:
            continue
        seen.add(parent)
        ok, _ = validate(parent)
        if ok:
            valid.append(parent)

    if not valid:
        raise FileNotFoundError(empty_message.format(root=root_p))
    return sorted(valid, key=str)


def discover_tartanground_sequence_dirs(root: str | Path) -> list[Path]:
    return _discover_under_marker(
        root, marker="imu.csv", validate=is_valid_tartanground_sequence_dir,
        empty_message="no valid TartanGround sequences under {root} (need imu.csv + one *_bag.csv each)",
    )


def discover_ocelot_sequence_dirs(root: str | Path) -> list[Path]:
    return _discover_under_marker(
        root, marker="lowstate.csv", validate=is_valid_ocelot_sequence_dir,
        empty_message="no valid OCELOT sequences under {root} (need lowstate.csv each)",
    )


def discover_sequence_dirs(dataset_kind: str, root: str | Path) -> list[Path]:
    if dataset_kind == "tartanground":
        return discover_tartanground_sequence_dirs(root)
    if dataset_kind == "ocelot":
        return discover_ocelot_sequence_dirs(root)
    raise ValueError(f"unsupported dataset_kind {dataset_kind!r}")


def infer_dataset_kind_from_sequence_dir(sequence_dir: str | Path) -> str:
    seq = Path(sequence_dir).expanduser().resolve()
    if (seq / "lowstate.csv").is_file():
        return "ocelot"
    if is_valid_tartanground_sequence_dir(seq)[0]:
        return "tartanground"
    raise FileNotFoundError(f"could not infer dataset kind from {seq}")
