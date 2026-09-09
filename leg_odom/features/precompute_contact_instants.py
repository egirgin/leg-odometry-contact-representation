"""Precomputes per-sequence instant features for GMM/DAE training.

    python -m leg_odom.features.precompute_contact_instants --config leg_odom/features/default_precompute_config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from leg_odom.features.discovery import discover_sequence_dirs
from leg_odom.features.instant_spec import DEFAULT_INSTANT_FEATURE_FIELDS, build_timeline_features_for_leg, parse_instant_feature_fields
from leg_odom.features.precompute_config import load_precompute_config
from leg_odom.features.precomputed_io import PRECOMPUTED_INSTANTS_FILENAME, precomputed_npz_relpath, save_sequence_npz
from leg_odom.io.split_imu_bag import load_prepared_split_sequence
from leg_odom.io.ocelot_recording import load_prepared_ocelot
from leg_odom.run.kinematics_factory import build_kinematics_by_name

MANIFEST_NAME = "precompute_manifest.json"


def _load_frames(dataset_kind: str, sequence_dir: Path):
    if dataset_kind == "tartanground":
        df, _, _, _ = load_prepared_split_sequence(sequence_dir)
        return df
    df, _, _, _, _ = load_prepared_ocelot(sequence_dir)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute instant kinematic features per sequence.")
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()

    cfg = load_precompute_config(args.config)
    dataset_root = Path(cfg["dataset_root"]).expanduser().resolve()
    output_root = Path(cfg["output_root"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_kind = str(cfg["dataset_kind"]).lower()
    robot = str(cfg["robot"]).lower()
    feature_fields = tuple(cfg.get("feature_fields", DEFAULT_INSTANT_FEATURE_FIELDS))
    spec = parse_instant_feature_fields(feature_fields)
    kin = build_kinematics_by_name(robot)

    sequences = discover_sequence_dirs(dataset_kind, dataset_root)
    if cfg.get("max_sequences"):
        sequences = sequences[: int(cfg["max_sequences"])]

    manifest: dict[str, str] = {}
    for seq_dir in tqdm(sequences, desc="precompute", unit="seq", disable=not cfg["verbose"]):
        out_dir = output_root / precomputed_npz_relpath(dataset_root, seq_dir)
        npz_path = out_dir / PRECOMPUTED_INSTANTS_FILENAME
        if npz_path.is_file() and not cfg["overwrite"]:
            manifest[str(seq_dir)] = str(npz_path)
            continue

        frames = _load_frames(dataset_kind, seq_dir)
        instants_by_leg = {leg: build_timeline_features_for_leg(frames, kin, leg, spec) for leg in range(kin.n_legs)}
        save_sequence_npz(
            npz_path, instants_by_leg=instants_by_leg, feature_fields=feature_fields,
            sequence_dir=str(seq_dir.resolve()), robot_kinematics=type(kin).__name__,
        )
        manifest[str(seq_dir)] = str(npz_path)

    (output_root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    print(f"[precompute] wrote {len(manifest)} bundles under {output_root}")


if __name__ == "__main__":
    main()
