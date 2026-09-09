"""Fits the HMM+GMM baseline's 2-component GMM on pooled instant features.

Run leg_odom.features.precompute_contact_instants first, then:

    python -m leg_odom.training.gmm.train_gmm \\
      --precomputed-root leg_odom/features/precomputed_tartanground \\
      --output pretrained/tartanground/gmm/kinematic_weights.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from leg_odom.contact.hmm_gmm.fitting import fit_gmm_ordered
from leg_odom.features import DEFAULT_INSTANT_FEATURE_FIELDS, parse_instant_feature_fields
from leg_odom.features.precomputed_io import discover_precomputed_instants_npz, load_precomputed_sequence_npz
from leg_odom.run.kinematics_factory import build_kinematics_by_name

_HISTORY_LENGTH = 1  # pretraining always uses instant emissions; sliding windows are an online-mode detail


def save_pretrained_gmm_npz(path: Path, *, means: npt.NDArray, covariances: npt.NDArray, feature_fields: tuple[str, ...], instant_dim: int, n_samples: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, means=means, covariances=covariances, history_length=np.int64(_HISTORY_LENGTH),
        instant_dim=np.int64(instant_dim), n_samples=np.int64(n_samples), feature_fields_str=np.array(",".join(feature_fields)),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Fit the HMM+GMM baseline's GMM from precomputed instant features.")
    p.add_argument("--precomputed-root", type=str, required=True)
    p.add_argument("--robot-kinematics", type=str, default="anymal", choices=("anymal", "go2"))
    p.add_argument("--feature-fields", type=str, default=",".join(DEFAULT_INSTANT_FEATURE_FIELDS))
    p.add_argument("--output", type=str, default="pretrained/gmm/kinematic_weights.npz")
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    fields = tuple(s.strip() for s in args.feature_fields.split(",") if s.strip())
    spec = parse_instant_feature_fields(fields)
    kin = build_kinematics_by_name(args.robot_kinematics)

    paths = discover_precomputed_instants_npz(args.precomputed_root)
    blocks = []
    for npz_path in tqdm(paths, desc="load bundles", unit="seq"):
        bundle = load_precomputed_sequence_npz(npz_path, n_legs=kin.n_legs)
        if bundle.feature_fields != fields:
            raise ValueError(f"{npz_path}: precomputed feature_fields {bundle.feature_fields} != requested {fields}")
        blocks.extend(bundle.instants_by_leg[leg] for leg in range(kin.n_legs))

    X = np.vstack(blocks)
    means, covariances, degenerate = fit_gmm_ordered(X, spec, _HISTORY_LENGTH, random_state=args.random_state)
    if degenerate:
        raise RuntimeError("GMM fit degenerated; try more sequences or different feature_fields")

    out = Path(args.output)
    save_pretrained_gmm_npz(out, means=means, covariances=covariances, feature_fields=fields, instant_dim=spec.instant_dim, n_samples=X.shape[0])
    print(f"wrote {out.resolve()}  samples={X.shape[0]}")


if __name__ == "__main__":
    main()
