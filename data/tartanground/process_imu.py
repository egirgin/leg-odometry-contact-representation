"""Converts downloaded IMU .npy arrays from FRD to FLU convention and writes imu.csv.

Reads anymal/<ENV>/Data_anymal/<TRAJ>/imu/*.npy, writes processed/<ENV>/<TRAJ>/imu.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path("anymal")
OUTPUT_ROOT = Path("processed")

# 180-degree rotation about X: right-handed FRD -> right-handed FLU.
_T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])


def _load_npy(directory: Path, filename: str) -> np.ndarray | None:
    path = directory / filename
    return np.load(path) if path.exists() else None


def _split_time_to_sec_nanosec(time_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sec = np.floor(time_values).astype(np.int64)
    nanosec = np.round((time_values - sec) * 1e9).astype(np.int64)
    carry = nanosec // 1_000_000_000  # rounding can push a fractional part to exactly 1e9 ns
    return sec + carry, nanosec % 1_000_000_000


def process_trajectory(imu_dir: Path, output_file: Path) -> None:
    time = _load_npy(imu_dir, "imu_time.npy")
    if time is None:
        return
    time = time.flatten()

    arrays = {name: _load_npy(imu_dir, f"{name}.npy") for name in ("acc", "acc_nograv", "gyro", "pos_global", "vel_global", "ori_global")}
    if any(v is None for v in arrays.values()):
        print(f"[process_imu] incomplete arrays in {imu_dir}, skipping")
        return
    if any(len(v) != len(time) for v in arrays.values()):
        print(f"[process_imu] array length mismatch in {imu_dir}, skipping")
        return

    acc, acc_nograv, gyro = arrays["acc"], arrays["acc_nograv"], arrays["gyro"]
    pos_global = arrays["pos_global"] - arrays["pos_global"][0]

    # ori_global is ZYX (yaw, pitch, roll) in the FRD frame; rotate the basis into FLU.
    r_frd = Rotation.from_euler("ZYX", arrays["ori_global"][:, [2, 1, 0]], degrees=False)
    r_flu = np.einsum("ij,njk,kl->nil", _T, r_frd.as_matrix(), _T.T)
    quats_flu = Rotation.from_matrix(r_flu).as_quat()

    sec, nanosec = _split_time_to_sec_nanosec(time)

    df = pd.DataFrame(
        {
            "sec": sec, "nanosec": nanosec,
            "accel_x": acc[:, 0], "accel_y": -acc[:, 1], "accel_z": -acc[:, 2],
            "accel_nograv_x": acc_nograv[:, 0], "accel_nograv_y": -acc_nograv[:, 1], "accel_nograv_z": -acc_nograv[:, 2],
            "pos_x": pos_global[:, 0], "pos_y": -pos_global[:, 1], "pos_z": -pos_global[:, 2],
            "vel_x": arrays["vel_global"][:, 0], "vel_y": -arrays["vel_global"][:, 1], "vel_z": -arrays["vel_global"][:, 2],
            "gyro_x": gyro[:, 0], "gyro_y": -gyro[:, 1], "gyro_z": -gyro[:, 2],
            "ori_qx": quats_flu[:, 0], "ori_qy": quats_flu[:, 1], "ori_qz": quats_flu[:, 2], "ori_qw": quats_flu[:, 3],
        }
    )
    df.to_csv(output_file, index=False)


def main() -> None:
    if not ROOT.exists():
        raise SystemExit(f"root directory not found: {ROOT}")

    for traj_dir in ROOT.glob("*/Data_anymal/*"):
        imu_dir = traj_dir / "imu"
        if not imu_dir.exists():
            continue
        out_dir = OUTPUT_ROOT / traj_dir.parents[1].name / traj_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = out_dir / "imu.csv"
        print(f"[process_imu] {imu_dir} -> {output_file}")
        process_trajectory(imu_dir, output_file)


if __name__ == "__main__":
    main()
