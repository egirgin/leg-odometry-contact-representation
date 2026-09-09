"""Extracts /state_estimator/anymal_state from each downloaded rosbag into a kinematics CSV.

Reads anymal/<ENV>/Data_anymal/<TRAJ>/rosbags/*.bag, writes processed/<ENV>/<TRAJ>/<BAG_STEM>_bag.csv.
"""

import csv
from pathlib import Path

from rosbags.highlevel import AnyReader

ROOT = Path("anymal")
OUTPUT_ROOT = Path("processed")

HEADER = (
    ["sec", "nanosec"]
    + [f"foot_force_{i}" for i in range(4)]
    + [f"motor_{i}_{field}" for i in range(12) for field in ("q", "dq", "ddq", "tau_est")]
    + ["p_x", "p_y", "p_z", "vel_lin_x", "vel_lin_y", "vel_lin_z", "vel_ang_x", "vel_ang_y", "vel_ang_z"]
)


def process_bag(bag_path: Path, output_csv: Path) -> None:
    with AnyReader([bag_path]) as reader:
        conn = next((c for c in reader.connections if c.topic == "/state_estimator/anymal_state"), None)
        if conn is None:
            print(f"[process_bag] {bag_path.name}: topic not found, skipping")
            return

        with output_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)

            t0_sec = t0_nano = p0 = None
            for _, _, rawdata in reader.messages(connections=[conn]):
                msg = reader.deserialize(rawdata, conn.msgtype)
                sec, nano = msg.header.stamp.sec, msg.header.stamp.nanosec
                px, py, pz = msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z

                if t0_sec is None:
                    t0_sec, t0_nano, p0 = sec, nano, (px, py, pz)

                rel_sec, rel_nano = sec - t0_sec, nano - t0_nano
                if rel_nano < 0:
                    rel_sec -= 1
                    rel_nano += 1_000_000_000

                row = [rel_sec, rel_nano]
                row += [msg.contacts[i].wrench.force.z for i in range(4)]
                for i in range(12):
                    row += [msg.joints.position[i], msg.joints.velocity[i], msg.joints.acceleration[i], msg.joints.effort[i]]
                row += [
                    px - p0[0], py - p0[1], pz - p0[2],
                    msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z,
                    msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z,
                ]
                writer.writerow(row)


def main() -> None:
    if not ROOT.exists():
        raise SystemExit(f"root directory not found: {ROOT}")

    for traj_dir in ROOT.glob("*/Data_anymal/*"):
        rosbags_dir = traj_dir / "rosbags"
        bag_files = list(rosbags_dir.glob("*.bag")) if rosbags_dir.exists() else []
        if not bag_files:
            continue

        out_dir = OUTPUT_ROOT / traj_dir.parents[1].name / traj_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for bag_path in bag_files:
            output_csv = out_dir / f"{bag_path.stem}_bag.csv"
            print(f"[process_bag] {bag_path} -> {output_csv}")
            process_bag(bag_path, output_csv)


if __name__ == "__main__":
    main()
