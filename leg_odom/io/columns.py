"""Column-name conventions for the split logs (IMU CSV + kinematics CSV)."""

from __future__ import annotations

TIME_SEC_COL = "sec"
TIME_NANOSEC_COL = "nanosec"

IMU_GYRO_COLS = ("gyro_x", "gyro_y", "gyro_z")
IMU_ACCEL_COLS = ("accel_x", "accel_y", "accel_z")
IMU_CORE_COLS = IMU_GYRO_COLS + IMU_ACCEL_COLS

# Body orientation, only required when the accelerometer is gravity-compensated.
IMU_BODY_QUAT_COLS = ("ori_qx", "ori_qy", "ori_qz", "ori_qw")

# 12-DoF leg layout (3 joints x 4 legs), radians / rad/s / Nm.
def motor_position_cols() -> tuple[str, ...]:
    return tuple(f"motor_{i}_q" for i in range(12))


def motor_velocity_cols() -> tuple[str, ...]:
    return tuple(f"motor_{i}_dq" for i in range(12))


def motor_torque_cols() -> tuple[str, ...]:
    return tuple(f"motor_{i}_tau_est" for i in range(12))


# Optional per-foot load proxy. Not used by the contact detectors in this repo (they are
# kinematics-only), but recordings may still carry it.
FOOT_FORCE_COLS = tuple(f"foot_force_{i}" for i in range(4))
