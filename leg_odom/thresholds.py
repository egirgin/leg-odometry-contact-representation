"""Numeric constants used to sanitize sensor data and guard degenerate fits."""

# IMU sanitization (leg_odom.io.imu_sanitize)
IMU_GRAVITY_REMOVED_MEAN_MAG_THRESHOLD = 3.0  # below this, accel is treated as gravity-removed
IMU_GYRO_MEDIAN_NORM_DEG_S_HINT = 10.0  # above this, gyro is probably deg/s instead of rad/s
IMU_FLU_SPECIFIC_FORCE_MAG_MIN = 7.0  # at-rest specific force should be near 9.81 m/s^2
IMU_FLU_SPECIFIC_FORCE_MAG_MAX = 12.0
IMU_FLU_SPECIFIC_FORCE_MAX_TILT_DEG = 35.0  # max angle between mean accel and body +Z at rest
IMU_VECTOR_NEAR_ZERO_NORM = 1e-9

# Timebase reconstruction (leg_odom.io.timebase)
TIMEBASE_RATE_FALLBACK_HZ = 400.0
TIMEBASE_MIN_POSITIVE_DT_SAMPLES = 10
TIMEBASE_DT_CLIP_MIN_S = 1e-4
TIMEBASE_DT_CLIP_MAX_S = 0.2

# Kinematics (leg_odom.kinematics)
KINEMATICS_NUMERICAL_JACOBIAN_STEP = 1e-6  # forward-difference step, rad
