"""The main loop: dataset -> per-timestep IMU predict + contact detection + ZUPT correction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from leg_odom.contact.base import ContactDetectorStepInput
from leg_odom.datasets.types import LegOdometrySequence
from leg_odom.eval.ekf_step_log import EkfStepLogWriter, build_ekf_step_log_row, empty_zupt_info, sanitize_sequence_slug
from leg_odom.filters.esekf import ErrorStateEkf, build_error_state_ekf
from leg_odom.filters.zupt_measurement import zupt_isotropic_meas_from_p_stance
from leg_odom.io.columns import IMU_ACCEL_COLS, IMU_GYRO_COLS, motor_position_cols, motor_torque_cols, motor_velocity_cols
from leg_odom.kinematics.base import BaseKinematics
from leg_odom.run.contact_factory import ContactStack, build_contact_stack
from leg_odom.run.dataset_factory import build_leg_odometry_dataset
from leg_odom.run.ekf_nominal_init import apply_nominal_init_from_timeline, ekf_initialize_nominal_from_data_enabled
from leg_odom.run.kinematics_factory import build_kinematics_backend


@dataclass
class EkfProcessSummary:
    robot_kinematics: str
    dataset_kind: str
    contact_detector: str
    sequence_name: str = ""
    median_rate_hz: float = 0.0
    ekf_history_csv: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        d = {
            "robot_kinematics": self.robot_kinematics,
            "dataset_kind": self.dataset_kind,
            "contact_detector": self.contact_detector,
            "sequence_name": self.sequence_name,
            "median_rate_hz": self.median_rate_hz,
        }
        if self.ekf_history_csv:
            d["ekf_history_csv"] = self.ekf_history_csv
        return d


def run_ekf_on_recording(
    recording: LegOdometrySequence,
    *,
    kin_model: BaseKinematics,
    filter_state: ErrorStateEkf,
    contact_stack: ContactStack,
    experiment_cfg: Mapping[str, Any] | None = None,
    history_csv_path: Path | None = None,
    debug: bool = False,
) -> tuple[str, float, str | None]:
    """Runs IMU prediction and, when a detector is configured, ZUPT correction over one recording.

    Returns (sequence_name, median_rate_hz, ekf_history_csv_path_or_none).
    """
    filter_state.reset()
    cfg_map = experiment_cfg if isinstance(experiment_cfg, Mapping) else {}
    foot_dets = contact_stack.per_foot
    if foot_dets is not None:
        for d in foot_dets:
            d.reset()

    timeline: DataFrame = recording.frames
    if ekf_initialize_nominal_from_data_enabled(cfg_map):
        apply_nominal_init_from_timeline(filter_state, timeline)

    gyro_cols, accel_cols = list(IMU_GYRO_COLS), list(IMU_ACCEL_COLS)
    motor_cols, vel_cols, tau_cols = list(motor_position_cols()), list(motor_velocity_cols()), list(motor_torque_cols())
    accel_gc = bool(recording.meta.get("accel_gravity_compensated", False))
    n_legs = kin_model.n_legs

    log_writer: EkfStepLogWriter | None = None
    hist_resolved: str | None = None
    if history_csv_path is not None:
        log_writer = EkfStepLogWriter(Path(history_csv_path), n_legs=n_legs)
        hist_resolved = str(Path(history_csv_path).resolve())

    try:
        for k in range(len(timeline)):
            row = timeline.iloc[k]
            dt_s = float(row["dt"])
            gyro = row[gyro_cols].to_numpy(dtype=np.float64)
            accel = row[accel_cols].to_numpy(dtype=np.float64)
            filter_state.imu_predict(dt_s, gyro, accel, accel_gravity_compensated=accel_gc)

            gyro_corr = gyro - filter_state.bias_gyro
            accel_corr = accel - filter_state.bias_accel
            q_all = row[motor_cols].to_numpy(dtype=np.float64)
            dq_all = row.reindex(vel_cols, fill_value=0.0).to_numpy(dtype=np.float64)
            tau_all = row.reindex(tau_cols, fill_value=0.0).to_numpy(dtype=np.float64)

            stance = [False] * n_legs
            contact_scores = [0.0] * n_legs
            contact_vars = [float("nan")] * n_legs
            foot_kin: list[tuple[NDArray, NDArray, NDArray]] = []
            stance_legs: list[dict[str, Any]] = []

            for leg_index in range(n_legs):
                sl = slice(leg_index * kin_model.joints_per_leg, (leg_index + 1) * kin_model.joints_per_leg)
                q_leg, dq_leg, tau_leg = q_all[sl], dq_all[sl], tau_all[sl]
                p_b = np.asarray(kin_model.fk(leg_index, q_leg), dtype=np.float64).reshape(3)
                jacobian = np.asarray(kin_model.J_analytical(leg_index, q_leg), dtype=np.float64).reshape(3, -1)
                v_foot_body = np.cross(gyro_corr, p_b) + jacobian @ dq_leg
                foot_kin.append((p_b, jacobian, dq_leg))

                if foot_dets is not None:
                    step_in = ContactDetectorStepInput(
                        p_foot_body=np.ascontiguousarray(p_b),
                        v_foot_body=np.ascontiguousarray(v_foot_body),
                        q_leg=np.ascontiguousarray(q_leg),
                        dq_leg=np.ascontiguousarray(dq_leg),
                        tau_leg=np.ascontiguousarray(tau_leg),
                        gyro_body_corrected=np.ascontiguousarray(gyro_corr),
                        accel_body_corrected=np.ascontiguousarray(accel_corr),
                    )
                    est = foot_dets[leg_index].update(step_in)
                    stance[leg_index] = bool(est.stance)
                    contact_scores[leg_index] = float(est.p_stance)
                    sigma_sq, r_foot = zupt_isotropic_meas_from_p_stance(float(est.p_stance))
                    contact_vars[leg_index] = sigma_sq
                    if est.stance and np.all(np.isfinite(r_foot)):
                        stance_legs.append(
                            {"leg_id": leg_index, "p_foot_body": p_b, "J": jacobian, "qdot": dq_leg, "R_foot": r_foot, "qscore": float(est.p_stance)}
                        )

            zupt_info = filter_state.update_zupt(stance_legs, gyro) if stance_legs else empty_zupt_info()

            if log_writer is not None:
                log_writer.write_row(
                    build_ekf_step_log_row(
                        row, filter_state, gyro_raw=gyro, foot_kin=foot_kin, stance=stance,
                        contact_score=contact_scores, contact_zupt_var=contact_vars, zupt_info=zupt_info, n_legs=n_legs,
                    )
                )
    finally:
        if log_writer is not None:
            log_writer.close()

    if debug:
        print(
            f"[ekf] {recording.sequence_name!r}: steps={len(timeline)}, hz~{recording.median_rate_hz:.1f}, "
            f"detector={contact_stack.detector_id!r}"
        )

    return recording.sequence_name, float(recording.median_rate_hz), hist_resolved


def run_ekf_pipeline(
    resolved_cfg: Mapping[str, Any], *, run_dir: Path | None = None, debug: bool = False, workspace_root: Path | None = None
) -> EkfProcessSummary:
    dataset = build_leg_odometry_dataset(resolved_cfg)
    kin_model = build_kinematics_backend(resolved_cfg)
    recording = dataset[0]
    contact_stack = build_contact_stack(resolved_cfg, recording=recording, kin_model=kin_model, workspace_root=workspace_root)

    summary = EkfProcessSummary(
        robot_kinematics=str(resolved_cfg["robot"]["kinematics"]).lower(),
        dataset_kind=str(resolved_cfg["dataset"]["kind"]).lower(),
        contact_detector=contact_stack.detector_id,
    )

    ekf = build_error_state_ekf(resolved_cfg, workspace_root=workspace_root)
    hist_path = Path(run_dir) / f"ekf_history_{sanitize_sequence_slug(recording.sequence_name)}.csv" if run_dir is not None else None
    seq_name, hz, hist = run_ekf_on_recording(
        recording, kin_model=kin_model, filter_state=ekf, contact_stack=contact_stack,
        experiment_cfg=resolved_cfg, history_csv_path=hist_path, debug=debug,
    )
    summary.sequence_name, summary.median_rate_hz, summary.ekf_history_csv = seq_name, hz, hist

    if run_dir is not None:
        (Path(run_dir) / "ekf_process_summary.json").write_text(json.dumps(summary.to_json_dict(), indent=2))

    return summary
