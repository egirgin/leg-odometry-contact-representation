"""After the EKF run: trajectory plot + evaluation_metrics.csv under <run_dir>/plots/."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from leg_odom.eval.metrics import TrajectoryEvaluator
from leg_odom.eval.plots import plot_trajectory
from leg_odom.run.dataset_factory import build_leg_odometry_dataset
from leg_odom.run.ekf_process import EkfProcessSummary


def run_post_ekf_analysis_and_eval(run_dir: Path, resolved_cfg: Mapping[str, Any], summary: EkfProcessSummary, *, output_subdir: str = "plots") -> None:
    if not summary.ekf_history_csv:
        return

    out_dir = Path(run_dir) / output_subdir
    recording = build_leg_odometry_dataset(resolved_cfg)[0]
    hist = pd.read_csv(summary.ekf_history_csv)
    gt_df = recording.position_ground_truth

    row = TrajectoryEvaluator().evaluate(hist, gt_df, sequence_name=summary.sequence_name, print_report=False)
    TrajectoryEvaluator.write_metrics_csv(out_dir / "evaluation_metrics.csv", [row])
    plot_trajectory(hist, gt_df if gt_df is not None and not gt_df.empty else None, out_dir / "trajectory.png")
    print(f"[post_ekf] wrote evaluation metrics and trajectory plot under {out_dir}")
