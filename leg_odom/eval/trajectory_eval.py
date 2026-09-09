"""CLI: trajectory metrics from an EKF history CSV + ground truth -> evaluation_metrics.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from leg_odom.eval.metrics import TrajectoryEvaluator, time_alignment_report
from leg_odom.io.ground_truth import extract_position_ground_truth


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Trajectory evaluation vs embedded ground truth.")
    p.add_argument("--ekf-csv", type=Path, required=True)
    p.add_argument("--gt-csv", type=Path, default=None)
    p.add_argument("--merged-csv", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--sequence-name", type=str, default="")
    p.add_argument("--check-only", action="store_true", help="Print time-overlap diagnostic and exit.")
    args = p.parse_args(argv)

    hist = pd.read_csv(args.ekf_csv)
    if args.gt_csv and args.gt_csv.is_file():
        gt_df = extract_position_ground_truth(pd.read_csv(args.gt_csv))
    elif args.merged_csv and args.merged_csv.is_file():
        gt_df = extract_position_ground_truth(pd.read_csv(args.merged_csv))
    else:
        print("[trajectory_eval] need --gt-csv or --merged-csv with embedded position GT")
        return 1

    if args.check_only:
        report = time_alignment_report(hist, gt_df)
        print(json.dumps(report, indent=2, default=str))
        if report.get("error") or int(report.get("n_gt_in_overlap") or 0) < 2:
            return 1
        return 0

    out_dir = args.output_dir or args.ekf_csv.parent
    row = TrajectoryEvaluator().evaluate(hist, gt_df, sequence_name=args.sequence_name, print_report=True)
    TrajectoryEvaluator.write_metrics_csv(out_dir / "evaluation_metrics.csv", [row])
    print(f"[trajectory_eval] wrote {out_dir / 'evaluation_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
