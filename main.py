#!/usr/bin/env python3
"""Entry point: run the ESEKF over one recording, driven by an experiment YAML.

    python main.py --config config/tartanground_hmm_gmm.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from leg_odom.run.ekf_process import run_ekf_pipeline
from leg_odom.run.experiment_config import debug_enabled, load_experiment_yaml
from leg_odom.run.output_layout import prepare_run_output_dir
from leg_odom.run.post_ekf import run_post_ekf_analysis_and_eval

_REPO_ROOT = Path(__file__).resolve().parent


def run_experiment(config_path: Path, *, workspace_root: Path = _REPO_ROOT) -> int:
    cfg_path = config_path.expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (workspace_root / cfg_path).resolve()

    cfg = load_experiment_yaml(cfg_path)
    debug = debug_enabled(cfg)

    run_dir, resolved_cfg = prepare_run_output_dir(cfg, workspace_root=workspace_root, source_config_path=cfg_path)
    print(f"[run] output directory: {run_dir}")

    summary = run_ekf_pipeline(resolved_cfg, run_dir=run_dir, debug=debug, workspace_root=workspace_root)
    print(f"[ekf] wrote {run_dir / 'ekf_process_summary.json'}")
    if summary.ekf_history_csv:
        print(f"[ekf] history csv: {summary.ekf_history_csv}")

    run_post_ekf_analysis_and_eval(run_dir, resolved_cfg, summary)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leg odometry ESEKF with contact-detector ZUPT.")
    p.add_argument("--config", type=Path, required=True, help="Experiment YAML (see config/).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_experiment(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
