"""Pulls embedded position ground truth out of a prepared sequence frame."""

from __future__ import annotations

import pandas as pd


def extract_position_ground_truth(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return columns local_x/local_y/local_z (+ t_abs if present), or an empty frame."""
    for src_cols in (("pos_x", "pos_y", "pos_z"), ("p_x", "p_y", "p_z")):
        if all(c in dataframe.columns for c in src_cols):
            gt = dataframe[list(src_cols)].copy()
            gt.columns = ["local_x", "local_y", "local_z"]
            if "t_abs" in dataframe.columns:
                gt["t_abs"] = dataframe["t_abs"].to_numpy()
            return gt
    return pd.DataFrame()
