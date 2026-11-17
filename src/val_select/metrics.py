"""Citywide valid-mask metrics in float64, matching PR20/17 score_arrays."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def score_arrays(prediction: np.ndarray, target: np.ndarray, value_scale: float) -> dict[str, Any]:
    """MAE / RMSE / positive-only MAPE over every provided (already valid-masked) point.

    Accumulation is float64. Predictions are not clipped to be non-negative.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError("prediction and target must be 2-D and aligned")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("non-finite values in score arrays")
    if not math.isfinite(float(value_scale)) or float(value_scale) <= 0:
        raise ValueError("value scale must be a positive finite number")
    hours, n_cells = pred.shape
    valid_points = float(hours * n_cells)
    if valid_points <= 0:
        raise ValueError("no points to score")
    gap = pred - truth
    abs_sum = float(np.abs(gap).sum())
    se_norm = float(np.sum((gap / float(value_scale)) ** 2))
    positive = truth > 0
    mape_points = float(np.count_nonzero(positive))
    if mape_points <= 0:
        mape = float("nan")
    else:
        mape = float(np.abs(gap[positive] / truth[positive]).sum()) / mape_points * 100.0
    return {
        "mae": abs_sum / valid_points,
        "rmse": (se_norm / valid_points) ** 0.5 * float(value_scale),
        "mape": mape,
        "hours": int(hours),
        "n_cells": int(n_cells),
        "n_points": int(valid_points),
        "n_mape_points": int(mape_points),
    }


def gated_mae(prediction: np.ndarray, target: np.ndarray, eligible: np.ndarray) -> float:
    keep = np.asarray(eligible, dtype=bool)
    if keep.ndim != 1 or keep.shape[0] != prediction.shape[1]:
        raise ValueError("eligible mask must match the cell axis")
    if int(keep.sum()) <= 0:
        raise ValueError("no gated cells")
    pred = np.asarray(prediction, dtype=np.float64)[:, keep]
    truth = np.asarray(target, dtype=np.float64)[:, keep]
    gap = pred - truth
    return float(np.abs(gap).sum()) / float(gap.size)
