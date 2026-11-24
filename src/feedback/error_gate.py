"""Error checks in front of the 150-count gate.

Distances follow docs/ERROR_GATE_PLAN.md. A cell is released only when its
distance is strictly above the threshold. The published line is not refit here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.feedback.published_hour import DAY_HOURS, fit_window

DISTANCES = ("level", "wape", "profile", "utility")
MODES = ("once", "daily")
GRIDS: dict[str, tuple[float, ...]] = {
    "level": (0.0, 0.05, 0.10, 0.15, 0.20),
    "wape": (0.0, 0.25, 0.35, 0.45, 0.55),
    "profile": (0.0, 0.15, 0.25, 0.35, 0.45),
    "utility": (0.0, 0.01, 0.02, 0.05, 0.10),
}


def _denominator(total: np.ndarray) -> np.ndarray:
    denom = np.asarray(total, dtype=np.float64)
    safe = np.array(denom, copy=True)
    safe[safe <= 0] = np.inf
    return safe


def level_gap(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Absolute 72-hour total gap, divided by the actual total."""
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    return np.abs(truth.sum(axis=0) - pred.sum(axis=0)) / _denominator(truth.sum(axis=0))


def wape_gap(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Sum of absolute hourly gaps, divided by the actual total."""
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    return np.abs(truth - pred).sum(axis=0) / _denominator(truth.sum(axis=0))


def profile_gap(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """WAPE of the 24 clock-hour means."""
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 2 or pred.shape[0] % DAY_HOURS:
        raise ValueError("profile gap expects whole days")
    pred_hour = pred.reshape(-1, DAY_HOURS, pred.shape[1]).mean(axis=0)
    true_hour = truth.reshape(-1, DAY_HOURS, truth.shape[1]).mean(axis=0)
    return np.abs(true_hour - pred_hour).sum(axis=0) / _denominator(true_hour.sum(axis=0))


def utility_gap(prediction: np.ndarray, target: np.ndarray, solve_old, eligible: np.ndarray | None = None) -> np.ndarray:
    """Mean relative MAE drop on a left-out day. Positive means the correction helps."""
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 2 or pred.shape[0] % DAY_HOURS:
        raise ValueError("utility gap expects whole days")
    n_days = pred.shape[0] // DAY_HOURS
    if n_days < 2:
        raise ValueError("utility gap needs two days")
    n_cells = pred.shape[1]
    total = np.zeros(n_cells, dtype=np.float64)
    keep = np.ones(n_cells, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    for hold in range(n_days):
        fit_idx = np.concatenate(
            [np.arange(day * DAY_HOURS, (day + 1) * DAY_HOURS) for day in range(n_days) if day != hold]
        )
        scale, shift, offsets = fit_window(pred[fit_idx], truth[fit_idx], keep, solve_old, with_hour=True)
        start = hold * DAY_HOURS
        base = pred[start : start + DAY_HOURS]
        corrected = base * scale.reshape(1, -1) + shift.reshape(1, -1) + offsets
        mae_base = np.abs(base - truth[start : start + DAY_HOURS]).mean(axis=0)
        mae_corr = np.abs(corrected - truth[start : start + DAY_HOURS]).mean(axis=0)
        usable = mae_base > 1e-8
        total[usable] += (mae_base[usable] - mae_corr[usable]) / mae_base[usable]
    return total / float(n_days)


def gaps(prediction: np.ndarray, target: np.ndarray, solve_old, eligible: np.ndarray | None = None) -> dict[str, np.ndarray]:
    return {
        "level": level_gap(prediction, target),
        "wape": wape_gap(prediction, target),
        "profile": profile_gap(prediction, target),
        "utility": utility_gap(prediction, target, solve_old, eligible),
    }


def release(distance: np.ndarray, tau: float, eligible: np.ndarray) -> np.ndarray:
    """Cells above ``tau`` that also pass the 150-count gate."""
    keep = np.asarray(eligible, dtype=bool).copy()
    values = np.asarray(distance, dtype=np.float64)
    keep &= np.isfinite(values) & (values > float(tau))
    return keep


def choose_threshold(
    rows: list[dict[str, Any]],
    baseline_ny: float,
    baseline_chi: float,
) -> dict[str, Any] | None:
    """Smallest threshold that does not raise either city's validation MAE.

    Returns ``None`` when every threshold is worse on at least one city, or when
    the best row matches the all-cells baseline on both cities.
    """
    allowed = []
    for row in rows:
        if float(row["ny"]) <= float(baseline_ny) + 1e-12 and float(row["chi"]) <= float(baseline_chi) + 1e-12:
            allowed.append(row)
    if not allowed:
        return None
    allowed.sort(key=lambda row: ((float(row["ny"]) + float(row["chi"])) / 2.0, float(row["tau"])))
    best = allowed[0]
    same = abs(float(best["ny"]) - float(baseline_ny)) <= 1e-12 and abs(float(best["chi"]) - float(baseline_chi)) <= 1e-12
    if same:
        return None
    return best
