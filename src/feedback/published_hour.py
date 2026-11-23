"""Daily refit of the published line plus the kappa-3 clock-hour bias.

The numbers come from ``solve_old(..., 1)`` and from ``hour_offsets`` in
``scripts/23_cell_affine_next.py``. A day is corrected from earlier days only.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np

from src.utils.pearson_weekday_profile import CALENDAR_START

DAY_HOURS = 24
HOUR_KAPPA = 3.0
WINDOW_DAYS = 3
TRAIN_DAY = 243  # 2016-08-31, label hour 5832
VAL_DAY = 246  # 2016-09-03
TEST_DAY = 306  # 2016-11-02
SERIES_DAY = 366  # 2017-01-01
PREFIX_HOUR = 5592
TRAIN_HOUR = 5832
VAL_HOUR = 5904
TEST_HOUR = 7344
SERIES_HOUR = 8784
ARMS = ("B0", "O72", "OL", "RL", "CL", "CL72")


def hour_offsets(residual: np.ndarray, kappa: float = HOUR_KAPPA) -> np.ndarray:
    """Shrink each clock-hour mean toward zero. Same body as script 23."""
    resid = np.asarray(residual, dtype=np.float64).reshape(-1)
    if resid.size % DAY_HOURS:
        raise ValueError("residual length must be a whole number of days")
    folded = resid.reshape(-1, DAY_HOURS)
    return folded.sum(axis=0) / (folded.shape[0] + float(kappa))


def is_weekend(day: int) -> bool:
    """Saturday or Sunday. Monday 2016-09-05 stays a weekday."""
    return (CALENDAR_START + timedelta(days=int(day))).weekday() >= 5


def day_date(day: int):
    return CALENDAR_START + timedelta(days=int(day))


def previous_days(day: int, k: int = WINDOW_DAYS, *, same_class: bool) -> list[int]:
    """The ``k`` days strictly before ``day``, oldest first."""
    if k < 1:
        raise ValueError("window must contain at least one day")
    if not same_class:
        if day < k:
            return []
        return list(range(day - k, day))
    want = is_weekend(day)
    found: list[int] = []
    cursor = day - 1
    while cursor >= 0 and len(found) < k:
        if is_weekend(cursor) == want:
            found.append(cursor)
        cursor -= 1
    found.reverse()
    return found


def window_for(day: int, mode: str) -> list[int]:
    if mode in ("O72", "OL"):
        return [TRAIN_DAY, TRAIN_DAY + 1, TRAIN_DAY + 2]
    if mode == "CL72":
        return previous_days(day, WINDOW_DAYS, same_class=False)
    if mode in ("RL", "CL"):
        return previous_days(day, WINDOW_DAYS, same_class=True)
    if mode == "B0":
        return []
    raise ValueError(f"unknown mode {mode}")


def _hours(days: list[int]) -> np.ndarray:
    return np.concatenate([np.arange(day * DAY_HOURS, (day + 1) * DAY_HOURS) for day in days])


def fit_window(
    prediction: np.ndarray,
    target: np.ndarray,
    eligible: np.ndarray,
    solve_old,
    *,
    with_hour: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One (a, b, s) per cell. Cells outside ``eligible`` stay at identity."""
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    keep = np.asarray(eligible, dtype=bool)
    if pred.shape != truth.shape or pred.ndim != 2 or keep.shape != (pred.shape[1],):
        raise ValueError("expected (hours, cells) and one flag per cell")
    scale = np.ones(pred.shape[1], dtype=np.float64)
    shift = np.zeros(pred.shape[1], dtype=np.float64)
    offsets = np.zeros((DAY_HOURS, pred.shape[1]), dtype=np.float64)
    for index in np.flatnonzero(keep):
        found_a, found_b = solve_old(pred[:, index], truth[:, index], 1.0)
        scale[index] = found_a
        shift[index] = found_b
        if with_hour:
            residual = truth[:, index] - (found_a * pred[:, index] + found_b)
            offsets[:, index] = hour_offsets(residual, HOUR_KAPPA)
    return scale, shift, offsets


def apply_day(prediction: np.ndarray, scale: np.ndarray, shift: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float64)
    if pred.shape != (DAY_HOURS, scale.shape[0]) or offsets.shape != (DAY_HOURS, scale.shape[0]):
        raise ValueError("a day must be 24 hours and match the cell axis")
    clock = np.arange(DAY_HOURS)
    out = pred.copy()
    moved = (scale != 1.0) | (shift != 0.0) | np.any(offsets != 0.0, axis=0)
    out[:, moved] = pred[:, moved] * scale[moved].reshape(1, -1) + shift[moved].reshape(1, -1) + offsets[clock][:, moved]
    return out


def correct_range(
    prediction: np.ndarray,
    target: np.ndarray,
    eligible: np.ndarray,
    solve_old,
    mode: str,
    first_day: int,
    last_day: int,
) -> np.ndarray:
    """Correct days ``[first_day, last_day)``. Hours are absolute label hours.

    ``prediction`` and ``target`` are indexed by label hour and are never
    replaced by the corrected values. A day that does not have three earlier
    days stays at the fine-tune.
    """
    if mode not in ARMS:
        raise ValueError(f"unknown mode {mode}")
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError("prediction and target must share the hour-by-cell shape")
    out = pred.copy()
    if mode == "B0":
        return out
    with_hour = mode in ("OL", "CL", "CL72")
    cached: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    for day in range(first_day, last_day):
        days = window_for(day, mode)
        if len(days) < WINDOW_DAYS:
            continue
        hours = _hours(days)
        if int(hours.min()) < 0 or int(hours.max()) >= pred.shape[0]:
            raise ValueError(f"{mode} day {day} reads hour {int(hours.max())} outside the series")
        if mode in ("O72", "OL"):
            if cached is None:
                if not np.isfinite(pred[hours]).all() or not np.isfinite(truth[hours]).all():
                    raise ValueError(f"{mode} training window contains a non-finite value")
                cached = fit_window(pred[hours], truth[hours], eligible, solve_old, with_hour=with_hour)
            scale, shift, offsets = cached
        else:
            if not np.isfinite(pred[hours]).all() or not np.isfinite(truth[hours]).all():
                raise ValueError(f"{mode} day {day} window contains a non-finite value")
            scale, shift, offsets = fit_window(pred[hours], truth[hours], eligible, solve_old, with_hour=with_hour)
        start = day * DAY_HOURS
        out[start : start + DAY_HOURS] = apply_day(pred[start : start + DAY_HOURS], scale, shift, offsets)
    return out


def weekday_rows(first_day: int, last_day: int) -> np.ndarray:
    """Boolean mask over the hours of ``[first_day, last_day)``, true on Monday–Friday."""
    flags = []
    for day in range(first_day, last_day):
        flags.extend([not is_weekend(day)] * DAY_HOURS)
    return np.asarray(flags, dtype=bool)
