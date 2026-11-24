"""Let cells below the 150 gate join the daily correction after a holdout vote.

The rule is docs/ADD_BELOW_GATE_PLAN.md. Cells that already passed the volume
gate are left untouched. A vote counts only when the absolute error falls.
"""

from __future__ import annotations

import numpy as np

from src.feedback.published_hour import DAY_HOURS, fit_window

HOLDOUT_MARGIN = 1e-6
VOTE_COUNTS = (1, 2, 3)
PRIMARY_VOTES = 2


def holdout_wins(
    prediction: np.ndarray,
    target: np.ndarray,
    solve_old,
    candidates: np.ndarray,
) -> np.ndarray:
    """How many left-out days the correction beats the fine-tune, per cell.

    Cells outside ``candidates`` stay at 0 and are not fit.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    keep = np.asarray(candidates, dtype=bool)
    if pred.shape != truth.shape or pred.ndim != 2 or pred.shape[0] % DAY_HOURS:
        raise ValueError("holdout wins expect whole days")
    if keep.shape != (pred.shape[1],):
        raise ValueError("one candidate flag per cell")
    n_days = pred.shape[0] // DAY_HOURS
    if n_days < 2:
        raise ValueError("holdout wins need two days")
    wins = np.zeros(pred.shape[1], dtype=np.int64)
    if not np.any(keep):
        return wins
    for hold in range(n_days):
        fit_idx = np.concatenate(
            [np.arange(day * DAY_HOURS, (day + 1) * DAY_HOURS) for day in range(n_days) if day != hold]
        )
        scale, shift, offsets = fit_window(pred[fit_idx], truth[fit_idx], keep, solve_old, with_hour=True)
        start = hold * DAY_HOURS
        base = pred[start : start + DAY_HOURS]
        corrected = base * scale.reshape(1, -1) + shift.reshape(1, -1) + offsets
        held = truth[start : start + DAY_HOURS]
        base_ae = np.abs(base - held).sum(axis=0)
        corr_ae = np.abs(corrected - held).sum(axis=0)
        wins[keep] += (base_ae[keep] - corr_ae[keep] > HOLDOUT_MARGIN).astype(np.int64)
    return wins


def admit(wins: np.ndarray, votes: int, eligible: np.ndarray) -> np.ndarray:
    """Volume-gated cells stay in. Other cells need ``votes`` winning holdouts."""
    if votes < 1:
        raise ValueError("votes must be positive")
    keep = np.asarray(eligible, dtype=bool).copy()
    keep |= np.asarray(wins) >= int(votes)
    return keep
