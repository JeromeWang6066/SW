"""Dual-kappa closed-form solver and cell fitting.

Diagonal pairs (kappa_alpha == kappa_c == kappa) must be numerically identical
to the frozen centered single-kappa solver, including near-constant and
denominator degeneracy. Off-diagonal pairs generalize:

    alpha = T / (S + kappa_alpha * n * mu^2)
    c     = rbar / (1 + kappa_c)

Identity is a separate branch. Non-gated cells keep a=1, b=0 (original FT).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.dual_kappa_pilot.protocol import IDENTITY, N_TRAIN_HOURS, is_identity
from src.val_select.solvers import (
    apply_affine,
    identity_ab,
    is_near_constant,
    near_constant_report,
    require_train_hours,
    solve_centered,
    solve_old,
    zero_strength_constant_ab,
)


def _vec(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("empty hour series")
    if not np.isfinite(arr).all():
        raise ValueError("non-finite values in series")
    return arr


def solve_dual(prediction: np.ndarray, target: np.ndarray, kappa_alpha: Any, kappa_c: Any) -> tuple[float, float]:
    """Return (a, b) with a=1+alpha, b=c-alpha*mu."""
    if is_identity(kappa_alpha) or is_identity(kappa_c):
        if is_identity(kappa_alpha) and is_identity(kappa_c):
            return identity_ab()
        raise ValueError("identity is a separate candidate and cannot be mixed with a finite kappa")
    if not math.isfinite(float(kappa_alpha)) or float(kappa_alpha) < 0:
        raise ValueError("kappa_alpha must be a finite non-negative number")
    if not math.isfinite(float(kappa_c)) or float(kappa_c) < 0:
        raise ValueError("kappa_c must be a finite non-negative number")
    kappa_a = float(kappa_alpha)
    kappa_b = float(kappa_c)
    # Diagonal pairs are exactly the inherited single-kappa path.
    if kappa_a == kappa_b:
        return solve_centered(prediction, target, kappa_a)
    pred = _vec(prediction)
    truth = _vec(target)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target must have the same length")
    n = int(pred.size)
    center = float(pred.mean())
    residual = truth - pred
    if kappa_a == 0.0 and is_near_constant(pred):
        # Unidentifiable slope at zero slope-penalty: alpha=0, shrink c only.
        level = float(residual.mean()) / (1.0 + kappa_b)
        return 1.0, float(level)
    centered = pred - center
    denom = float(centered @ centered) + kappa_a * n * center * center
    if denom <= 0.0:
        deviation = 0.0
    else:
        deviation = float(centered @ residual) / denom
    level = float(residual.sum()) / (n * (1.0 + kappa_b))
    scale = 1.0 + deviation
    shift = level - deviation * center
    if not (math.isfinite(scale) and math.isfinite(shift)):
        raise ValueError("dual-kappa solver returned a non-finite coefficient")
    return float(scale), float(shift)


def solve_candidate(prediction: np.ndarray, target: np.ndarray, candidate: Any) -> tuple[float, float]:
    if is_identity(candidate):
        return identity_ab()
    if isinstance(candidate, (tuple, list)) and len(candidate) == 2:
        return solve_dual(prediction, target, candidate[0], candidate[1])
    raise ValueError(f"unrecognized candidate {candidate!r}")


def fit_cells_dual(
    prediction: np.ndarray,
    target: np.ndarray,
    eligible: np.ndarray,
    candidate: Any,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    keep = np.asarray(eligible, dtype=bool)
    if pred.ndim != 2 or pred.shape != truth.shape or keep.shape != (pred.shape[1],):
        raise ValueError("expected prediction/target (hours, cells) and one flag per cell")
    if pred.shape[0] != N_TRAIN_HOURS:
        raise ValueError(f"train hours {pred.shape[0]} != {N_TRAIN_HOURS}; refusing to drop hours")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("non-finite train values")
    scale = np.ones(pred.shape[1], dtype=np.float64)
    shift = np.zeros(pred.shape[1], dtype=np.float64)
    for index in np.flatnonzero(keep):
        require_train_hours(pred[:, index], truth[:, index])
        scale[index], shift[index] = solve_candidate(pred[:, index], truth[:, index], candidate)
    if not (np.isfinite(scale).all() and np.isfinite(shift).all()):
        raise ValueError("non-finite fitted coefficients")
    return scale, shift


def fit_old_m1(prediction: np.ndarray, target: np.ndarray, eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    keep = np.asarray(eligible, dtype=bool)
    if pred.ndim != 2 or pred.shape != truth.shape or keep.shape != (pred.shape[1],):
        raise ValueError("expected prediction/target (hours, cells) and one flag per cell")
    if pred.shape[0] != N_TRAIN_HOURS:
        raise ValueError(f"train hours {pred.shape[0]} != {N_TRAIN_HOURS}; refusing to drop hours")
    scale = np.ones(pred.shape[1], dtype=np.float64)
    shift = np.zeros(pred.shape[1], dtype=np.float64)
    for index in np.flatnonzero(keep):
        require_train_hours(pred[:, index], truth[:, index])
        scale[index], shift[index] = solve_old(pred[:, index], truth[:, index], 1.0)
    return scale, shift


def fit_new_k1(prediction: np.ndarray, target: np.ndarray, eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return fit_cells_dual(prediction, target, eligible, (1.0, 1.0))


def fit_identity(prediction: np.ndarray, target: np.ndarray, eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float64)
    scale = np.ones(pred.shape[1], dtype=np.float64)
    shift = np.zeros(pred.shape[1], dtype=np.float64)
    return scale, shift


__all__ = [
    "apply_affine",
    "fit_cells_dual",
    "fit_identity",
    "fit_new_k1",
    "fit_old_m1",
    "identity_ab",
    "is_near_constant",
    "near_constant_report",
    "solve_candidate",
    "solve_dual",
    "zero_strength_constant_ab",
]
