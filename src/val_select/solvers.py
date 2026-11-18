"""Closed-form per-cell solvers for the val-selection protocol.

Finite m>0 / kappa>0 agree with PR20 ``scripts/19_cell_affine_v2.py``.
Zero-strength rank-deficient (constant prediction) uses the OLS intercept
convention: alpha=0, c=rbar, i.e. a=1, b=rbar. Exact identity is a separate
branch (a=1, b=0) and is never approximated by a large finite strength.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from src.val_select.protocol import IDENTITY, NEAR_CONSTANT_REL, N_TRAIN_HOURS, is_identity


def _vec(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("empty hour series")
    if not np.isfinite(arr).all():
        raise ValueError("non-finite values in series")
    return arr


def is_near_constant(pred: np.ndarray) -> bool:
    """Predetermined near-rank-deficiency test on fixed DC-count units.

    A series is near-constant when range(p) <= NEAR_CONSTANT_REL * max(1, max|p|).
    Both zero-strength solvers share this test. It is not a claim of strict
    invariance under arbitrary unit rescaling.
    """
    arr = np.asarray(pred, dtype=np.float64).reshape(-1)
    span = float(np.ptp(arr))
    scale = max(1.0, float(np.max(np.abs(arr))))
    return span <= NEAR_CONSTANT_REL * scale


def near_constant_report(prediction: np.ndarray, eligible: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64)
    keep = np.asarray(eligible, dtype=bool)
    triggered = [int(i) for i in np.flatnonzero(keep) if is_near_constant(pred[:, int(i)])]
    return {
        "near_constant_rel": NEAR_CONSTANT_REL,
        "triggered_cells": triggered,
        "n_triggered": len(triggered),
        "unit_convention": "fixed_dc_clipped_count",
        "note": (
            "Predetermined numerical convention on fixed DC counts; "
            "shared by m=0 and kappa=0. Not a claim of strict unit-scale invariance."
        ),
    }


def mean_residual(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = _vec(prediction)
    truth = _vec(target)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target must have the same length")
    return float((truth - pred).mean())


def identity_ab() -> tuple[float, float]:
    return 1.0, 0.0


def zero_strength_constant_ab(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """OLS intercept-only when slope is unidentifiable or forced to 1 at m=κ=0."""
    return 1.0, mean_residual(prediction, target)


def solve_old(prediction: np.ndarray, target: np.ndarray, mult: Any) -> tuple[float, float]:
    """Published ridge with lambda = m * n. Identity is a separate branch."""
    if is_identity(mult):
        return identity_ab()
    if not math.isfinite(float(mult)) or float(mult) < 0:
        raise ValueError("multiplier must be a finite non-negative number or identity")
    pred = _vec(prediction)
    truth = _vec(target)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target must have the same length")
    m = float(mult)
    if m == 0.0:
        # Zero-strength old affine uses the same centered OLS path as kappa=0.
        # Positive m keeps the published ridge with lambda = m * n.
        return solve_centered(pred, truth, 0.0)
    lam = m * float(pred.size)
    gram = np.array(
        [[float(pred @ pred) + lam, float(pred.sum())], [float(pred.sum()), float(pred.size) + lam]],
        dtype=np.float64,
    )
    rhs = np.array([float(pred @ truth) + lam, float(truth.sum())], dtype=np.float64)
    try:
        scale, shift = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("old-affine normal equations are singular") from exc
    if not (math.isfinite(float(scale)) and math.isfinite(float(shift))):
        raise ValueError("old-affine solver returned a non-finite coefficient")
    return float(scale), float(shift)


def solve_centered(prediction: np.ndarray, target: np.ndarray, kappa: Any) -> tuple[float, float]:
    """Centered penalty. Returns (a, b) with a=1+alpha, b=c-alpha*pbar."""
    if is_identity(kappa):
        return identity_ab()
    if not math.isfinite(float(kappa)) or float(kappa) < 0:
        raise ValueError("kappa must be a finite non-negative number or identity")
    pred = _vec(prediction)
    truth = _vec(target)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target must have the same length")
    n = int(pred.size)
    kappa_f = float(kappa)
    center = float(pred.mean())
    residual = truth - pred
    if kappa_f == 0.0 and is_near_constant(pred):
        return zero_strength_constant_ab(pred, truth)
    centered = pred - center
    denom = float(centered @ centered) + kappa_f * n * center * center
    if denom <= 0.0:
        # kappa>0 and pbar=0 and S=0, or numerical zero: keep alpha=0, retain c.
        deviation = 0.0
    else:
        deviation = float(centered @ residual) / denom
    level = float(residual.sum()) / (n * (1.0 + kappa_f))
    scale = 1.0 + deviation
    shift = level - deviation * center
    if not (math.isfinite(scale) and math.isfinite(shift)):
        raise ValueError("centered solver returned a non-finite coefficient")
    return float(scale), float(shift)


def solve_shift(prediction: np.ndarray, target: np.ndarray, kappa: Any) -> tuple[float, float]:
    """Translation-only: alpha=0, c=rbar/(1+kappa), a=1, b=c."""
    if is_identity(kappa):
        return identity_ab()
    if not math.isfinite(float(kappa)) or float(kappa) < 0:
        raise ValueError("kappa must be a finite non-negative number or identity")
    c = mean_residual(prediction, target) / (1.0 + float(kappa))
    if not math.isfinite(c):
        raise ValueError("shift solver returned a non-finite coefficient")
    return 1.0, float(c)


def solver_for(method: str) -> Callable[..., tuple[float, float]]:
    if method == "old":
        return solve_old
    if method == "centered":
        return solve_centered
    if method == "shift":
        return solve_shift
    if method == "identity":
        def _identity(prediction: np.ndarray, target: np.ndarray, strength: Any = IDENTITY) -> tuple[float, float]:
            return identity_ab()

        return _identity
    raise ValueError(f"unknown method {method}")


def require_train_hours(prediction: np.ndarray, target: np.ndarray, expected: int = N_TRAIN_HOURS) -> None:
    pred = np.asarray(prediction)
    truth = np.asarray(target)
    if pred.shape != truth.shape or pred.ndim != 1:
        raise ValueError("per-cell train series must be 1-D and aligned")
    if pred.shape[0] != expected:
        raise ValueError(f"train hours {pred.shape[0]} != {expected}; refusing to drop hours")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("non-finite train values")


def fit_cells(
    prediction: np.ndarray,
    target: np.ndarray,
    eligible: np.ndarray,
    method: str,
    strength: Any,
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
    fn = solver_for(method)
    for index in np.flatnonzero(keep):
        require_train_hours(pred[:, index], truth[:, index])
        scale[index], shift[index] = fn(pred[:, index], truth[:, index], strength)
    if not (np.isfinite(scale).all() and np.isfinite(shift).all()):
        raise ValueError("non-finite fitted coefficients")
    return scale, shift


def apply_affine(prediction: np.ndarray, scale: np.ndarray, shift: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float64)
    scale_a = np.asarray(scale, dtype=np.float64)
    shift_b = np.asarray(shift, dtype=np.float64)
    if pred.ndim != 2 or scale_a.shape != (pred.shape[1],) or shift_b.shape != scale_a.shape:
        raise ValueError("scale and shift must match the cell axis")
    return pred * scale_a.reshape(1, -1) + shift_b.reshape(1, -1)
