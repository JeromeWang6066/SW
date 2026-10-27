"""Weekday 24h Pearson ρ̃ multiplier — Group 3 hook.

Approved direction (advisor):
  Non-learnable weekday 24-hour Pearson correlation multiplier modulating KD,
  based on region / hour profile similarity between source and target.

NOT finalized:
  - Region matching / alignment across cities with different grids
  - Multi-channel profile definition (flow only vs POI-augmented, etc.)

Do not treat the helpers below as the final paper formula.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def weekday_24h_profile(
    flow: np.ndarray,
    *,
    weekday_mask: Optional[np.ndarray] = None,
    region_axis: Tuple[int, ...] = (1, 2),
) -> np.ndarray:
    """Build a per-region mean weekday 24h profile.

    Parameters
    ----------
    flow :
        Time-major array ``(T, H, W)`` or ``(T, N)``.
    weekday_mask :
        Boolean length-T mask selecting weekdays. If None, all timesteps are
        used (caller should pass a real calendar mask later).
    region_axis :
        Axes to flatten into regions when ``flow`` is grid-shaped.

    Returns
    -------
    profile : ndarray, shape ``(N, 24)``
        Mean value for each region and hour-of-day on selected days.

    Notes
    -----
    Formula sketch (subject to change)::

        for each region r:
            for hour h in 0..23:
                profile[r, h] = mean_t { flow[t, r] | weekday(t) and hour(t)=h }

        ρ̃(r_s, r_t) = PearsonCorr( profile_source[r_s], profile_target[r_t] )

    Matching which ``r_s`` pairs with which ``r_t`` is **TODO**.
    """
    x = np.asarray(flow, dtype=np.float64)
    if x.ndim == 3:
        t, h, w = x.shape
        x = x.reshape(t, h * w)
    elif x.ndim != 2:
        raise ValueError(f"expected (T,H,W) or (T,N), got {x.shape}")

    t, n = x.shape
    if weekday_mask is None:
        weekday_mask = np.ones(t, dtype=bool)
    else:
        weekday_mask = np.asarray(weekday_mask, dtype=bool)
        if weekday_mask.shape != (t,):
            raise ValueError("weekday_mask must have shape (T,)")

    # Hour-of-day assumes hourly series starting at an aligned midnight — TODO verify.
    hours = np.arange(t) % 24
    profile = np.zeros((n, 24), dtype=np.float64)
    counts = np.zeros(24, dtype=np.int64)
    for h in range(24):
        sel = weekday_mask & (hours == h)
        counts[h] = int(sel.sum())
        if counts[h] == 0:
            continue
        profile[:, h] = x[sel].mean(axis=0)
    return profile


def pearson_corr_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise Pearson correlation between ``(N, D)`` profiles."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    a0 = a - a.mean(axis=1, keepdims=True)
    b0 = b - b.mean(axis=1, keepdims=True)
    num = (a0 * b0).sum(axis=1)
    den = np.sqrt((a0 ** 2).sum(axis=1) * (b0 ** 2).sum(axis=1)) + eps
    return num / den


def compute_weekday_pearson_mask(
    source_flow: np.ndarray,
    target_flow: np.ndarray,
    *,
    matching: Optional[np.ndarray] = None,
    source_weekday_mask: Optional[np.ndarray] = None,
    target_weekday_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a non-learnable ρ̃ vector/matrix for KD modulation.

    If ``matching`` is None, requires equal region counts and assumes identity
    alignment — **almost certainly wrong for cross-city grids**. Raise unless
    shapes match, and always document the assumption in experiment notes.

    Returns
    -------
    rho : ndarray, shape ``(N_target,)``
        Pearson ρ̃ per target region (placeholder aggregation).
    """
    src_p = weekday_24h_profile(source_flow, weekday_mask=source_weekday_mask)
    tgt_p = weekday_24h_profile(target_flow, weekday_mask=target_weekday_mask)

    if matching is None:
        if src_p.shape[0] != tgt_p.shape[0]:
            raise NotImplementedError(
                "Cross-city region matching is TODO (Group 3/4). "
                "Provide a matching index array or equal-N identity assumption."
            )
        paired_src = src_p
    else:
        matching = np.asarray(matching, dtype=np.int64)
        paired_src = src_p[matching]

    return pearson_corr_rows(paired_src, tgt_p)
