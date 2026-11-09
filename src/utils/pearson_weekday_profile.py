"""Weekday 24h Pearson ρ̃ multiplier — Group 3 hook.

Approved direction (advisor, updated 2026-11-05):
  Pearson correlation first chooses the method: high similarity -> fine-tune,
  medium similarity -> distillation. Thresholds 0.8 / 0.5 are examples only.
  The non-learnable weekday 24-hour multiplier stays a hook until Group 5
  shows distillation wins in the medium band. See
  docs/advisor/2026-11-05-meeting.md.

NOT finalized:
  - Region matching / alignment across cities with different grids
  - Multi-channel profile definition (flow only vs POI-augmented, etc.)

Do not treat the helpers below as the final paper formula.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional, Tuple

import numpy as np

# CrossTReS Table 2 lists New York taxi/bike as 1 Jan–31 Dec 2016.
# Chicago and Washington use the same 8784-hour length, so they share this origin.
# Hour 0 is 00:00. No daylight-saving insertion or deletion is applied.
CALENDAR_START = date(2016, 1, 1)
EXPECTED_SERIES_HOURS = 8784
VAL_OR_TEST_HOURS = 60 * 24
TRAIN_PREFIX_HOURS = EXPECTED_SERIES_HOURS - 2 * VAL_OR_TEST_HOURS
LABEL_LAG_HOURS = 6
FEW_SHOT_LABEL_HOURS = 3 * 24
PROFILE_STD_EPS = 1e-8
BIN_LOW = 0.5
BIN_HIGH = 0.8
BIN_AGREEMENT_GATE = 0.7


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


def hourly_calendar(
    num_timesteps: int,
    start: date = CALENDAR_START,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hour-of-day and Monday–Friday mask for a series that starts at midnight.

    Parameters
    ----------
    num_timesteps :
        Length ``T``. Must be a multiple of 24.
    start :
        Calendar date of hour 0. Default is 2016-01-01, a Friday.

    Returns
    -------
    hour : int64 ndarray, shape ``(T,)``
        ``0 .. 23``. Index ``t`` is ``start`` plus ``t`` hours, without a
        daylight-saving adjustment.
    is_weekday : bool ndarray, shape ``(T,)``
        True on Monday through Friday.
    """
    if num_timesteps < 0 or num_timesteps % 24 != 0:
        raise ValueError(f"T={num_timesteps} must be a non-negative multiple of 24")
    hour = np.arange(num_timesteps, dtype=np.int64) % 24
    n_days = num_timesteps // 24
    day_is_weekday = np.fromiter(
        ((start + timedelta(days=i)).weekday() < 5 for i in range(n_days)),
        dtype=bool,
        count=n_days,
    )
    return hour, np.repeat(day_is_weekday, 24)


def grid_hour_profile(
    flow: np.ndarray,
    select: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean 24-hour profile on every cell, plus how many hours were averaged.

    Parameters
    ----------
    flow :
        ``(T, H, W)`` counts. ``T`` must be a multiple of 24 and hour 0 of this
        array must be local midnight. A train-prefix slice is valid only when
        it still starts at midnight.
    select :
        Boolean length-``T`` mask, typically weekdays or weekends.

    Returns
    -------
    profile : float64 ndarray, shape ``(H, W, 24)``
        ``profile[row, col, h]`` is the mean of ``flow`` on selected steps
        whose hour-of-day is ``h``. Cells that are zero all year are kept.
    hour_counts : int64 ndarray, shape ``(24,)``
        Number of selected timesteps for each hour-of-day.
    """
    x = np.asarray(flow, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"expected (T, H, W), got {x.shape}")
    t, h, w = x.shape
    if t % 24 != 0:
        raise ValueError(f"T={t} must be a multiple of 24")
    mask = np.asarray(select, dtype=bool)
    if mask.shape != (t,):
        raise ValueError(f"select must have shape {(t,)}, got {mask.shape}")
    flat = weekday_24h_profile(x, weekday_mask=mask)
    hours = np.arange(t) % 24
    counts = np.bincount(hours[mask], minlength=24).astype(np.int64)
    return flat.reshape(h, w, 24), counts


def few_shot_label_bounds(
    num_timesteps: int = EXPECTED_SERIES_HOURS,
    few_shot_hours: int = FEW_SHOT_LABEL_HOURS,
    val_hours: int = VAL_OR_TEST_HOURS,
    test_hours: int = VAL_OR_TEST_HOURS,
    lag_hours: int = LABEL_LAG_HOURS,
) -> Tuple[int, int]:
    """Raw ``[start, end)`` hours of the last few-shot training labels.

    ``split_x_y`` emits a label at index ``i`` for ``i`` from ``lag_hours`` up to,
    but not including, ``num_timesteps - val_hours - test_hours``. The target
    protocol keeps the last ``few_shot_hours`` of those labels. The returned
    slice starts at midnight for the default 8784-hour series.
    """
    train_end = num_timesteps - val_hours - test_hours
    if train_end - lag_hours < few_shot_hours:
        raise ValueError(
            f"train labels {train_end - lag_hours} < few-shot {few_shot_hours}"
        )
    start = train_end - few_shot_hours
    if start % 24 != 0 or few_shot_hours % 24 != 0:
        raise ValueError(
            f"few-shot slice [{start}, {train_end}) does not start at midnight"
        )
    return start, train_end


def _centered_norm(profiles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return centered ``(N, 24)`` profiles and their L2 norms."""
    x = np.asarray(profiles, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 24:
        raise ValueError(f"expected (N, 24), got {x.shape}")
    centered = x - x.mean(axis=1, keepdims=True)
    norm = np.sqrt((centered ** 2).sum(axis=1))
    return centered, norm


def paired_profile_pearson(
    left: np.ndarray,
    right: np.ndarray,
    std_eps: float = PROFILE_STD_EPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pearson for each pair of rows. Flat rows are undefined, not zero.

    Returns
    -------
    rho : float64 ndarray, shape ``(N,)``
        ``nan`` where either row is constant within ``std_eps``.
    defined : bool ndarray, shape ``(N,)``
    """
    left_c, left_n = _centered_norm(left)
    right_c, right_n = _centered_norm(right)
    if left_c.shape[0] != right_c.shape[0]:
        raise ValueError(f"row count {left_c.shape[0]} != {right_c.shape[0]}")
    defined = (left_n >= std_eps) & (right_n >= std_eps)
    rho = np.full(left_c.shape[0], np.nan, dtype=np.float64)
    if np.any(defined):
        rho[defined] = (left_c[defined] * right_c[defined]).sum(axis=1) / (
            left_n[defined] * right_n[defined]
        )
    return rho, defined


def max_source_pearson(
    source: np.ndarray,
    target: np.ndarray,
    std_eps: float = PROFILE_STD_EPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Max Pearson from each target row to any non-constant source row.

    Flat target rows, and targets with no usable source row, are undefined.
    """
    source_c, source_n = _centered_norm(source)
    target_c, target_n = _centered_norm(target)
    defined = target_n >= std_eps
    usable_source = source_n >= std_eps
    rho = np.full(target_c.shape[0], np.nan, dtype=np.float64)
    if not np.any(defined) or not np.any(usable_source):
        return rho, np.zeros(target_c.shape[0], dtype=bool)
    corr = (target_c[defined] / target_n[defined, None]) @ (
        source_c[usable_source] / source_n[usable_source, None]
    ).T
    rho[defined] = corr.max(axis=1)
    return rho, defined


def assign_similarity_bin(
    rho: np.ndarray,
    low: float = BIN_LOW,
    high: float = BIN_HIGH,
) -> np.ndarray:
    """Map correlations to bins. Undefined values stay ``-1``.

    ``0`` is ``rho <= low``, ``1`` is ``low < rho <= high``, ``2`` is ``rho > high``.
    These cuts describe the distribution. They are not a fitted decision rule.
    """
    values = np.asarray(rho, dtype=np.float64)
    bins = np.full(values.shape, -1, dtype=np.int64)
    finite = np.isfinite(values)
    bins[finite & (values <= low)] = 0
    bins[finite & (values > low) & (values <= high)] = 1
    bins[finite & (values > high)] = 2
    return bins


def bin_agreement(
    protocol_bins: np.ndarray,
    control_bins: np.ndarray,
    gate: float = BIN_AGREEMENT_GATE,
) -> dict:
    """Share of cells whose bin matches when both profiles are defined."""
    protocol = np.asarray(protocol_bins, dtype=np.int64).reshape(-1)
    control = np.asarray(control_bins, dtype=np.int64).reshape(-1)
    if protocol.shape != control.shape:
        raise ValueError(f"bin length {protocol.shape} != {control.shape}")
    both = (protocol >= 0) & (control >= 0)
    n_both = int(both.sum())
    n_same = int((both & (protocol == control)).sum())
    agreement = None if n_both == 0 else n_same / n_both
    return {
        "n_both_defined": n_both,
        "n_same": n_same,
        "agreement": agreement,
        "passes_gate": agreement is not None and agreement >= gate,
        "n_only_protocol": int(((protocol >= 0) & (control < 0)).sum()),
        "n_only_control": int(((protocol < 0) & (control >= 0)).sum()),
    }


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
