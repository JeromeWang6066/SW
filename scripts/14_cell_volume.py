#!/usr/bin/env python3
"""Whether a 1 km cell has too few taxi pickups for a 3-day contour.

The published bin check compares each Washington cell's 3-weekday profile
with its profile on the 176 weekdays of the train prefix. This script counts
the pickups behind those profiles, then recomputes the same bin agreement
after pooling space, dropping hours, or setting sparse cells aside.

No training. Taxi pickup only. Weekdays are Monday–Friday, holidays kept,
matching the three-weekday bin check.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import load_city_array, valid_region_mask
from src.utils.pearson_weekday_profile import (
    CALENDAR_START,
    EXPECTED_SERIES_HOURS,
    PROFILE_STD_EPS,
    TRAIN_PREFIX_HOURS,
    assign_similarity_bin,
    bin_agreement,
    few_shot_label_bounds,
    hourly_calendar,
    max_source_match,
    paired_profile_pearson,
)

DAY_LENGTHS = (1, 2, 3, 5, 7, 10, 14, 21, 30)
PEAK_HOURS = (7, 8, 9, 10, 16, 17, 18, 19)
DAY_HOURS = tuple(range(6, 24))
PUBLISHED_SELF = {"ny": ("0.5646", 347), "chi": ("0.6810", 160), "dc": ("0.6054", 189)}
PUBLISHED_BINS = {"ny": (88, 189), "chi": (95, 189)}
VALID_EXPECTED = {"ny": 458, "chi": 270, "dc": 207}
CITY_NAME = {"ny": "纽约", "chi": "芝加哥", "dc": "华盛顿"}


def centered_norm(profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(profiles, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (N, D), got {x.shape}")
    centered = x - x.mean(axis=1, keepdims=True)
    norm = np.sqrt((centered ** 2).sum(axis=1))
    return centered, norm


def row_pearson(
    left: np.ndarray,
    right: np.ndarray,
    std_eps: float = PROFILE_STD_EPS,
) -> tuple[np.ndarray, np.ndarray]:
    left_c, left_n = centered_norm(left)
    right_c, right_n = centered_norm(right)
    if left_c.shape != right_c.shape:
        raise ValueError(f"shape {left_c.shape} != {right_c.shape}")
    defined = (left_n >= std_eps) & (right_n >= std_eps)
    rho = np.full(left_c.shape[0], np.nan, dtype=np.float64)
    if np.any(defined):
        rho[defined] = (left_c[defined] * right_c[defined]).sum(axis=1) / (
            left_n[defined] * right_n[defined]
        )
    return rho, defined


def max_source_match_d(
    source: np.ndarray,
    target: np.ndarray,
    std_eps: float = PROFILE_STD_EPS,
    tie_atol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same rule as ``max_source_match``, for a profile width other than 24."""
    source_c, source_n = centered_norm(source)
    target_c, target_n = centered_norm(target)
    n_target = target_c.shape[0]
    n_source = source_c.shape[0]
    defined = target_n >= std_eps
    usable_source = source_n >= std_eps
    rho = np.full(n_target, np.nan, dtype=np.float64)
    argmax = np.full(n_target, -1, dtype=np.int64)
    tied = np.zeros((n_target, n_source), dtype=bool)
    if not np.any(defined) or not np.any(usable_source):
        return rho, np.zeros(n_target, dtype=bool), argmax, tied
    usable_index = np.flatnonzero(usable_source)
    corr = (target_c[defined] / target_n[defined, None]) @ (
        source_c[usable_source] / source_n[usable_source, None]
    ).T
    best = corr.max(axis=1)
    local = np.argmax(np.abs(corr - best[:, None]) <= tie_atol, axis=1)
    rows = np.flatnonzero(defined)
    rho[defined] = best
    argmax[defined] = usable_index[local]
    tied[rows[:, None], usable_index[None, :]] = np.abs(corr - best[:, None]) <= tie_atol
    return rho, defined, argmax, tied


def block_sum(flow: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum ``k`` by ``k`` cells. The returned mask is true on complete blocks.

    A remainder row or column is padded with zeros so every original cell is
    kept, then marked incomplete. Callers that want a true ``k`` km block use
    the mask.
    """
    if k < 1:
        raise ValueError(f"k={k} must be positive")
    values = np.asarray(flow, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"expected (T, H, W), got {values.shape}")
    t, h, w = values.shape
    pad_h = (k - h % k) % k
    pad_w = (k - w % k) % k
    padded = np.pad(values, ((0, 0), (0, pad_h), (0, pad_w)))
    ph, pw = padded.shape[1], padded.shape[2]
    reduced = padded.reshape(t, ph // k, k, pw // k, k).sum(axis=(2, 4))
    full = np.ones(reduced.shape[1:], dtype=bool)
    if pad_h:
        full[-1, :] = False
    if pad_w:
        full[:, -1] = False
    return reduced, full


def box_sum(flow: np.ndarray, k: int) -> np.ndarray:
    """Centered ``k`` by ``k`` sum. ``k`` is odd. Outside the grid contributes 0."""
    values = np.asarray(flow, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"expected (T, H, W), got {values.shape}")
    if k == 1:
        return values.copy()
    if k % 2 != 1 or k < 1:
        raise ValueError(f"k={k} must be a positive odd integer")
    pad = k // 2
    padded = np.pad(values, ((0, 0), (pad, pad), (pad, pad)))
    cumulative = np.pad(np.cumsum(np.cumsum(padded, axis=1), axis=2), ((0, 0), (1, 0), (1, 0)))
    h, w = values.shape[1:]
    y0 = np.arange(h)
    x0 = np.arange(w)
    y1 = y0 + k
    x1 = x0 + k
    return (
        cumulative[:, y1[:, None], x1[None, :]]
        - cumulative[:, y0[:, None], x1[None, :]]
        - cumulative[:, y1[:, None], x0[None, :]]
        + cumulative[:, y0[:, None], x0[None, :]]
    )


def hour_energy_share(profile: np.ndarray) -> np.ndarray:
    """Largest hour's share of the centered sum of squares. Constant rows are nan."""
    centered, norm = centered_norm(profile)
    share = np.full(profile.shape[0], np.nan, dtype=np.float64)
    ok = norm >= PROFILE_STD_EPS
    if np.any(ok):
        share[ok] = (centered[ok] ** 2).max(axis=1) / (norm[ok] ** 2)
    return share


def knock_out_hottest(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace each row's largest hour with the mean of the other hours."""
    knocked = np.array(profile, dtype=np.float64, copy=True)
    hot = knocked.argmax(axis=1)
    idx = np.arange(knocked.shape[0])
    rest = knocked.sum(axis=1) - knocked[idx, hot]
    knocked[idx, hot] = rest / (knocked.shape[1] - 1)
    return knocked, hot


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return None
    lx = left[mask].argsort().argsort().astype(np.float64)
    ly = right[mask].argsort().argsort().astype(np.float64)
    lx -= lx.mean()
    ly -= ly.mean()
    denom = np.sqrt((lx ** 2).sum() * (ly ** 2).sum())
    if denom < 1e-12:
        return None
    return float((lx * ly).sum() / denom)


def nan_median(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def frac_gt(values: np.ndarray, threshold: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float((finite > threshold).mean())


def stratum_masks(daily: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        ("低于 1", daily < 1),
        ("1–9", (daily >= 1) & (daily < 10)),
        ("10–49", (daily >= 10) & (daily < 50)),
        ("50–199", (daily >= 50) & (daily < 200)),
        ("不低于 200", daily >= 200),
    ]


def weekday_days(flow: np.ndarray) -> np.ndarray:
    """Train-prefix weekdays as ``(176, 24, H, W)``. The last three match the protocol."""
    values = np.asarray(flow, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != EXPECTED_SERIES_HOURS:
        raise ValueError(f"expected {(EXPECTED_SERIES_HOURS, 'H', 'W')}, got {values.shape}")
    if float(np.min(values)) < 0:
        raise ValueError("negative pickup counts")
    _, is_weekday = hourly_calendar(values.shape[0])
    flags = is_weekday[:TRAIN_PREFIX_HOURS].reshape(-1, 24)
    if not np.all(flags == flags[:, :1]):
        raise ValueError("one calendar day mixes weekday and weekend hours")
    days = values[:TRAIN_PREFIX_HOURS].reshape(-1, 24, values.shape[1], values.shape[2])
    selected = days[flags[:, 0]]
    if selected.shape[0] != 176:
        raise ValueError(f"weekday count {selected.shape[0]} != 176")
    start, end = few_shot_label_bounds()
    if (start, end) != (5832, 5904):
        raise ValueError(f"protocol bounds {(start, end)} != (5832, 5904)")
    protocol = values[start:end].reshape(3, 24, values.shape[1], values.shape[2])
    if not np.array_equal(protocol, selected[-3:]):
        raise ValueError("protocol window is not the last three train weekdays")
    if not bool(is_weekday[start:end].all()):
        raise ValueError("protocol window is not all weekdays")
    return selected


def stack_valid(days: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``(n_days, 24, H, W)`` to ``(n_days, N, 24)`` on ``valid``."""
    moved = np.moveaxis(days, 1, -1)
    if valid.shape != moved.shape[1:3]:
        raise ValueError(f"valid {valid.shape} != {(moved.shape[1], moved.shape[2])}")
    return np.ascontiguousarray(moved[:, valid, :])


def _rate(flag: np.ndarray, eligible: np.ndarray) -> tuple[float | None, int]:
    den = int(np.asarray(eligible).sum())
    if den == 0:
        return None, 0
    return float(np.logical_and(flag, eligible).sum() / den), den


def _band_row(
    name: str,
    mask: np.ndarray,
    daily: np.ndarray,
    self_rho: np.ndarray | None = None,
    both: np.ndarray | None = None,
    same: np.ndarray | None = None,
    medians: dict[str, np.ndarray] | None = None,
    rates: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    chosen = np.asarray(mask, dtype=bool)
    total = float(daily.sum())
    row: dict[str, Any] = {
        "stratum": name,
        "n_cells": int(chosen.sum()),
        "cell_share": float(chosen.mean()) if chosen.size else None,
        "trip_share": float(daily[chosen].sum() / total) if total > 0 else None,
        "daily_median": nan_median(daily[chosen]),
    }
    if self_rho is not None:
        row["self_median"] = nan_median(self_rho[chosen])
        row["self_frac_gt_0.8"] = frac_gt(self_rho[chosen], 0.8)
        row["n_self_defined"] = int(np.isfinite(self_rho[chosen]).sum())
    if both is not None and same is not None:
        defined = both & chosen
        agreed = same & chosen
        row["n_both"] = int(defined.sum())
        row["n_same"] = int(agreed.sum())
        row["agreement"] = float(agreed.sum() / defined.sum()) if defined.any() else None
        denom = float(daily[defined].sum())
        row["trip_weighted_agreement"] = float(daily[agreed].sum() / denom) if denom > 0 else None
    for key, values in (medians or {}).items():
        row[key] = nan_median(values[chosen])
    for key, (flag, eligible) in (rates or {}).items():
        value, den = _rate(flag[chosen], eligible[chosen])
        row[key] = value
        row[f"{key}_n"] = den
    return row


def _bands(daily: np.ndarray, **kwargs: Any) -> list[dict[str, Any]]:
    rows = [_band_row(name, mask, daily, **kwargs) for name, mask in stratum_masks(daily)]
    rows.append(_band_row("全部", np.ones(daily.shape[0], dtype=bool), daily, **kwargs))
    covered = sum(int(row["n_cells"]) for row in rows[:-1])
    if covered != int(daily.shape[0]):
        raise ValueError(f"strata cover {covered} != {daily.shape[0]}")
    return rows


class Match:
    def __init__(
        self,
        self_rho: np.ndarray,
        both: np.ndarray,
        same: np.ndarray,
        weight: np.ndarray,
    ) -> None:
        self.self_rho = self_rho
        self.both = both
        self.same = same
        self.weight = weight

    def public(self) -> dict[str, Any]:
        defined = self.both
        agreed = self.same
        denom = float(self.weight[defined].sum())
        return {
            "n_cells": int(self.weight.shape[0]),
            "n_both": int(defined.sum()),
            "n_same": int(agreed.sum()),
            "agreement": float(agreed.sum() / defined.sum()) if defined.any() else None,
            "trip_weighted_agreement": float(self.weight[agreed].sum() / denom) if denom > 0 else None,
            "self_median": nan_median(self.self_rho),
            "self_frac_gt_0.8": frac_gt(self.self_rho, 0.8),
            "n_self_defined": int(np.isfinite(self.self_rho).sum()),
        }


def match_profiles(source: np.ndarray, target_short: np.ndarray, target_long: np.ndarray, weight: np.ndarray) -> Match:
    if source.shape[1] != target_short.shape[1] or target_short.shape != target_long.shape:
        raise ValueError("profile widths do not match")
    if target_short.shape[0] != weight.shape[0]:
        raise ValueError("weight length does not match the target")
    if source.shape[1] == 24:
        short_rho, _, _, _ = max_source_match(source, target_short)
        long_rho, _, _, _ = max_source_match(source, target_long)
        self_rho, _ = paired_profile_pearson(target_short, target_long)
    else:
        short_rho, _, _, _ = max_source_match_d(source, target_short)
        long_rho, _, _, _ = max_source_match_d(source, target_long)
        self_rho, _ = row_pearson(target_short, target_long)
    short_bins = assign_similarity_bin(short_rho)
    long_bins = assign_similarity_bin(long_rho)
    both = (short_bins >= 0) & (long_bins >= 0)
    same = both & (short_bins == long_bins)
    check = bin_agreement(short_bins, long_bins)
    if check["n_both_defined"] != int(both.sum()) or check["n_same"] != int(same.sum()):
        raise ValueError("bin_agreement does not match the masks")
    return Match(self_rho, both, same, weight)


def successive_rhos(flat: np.ndarray, long_profile: np.ndarray, n_days: int) -> np.ndarray:
    n_windows = flat.shape[0] - n_days + 1
    out = np.empty((n_windows, flat.shape[1]), dtype=np.float64)
    for i in range(n_windows):
        out[i], _ = row_pearson(flat[i : i + n_days].mean(axis=0), long_profile)
    return out


def per_cell_median(rhos: np.ndarray) -> np.ndarray:
    out = np.full(rhos.shape[1], np.nan, dtype=np.float64)
    for i in range(rhos.shape[1]):
        finite = rhos[:, i]
        finite = finite[np.isfinite(finite)]
        if finite.size:
            out[i] = float(np.median(finite))
    return out


def volume_summary(daily: np.ndarray, three_total: np.ndarray) -> dict[str, Any]:
    quantiles = {
        str(q): float(np.quantile(daily, q / 100))
        for q in (0, 5, 10, 25, 50, 75, 90, 95, 100)
    }
    total = float(daily.sum())
    below = []
    for threshold in (1, 5, 10, 20, 50, 100, 200, 500):
        mask = daily < threshold
        below.append(
            {
                "threshold": threshold,
                "n_cells": int(mask.sum()),
                "cell_share": float(mask.mean()),
                "trip_share": float(daily[mask].sum() / total) if total else None,
            }
        )
    order = np.sort(daily)
    top_n = max(1, int(np.ceil(0.1 * daily.size)))
    return {
        "n": int(daily.size),
        "mean": float(daily.mean()),
        "quantiles": quantiles,
        "below": below,
        "n_zero_weekday": int(np.sum(daily == 0)),
        "top_10pct_cells": top_n,
        "top_10pct_trip_share": float(order[-top_n:].sum() / total) if total else None,
        "three_day_total_median": float(np.median(three_total)),
        "three_day_total_p25": float(np.quantile(three_total, 0.25)),
        "three_day_total_p75": float(np.quantile(three_total, 0.75)),
    }


def hour_rows(three_sum: np.ndarray, long_profile: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for hour in range(24):
        rows.append(
            {
                "hour": hour,
                "long_mean_over_cells": float(long_profile[:, hour].mean()),
                "long_median_over_cells": float(np.median(long_profile[:, hour])),
                "frac_cells_3day_zero": float((three_sum[:, hour] == 0).mean()),
                "frac_cells_3day_le_1": float((three_sum[:, hour] <= 1).mean()),
            }
        )
    return rows


def _prepare(flow: np.ndarray, valid: np.ndarray | None = None) -> dict[str, Any]:
    if valid is None:
        valid = valid_region_mask(flow)
    days = weekday_days(flow)
    flat = stack_valid(days, valid)
    return {
        "flat": flat,
        "daily": flat.sum(axis=2).mean(axis=0),
        "long": flat.mean(axis=0),
        "short": flat[-3:].mean(axis=0),
        "three_sum": flat[-3:].sum(axis=0),
        "valid": valid,
    }


def _dominance(flat: np.ndarray, long_profile: np.ndarray, three_sum: np.ndarray) -> dict[str, np.ndarray]:
    total = three_sum.sum(axis=1)
    hour_share = np.full(total.shape, np.nan, dtype=np.float64)
    day_share = np.full(total.shape, np.nan, dtype=np.float64)
    positive = total > 0
    hour_share[positive] = three_sum[positive].max(axis=1) / total[positive]
    day_total = flat[-3:].sum(axis=2)
    day_share[positive] = day_total[:, positive].max(axis=0) / total[positive]
    short_energy = hour_energy_share(three_sum)
    long_energy = hour_energy_share(long_profile)
    knocked, _hot = knock_out_hottest(three_sum)
    knocked_rho, _ = row_pearson(knocked, long_profile)
    base_rho, _ = row_pearson(three_sum, long_profile)
    loo = np.vstack(
        [row_pearson(flat[-3:][np.arange(3) != leave].mean(axis=0), long_profile)[0] for leave in range(3)]
    )
    spread = np.full(total.shape, np.nan, dtype=np.float64)
    for i in range(total.shape[0]):
        finite = loo[:, i]
        finite = finite[np.isfinite(finite)]
        if finite.size:
            spread[i] = float(finite.max() - finite.min())
    return {
        "max_hour_trip_share": hour_share,
        "max_day_trip_share": day_share,
        "short_energy_share": short_energy,
        "long_energy_share": long_energy,
        "knockout_delta": knocked_rho - base_rho,
        "leave_one_day_spread": spread,
        "self_rho": base_rho,
    }


def _peak_arrays(short: np.ndarray, long_profile: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    am_s = short[:, 7:11].mean(axis=1)
    pm_s = short[:, 16:20].mean(axis=1)
    am_l = long_profile[:, 7:11].mean(axis=1)
    pm_l = long_profile[:, 16:20].mean(axis=1)

    def rel(est: np.ndarray, ref: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
        error = np.full(est.shape, np.nan, dtype=np.float64)
        ok = ref >= floor
        error[ok] = np.abs(est[ok] - ref[ok]) / ref[ok]
        return error, ok

    am_rel, am_ok = rel(am_s, am_l, 0.5)
    pm_rel, pm_ok = rel(pm_s, pm_l, 0.5)
    short_sign = np.sign(pm_s - am_s)
    long_sign = np.sign(pm_l - am_l)
    short_sign[np.abs(pm_s - am_s) <= 1e-8] = 0
    long_sign[np.abs(pm_l - am_l) <= 1e-8] = 0
    sign_ok = (am_l + pm_l) >= 1
    am_within = np.abs((7 + short[:, 7:11].argmax(axis=1)) - (7 + long_profile[:, 7:11].argmax(axis=1))) <= 1
    pm_within = np.abs((16 + short[:, 16:20].argmax(axis=1)) - (16 + long_profile[:, 16:20].argmax(axis=1))) <= 1
    medians = {"am_rel_median": am_rel, "pm_rel_median": pm_rel}
    rates = {
        "peak_sign_agree": (short_sign == long_sign, sign_ok),
        "am_hour_within_1": (am_within, am_ok),
        "pm_hour_within_1": (pm_within, pm_ok),
    }
    return medians, rates


def _window_summary(rhos: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    n_cells = int(mask.sum())
    if n_cells == 0:
        return {
            "n_windows": int(rhos.shape[0]),
            "n_cells": 0,
            "median_of_cell_medians": None,
            "frac_cells_median_gt_0.8": None,
            "protocol_median": None,
            "protocol_frac_gt_0.8": None,
        }
    cell_med = per_cell_median(rhos[:, mask])
    finite = cell_med[np.isfinite(cell_med)]
    protocol = rhos[-1, mask]
    protocol_finite = protocol[np.isfinite(protocol)]
    return {
        "n_windows": int(rhos.shape[0]),
        "n_cells": n_cells,
        "median_of_cell_medians": float(np.median(finite)) if finite.size else None,
        "frac_cells_median_gt_0.8": float(np.mean(np.isfinite(cell_med) & (cell_med > 0.8))),
        "protocol_median": float(np.median(protocol_finite)) if protocol_finite.size else None,
        "protocol_frac_gt_0.8": float(np.mean(protocol_finite > 0.8)) if protocol_finite.size else None,
    }


def _protocol_rank(rhos: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    scores = []
    for i in range(rhos.shape[0]):
        finite = rhos[i, mask]
        finite = finite[np.isfinite(finite)]
        scores.append(None if finite.size == 0 else float(np.median(finite)))
    protocol = scores[-1]
    comparable = [score for score in scores if score is not None]
    n_le = sum(score <= protocol + 1e-12 for score in comparable) if protocol is not None else 0
    return {
        "n_windows": len(comparable),
        "protocol_median": protocol,
        "window_score_median": float(np.median(comparable)) if comparable else None,
        "n_windows_le_protocol": int(n_le),
        "share_windows_le_protocol": float(n_le / len(comparable)) if comparable and protocol is not None else None,
    }


def _check_library_match(source: np.ndarray, target: np.ndarray) -> None:
    lib = max_source_match(source, target)
    local = max_source_match_d(source, target)
    if not np.array_equal(lib[1], local[1]) or not np.array_equal(lib[2], local[2]):
        raise ValueError("generic source match disagrees with the 24-hour helper")
    if not np.allclose(lib[0], local[0], equal_nan=True, atol=1e-10, rtol=0):
        raise ValueError("generic source correlation disagrees with the 24-hour helper")


def _slice_hours(profile: np.ndarray, hours: tuple[int, ...]) -> np.ndarray:
    return profile[:, list(hours)]


def build_report(processed_dir: Path) -> dict[str, Any]:
    protocol_start = CALENDAR_START + timedelta(days=5832 // 24)
    raw: dict[str, dict[str, Any]] = {}
    cities: dict[str, Any] = {}
    for code in ("ny", "chi", "dc"):
        flow = load_city_array(code, processed_dir)
        if not bool(np.max(np.abs(flow - np.round(flow))) < 1e-6):
            raise ValueError(f"{code} pickup counts are not integers")
        valid = valid_region_mask(flow)
        if int(valid.sum()) != VALID_EXPECTED[code]:
            raise ValueError(f"{code} valid {int(valid.sum())} != {VALID_EXPECTED[code]}")
        prepared = _prepare(flow, valid)
        self_rho, self_defined = paired_profile_pearson(prepared["short"], prepared["long"])
        direct, direct_defined = row_pearson(prepared["short"], prepared["long"])
        if not np.array_equal(self_defined, direct_defined) or not np.allclose(self_rho, direct, equal_nan=True):
            raise ValueError(f"{code} row pearson disagrees with paired_profile_pearson")
        published, n_defined = PUBLISHED_SELF[code]
        if int(self_defined.sum()) != n_defined or f"{nan_median(self_rho):.4f}" != published:
            raise ValueError(
                f"{code} self profile {nan_median(self_rho)} on {int(self_defined.sum())} "
                f"!= {published} on {n_defined}"
            )
        dom = _dominance(prepared["flat"], prepared["long"], prepared["three_sum"])
        if not np.allclose(dom["self_rho"], self_rho, equal_nan=True):
            raise ValueError(f"{code} dominance correlation drifted")
        zero_hours = (prepared["three_sum"] == 0).sum(axis=1).astype(np.float64)
        three_total = prepared["three_sum"].sum(axis=1)
        medians = {
            "three_day_total_median": three_total,
            "zero_hours_median": zero_hours,
            "max_hour_trip_share_median": dom["max_hour_trip_share"],
            "max_day_trip_share_median": dom["max_day_trip_share"],
            "short_energy_median": dom["short_energy_share"],
            "long_energy_median": dom["long_energy_share"],
            "knockout_delta_median": dom["knockout_delta"],
            "leave_one_day_spread_median": dom["leave_one_day_spread"],
        }
        rates = {
            "frac_hour_share_gt_0.25": (dom["max_hour_trip_share"] > 0.25, np.isfinite(dom["max_hour_trip_share"])),
            "frac_knockout_helps_gt_0.05": (dom["knockout_delta"] > 0.05, np.isfinite(dom["knockout_delta"])),
        }
        peak_medians, peak_rates = _peak_arrays(prepared["short"], prepared["long"])
        window_rho = successive_rhos(prepared["flat"], prepared["long"], 3)
        if not np.allclose(window_rho[-1], self_rho, equal_nan=True):
            raise ValueError(f"{code} last 3-day window is not the protocol window")
        days: dict[str, list[dict[str, Any]]] = {}
        ranks: dict[str, Any] = {}
        full = np.ones(prepared["daily"].shape[0], dtype=bool)
        for n_days in DAY_LENGTHS:
            rhos = window_rho if n_days == 3 else successive_rhos(prepared["flat"], prepared["long"], n_days)
            for name, mask in stratum_masks(prepared["daily"]) + [("全部", full)]:
                item = _window_summary(rhos, mask)
                item["n"] = n_days
                item["stratum"] = name
                days.setdefault(name, []).append(item)
                if n_days == 3:
                    ranks[name] = _protocol_rank(rhos, mask)
        cities[code] = {
            "n_valid": int(valid.sum()),
            "n_weekdays": 176,
            "integer_counts": True,
            "volume": volume_summary(prepared["daily"], three_total),
            "spearman_daily_self": spearman(prepared["daily"], self_rho),
            "protocol_bands": _bands(
                prepared["daily"],
                self_rho=self_rho,
                medians=medians,
                rates=rates,
            ),
            "hours": hour_rows(prepared["three_sum"], prepared["long"]),
            "days_needed": days,
            "protocol_rank": ranks,
            "peak_bands": _bands(prepared["daily"], medians=peak_medians, rates=peak_rates),
        }
        raw[code] = prepared
        raw[code]["flow"] = flow

    _check_library_match(raw["ny"]["long"], raw["dc"]["short"])
    directions: dict[str, Any] = {}
    baseline_both: dict[str, np.ndarray] = {}
    for source in ("ny", "chi"):
        matched = match_profiles(raw[source]["long"], raw["dc"]["short"], raw["dc"]["long"], raw["dc"]["daily"])
        expected_same, expected_both = PUBLISHED_BINS[source]
        if matched.public()["n_same"] != expected_same or matched.public()["n_both"] != expected_both:
            raise ValueError(
                f"{source}->dc bins {matched.public()['n_same']}/{matched.public()['n_both']} "
                f"!= {expected_same}/{expected_both}"
            )
        baseline_both[source] = matched.both
        directions[source] = {
            "overall": matched.public(),
            "bands": _bands(
                raw["dc"]["daily"],
                self_rho=matched.self_rho,
                both=matched.both,
                same=matched.same,
            ),
        }

    mitigations = _mitigations(raw, baseline_both)
    return {
        "channel": "taxi_pickup",
        "trained": False,
        "grid_km": 1,
        "protocol_dates": [protocol_start.isoformat(), "2016-09-02"],
        "n_weekdays_train": 176,
        "published_self_median": PUBLISHED_SELF,
        "published_bins": PUBLISHED_BINS,
        "cities": cities,
        "directions": directions,
        "mitigations": mitigations,
        "reading": _reading(cities, directions, mitigations),
    }


def _original_agreement(matched: Match, baseline: np.ndarray) -> dict[str, Any]:
    defined = matched.both & baseline
    agreed = matched.same & baseline
    return {
        "n_both": int(defined.sum()),
        "n_same": int(agreed.sum()),
        "agreement": float(agreed.sum() / defined.sum()) if defined.any() else None,
    }


def _mitigation_row(
    name: str,
    source_rows: dict[str, np.ndarray],
    target_short: np.ndarray,
    target_long: np.ndarray,
    weight: np.ndarray,
    contour_daily: np.ndarray,
    baseline_both: dict[str, np.ndarray],
    band_daily: np.ndarray,
    keeps_cells: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "keeps_cells": keeps_cells,
        "contour_daily_median": nan_median(contour_daily),
        "frac_contour_daily_below_10": float(np.mean(contour_daily < 10)),
        "directions": {},
    }
    for source, profiles in source_rows.items():
        matched = match_profiles(profiles, target_short, target_long, weight)
        item = matched.public()
        item["bands"] = _bands(band_daily, self_rho=matched.self_rho, both=matched.both, same=matched.same)
        if keeps_cells:
            item["on_original_defined"] = _original_agreement(matched, baseline_both[source])
        row["directions"][source] = item
    return row


def _mitigations(raw: dict[str, dict[str, Any]], baseline_both: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_sources = {code: raw[code]["long"] for code in ("ny", "chi")}

    def add(name: str, sources: dict[str, np.ndarray], short: np.ndarray, long_profile: np.ndarray, weight: np.ndarray, contour_daily: np.ndarray, band_daily: np.ndarray, keeps: bool) -> None:
        rows.append(
            _mitigation_row(name, sources, short, long_profile, weight, contour_daily, baseline_both, band_daily, keeps)
        )

    dc = raw["dc"]
    add("1km 原轮廓", base_sources, dc["short"], dc["long"], dc["daily"], dc["daily"], dc["daily"], True)

    for k, label in ((3, "3×3"), (5, "5×5")):
        dc_box = _prepare(box_sum(dc["flow"], k), dc["valid"])
        add(
            f"只平滑华盛顿轮廓，{label} 求和",
            base_sources,
            dc_box["short"],
            dc_box["long"],
            dc["daily"],
            dc_box["daily"],
            dc["daily"],
            True,
        )
        both = {}
        contour = dc_box["daily"]
        for code in ("ny", "chi"):
            both[code] = _prepare(box_sum(raw[code]["flow"], k), raw[code]["valid"])["long"]
        add(
            f"两边轮廓都做 {label} 求和",
            both,
            dc_box["short"],
            dc_box["long"],
            dc["daily"],
            contour,
            dc["daily"],
            True,
        )

    for k in (2, 3):
        prepared = {}
        for code in ("ny", "chi", "dc"):
            blocked, full = block_sum(raw[code]["flow"], k)
            valid = valid_region_mask(blocked) & full
            prepared[code] = _prepare(blocked, valid)
        add(
            f"合成完整 {k}km 块",
            {code: prepared[code]["long"] for code in ("ny", "chi")},
            prepared["dc"]["short"],
            prepared["dc"]["long"],
            prepared["dc"]["daily"],
            prepared["dc"]["daily"],
            prepared["dc"]["daily"],
            False,
        )

    peak_daily = dc["flat"][:, :, list(PEAK_HOURS)].sum(axis=2).mean(axis=0)
    add(
        "只用 7–10 点与 16–19 点",
        {code: _slice_hours(raw[code]["long"], PEAK_HOURS) for code in ("ny", "chi")},
        _slice_hours(dc["short"], PEAK_HOURS),
        _slice_hours(dc["long"], PEAK_HOURS),
        dc["daily"],
        peak_daily,
        dc["daily"],
        True,
    )
    day_daily = dc["flat"][:, :, list(DAY_HOURS)].sum(axis=2).mean(axis=0)
    add(
        "去掉 0–5 点",
        {code: _slice_hours(raw[code]["long"], DAY_HOURS) for code in ("ny", "chi")},
        _slice_hours(dc["short"], DAY_HOURS),
        _slice_hours(dc["long"], DAY_HOURS),
        dc["daily"],
        day_daily,
        dc["daily"],
        True,
    )
    add(
        "24 小时均值再取 log(1+x)",
        {code: np.log1p(raw[code]["long"]) for code in ("ny", "chi")},
        np.log1p(dc["short"]),
        np.log1p(dc["long"]),
        dc["daily"],
        dc["daily"],
        dc["daily"],
        True,
    )
    logged = {code: np.log1p(raw[code]["flat"]) for code in ("ny", "chi", "dc")}
    add(
        "先对每小时取 log(1+x) 再对天平均",
        {code: logged[code].mean(axis=0) for code in ("ny", "chi")},
        logged["dc"][-3:].mean(axis=0),
        logged["dc"].mean(axis=0),
        dc["daily"],
        dc["daily"],
        dc["daily"],
        True,
    )
    for threshold, label in ((10, "不低于 10"), (50, "不低于 50"), (200, "不低于 200")):
        mask = dc["daily"] >= threshold
        add(
            f"只给日均{label}的格子分档",
            base_sources,
            dc["short"][mask],
            dc["long"][mask],
            dc["daily"][mask],
            dc["daily"][mask],
            dc["daily"][mask],
            False,
        )
    base = rows[0]
    for source, (n_same, n_both) in PUBLISHED_BINS.items():
        item = base["directions"][source]
        original = item["on_original_defined"]
        if item["n_same"] != n_same or item["n_both"] != n_both:
            raise ValueError(f"mitigation baseline {source} {item['n_same']}/{item['n_both']}")
        if original["n_same"] != n_same or original["n_both"] != n_both:
            raise ValueError(f"original-cell baseline {source} {original['n_same']}/{original['n_both']}")
    return rows


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.1f}%"


def _ratio(item: dict[str, Any] | None) -> str:
    if not item or item.get("n_both") in (None, 0) or item.get("agreement") is None:
        return "—"
    return f"{item['n_same']}/{item['n_both']} = {_fmt(item['agreement'])}"


def _band(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for row in rows:
        if row["stratum"] == name:
            return row
    raise KeyError(name)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _reading(cities: dict[str, Any], directions: dict[str, Any], mitigations: list[dict[str, Any]]) -> list[str]:
    dc = cities["dc"]
    low = _band(dc["protocol_bands"], "1–9")
    empty = _band(dc["protocol_bands"], "低于 1")
    busy = _band(dc["protocol_bands"], "不低于 200")
    ny = _band(directions["ny"]["bands"], "全部")
    chi = _band(directions["chi"]["bands"], "全部")
    ny_low = _band(directions["ny"]["bands"], "1–9")
    ny_busy = _band(directions["ny"]["bands"], "不低于 200")
    rank = dc["protocol_rank"]["全部"]
    lines = [
        (
            f"华盛顿有效格 {dc['n_valid']} 个。工作日日均上车量的中位数是 {dc['volume']['quantiles']['50']:.2f}，"
            f"平均数是 {dc['volume']['mean']:.1f}。日均低于 1 的有 {empty['n_cells']} 个，这三天上车中位数是 "
            f"{empty['three_day_total_median']:.0f}。日均 1–9 的有 {low['n_cells']} 个，占有效格 {_fmt_pct(low['cell_share'])}，"
            f"占上车量 {_fmt_pct(low['trip_share'])}；这三天每个格子上车中位数 {low['three_day_total_median']:.0f}，"
            f"24 小时里为 0 的个数中位数是 {low['zero_hours_median']:.0f}。"
            f"日均不低于 200 的有 {busy['n_cells']} 个，占上车量 {_fmt_pct(busy['trip_share'])}，"
            f"三天上车中位数 {busy['three_day_total_median']:.0f}，为 0 的小时中位数是 {busy['zero_hours_median']:.0f}。"
        ),
        (
            f"这三天轮廓对长轮廓的自身相关，1–9 档中位数 {_fmt(low['self_median'])}，不低于 200 档 {_fmt(busy['self_median'])}。"
            f"日均上车量和自身相关的 Spearman 是 {_fmt(dc['spearman_daily_self'])}。"
            f"1–9 档里，单个小时占这三天上车量超过 25% 的格子占 {_fmt_pct(low['frac_hour_share_gt_0.25'])}；"
            f"去掉最大的那个小时后，相关上升超过 0.05 的占 {_fmt_pct(low['frac_knockout_helps_gt_0.05'])}。"
            f"不低于 200 档这两项是 {_fmt_pct(busy['frac_hour_share_gt_0.25'])} 和 {_fmt_pct(busy['frac_knockout_helps_gt_0.05'])}。"
        ),
        (
            f"纽约→华盛顿，有定义格子的档位一致比例是 {_ratio(ny)}，按上车量加权是 {_fmt(ny['trip_weighted_agreement'])}。"
            f"1–9 档是 {_ratio(ny_low)}，不低于 200 档是 {_ratio(ny_busy)}。"
            f"芝加哥→华盛顿全部有定义格子是 {_ratio(chi)}，按上车量加权 {_fmt(chi['trip_weighted_agreement'])}。"
        ),
        (
            f"连续三个工作日共 {rank['n_windows']} 个窗口。这些窗口先在格子上取相关中位数，再在窗口之间取中位数，是 "
            f"{_fmt(rank['window_score_median'])}。协议这三天是 {_fmt(rank['protocol_median'])}，"
            f"{rank['n_windows']} 个窗口里只有 {rank['n_windows_le_protocol']} 个不高于它。"
            "这三天比大多数连续三个工作日更散。流量档之间的差距比这个窗口差更大。"
        ),
    ]
    named = {item["name"]: item for item in mitigations}
    gate50 = named["只给日均不低于 50的格子分档"]
    gate10 = named["只给日均不低于 10的格子分档"]
    box = named["只平滑华盛顿轮廓，3×3 求和"]
    coarse = named["合成完整 2km 块"]
    lines.append(
        "日均不低于 50 的格子，纽约方向 "
        f"{_ratio(gate50['directions']['ny'])}，芝加哥方向 {_ratio(gate50['directions']['chi'])}。"
        f"日均不低于 10 的格子是 {_ratio(gate10['directions']['ny'])} 和 {_ratio(gate10['directions']['chi'])}。"
        f"3×3 邻域求和后是 {_ratio(box['directions']['ny'])} 和 {_ratio(box['directions']['chi'])}。"
        f"完整 2km 块是 {_ratio(coarse['directions']['ny'])} 和 {_ratio(coarse['directions']['chi'])}。"
        "0.7 仍是原来的描述门槛，不是这条脚本新定的规则。"
    )
    return lines


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# 1km 格子每天的上车量够不够撑起三天轮廓",
        "",
        "通道是出租上车。格子里的数是该小时的上车次数，而且是整数。一个格子一天的数据量是 24 小时相加。",
        "有效格仍是全年上车之和大于 0，所以一个格子可以在工作日几乎没有上车、却仍然有效。",
        "工作日是周一到周五，假日留在工作日里。长窗口是训练段前 5904 小时里的 176 个工作日。短窗口是 "
        f"{report['protocol_dates'][0]} 至 {report['protocol_dates'][1]}，下标 [5832, 5904)。",
        "自身相关是同一个格子的短轮廓和长轮廓的皮尔逊相关。档位仍是不超过 0.5、大于 0.5 且不超过 0.8、大于 0.8。",
        "按上车量加权，是用长窗口的工作日日均上车量做权重，只加在两个窗口都有定义的格子上。",
        "这一步没有训练。脚本复算了已发表的自身相关中位数和三天分档人数，对不上就会停。",
        "",
        "## 每个有效格的工作日日均上车量",
        "",
        "| 城市 | 有效格 | 中位数 | 平均数 | 25% | 75% | 日均低于 10 的格子 | 这些格子的上车量 | 最忙的一成格子的上车量 | 三天总上车中位数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for code in ("ny", "chi", "dc"):
        city = report["cities"][code]
        vol = city["volume"]
        below10 = next(item for item in vol["below"] if item["threshold"] == 10)
        lines.append(
            f"| {CITY_NAME[code]} | {vol['n']} | {_fmt(vol['quantiles']['50'], 2)} | {_fmt(vol['mean'], 1)} | "
            f"{_fmt(vol['quantiles']['25'], 2)} | {_fmt(vol['quantiles']['75'], 2)} | "
            f"{below10['n_cells']} ({_fmt_pct(below10['cell_share'])}) | {_fmt_pct(below10['trip_share'])} | "
            f"{_fmt_pct(vol['top_10pct_trip_share'])} | {_fmt(vol['three_day_total_median'], 1)} |"
        )
    lines.extend(
        [
            "",
            "分档是工作日日均上车量。三天总上车是协议这三天的计数。为 0 的小时数是这三天 24 小时里计数为 0 的个数。",
            "",
            "| 城市 | 日均档 | 格子数 | 占格子 | 占上车量 | 三天总上车中位数 | 为 0 的小时中位数 | 自身相关中位数 | 自身相关高于 0.8 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for code in ("ny", "chi", "dc"):
        for row in report["cities"][code]["protocol_bands"]:
            lines.append(
                f"| {CITY_NAME[code]} | {row['stratum']} | {row['n_cells']} | {_fmt_pct(row['cell_share'])} | "
                f"{_fmt_pct(row['trip_share'])} | {_fmt(row['three_day_total_median'], 1)} | "
                f"{_fmt(row['zero_hours_median'], 1)} | {_fmt(row['self_median'])} | {_fmt_pct(row['self_frac_gt_0.8'])} |"
            )
    lines.extend(
        [
            "",
            "## 三天里的一个小时会不会改掉轮廓",
            "",
            "单小时上车占比是这三天里最大的那个小时，占这三天总上车的比例。单日占比是三天里最多的那一天。",
            "形状占比是居中之后的平方和里，最大的那个小时占多少。24 个小时若贡献相同，每个小时约是 0.0417。",
            "去掉最大小时，是把该小时换成其余 23 小时的均值，再和长轮廓做相关。差值为正，表示这个小时在把短轮廓拉离长轮廓。",
            "去掉一天的极差，是三次「留两天」得到的相关里，最大减最小。",
            "",
            "| 城市 | 日均档 | 单小时上车占比中位数 | 超过 25% 的格子 | 单日占比中位数 | 短轮廓形状占比 | 长轮廓形状占比 | 去掉最大小时后的相关变化 | 其中上升超过 0.05 | 去掉一天的相关极差 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for code in ("dc", "ny", "chi"):
        for row in report["cities"][code]["protocol_bands"]:
            if row["stratum"] not in {"1–9", "10–49", "不低于 200", "全部"}:
                continue
            lines.append(
                f"| {CITY_NAME[code]} | {row['stratum']} | {_fmt(row['max_hour_trip_share_median'])} | "
                f"{_fmt_pct(row['frac_hour_share_gt_0.25'])} | {_fmt(row['max_day_trip_share_median'])} | "
                f"{_fmt(row['short_energy_median'])} | {_fmt(row['long_energy_median'])} | "
                f"{_fmt(row['knockout_delta_median'])} | {_fmt_pct(row['frac_knockout_helps_gt_0.05'])} | "
                f"{_fmt(row['leave_one_day_spread_median'])} |"
            )
    lines.extend(
        [
            "",
            "## 这三天在所有连续三个工作日里的位置",
            "",
            "每个窗口都是 176 个工作日里连续的三行，先在该档格子上取相关中位数。协议窗口是最后一行。",
            "",
            "| 城市 | 日均档 | 窗口数 | 窗口中位数的中位数 | 协议窗口 | 相关不高于协议窗口的窗口数 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for code in ("dc", "ny", "chi"):
        for name in ("1–9", "不低于 200", "全部"):
            rank = report["cities"][code]["protocol_rank"][name]
            lines.append(
                f"| {CITY_NAME[code]} | {name} | {rank['n_windows']} | {_fmt(rank['window_score_median'])} | "
                f"{_fmt(rank['protocol_median'])} | "
                f"{rank['n_windows_le_protocol']} ({_fmt_pct(rank['share_windows_le_protocol'])}) |"
            )
    lines.extend(
        [
            "",
            "## 连续多少个工作日，轮廓才稳得住",
            "",
            "每个格子先取它在所有连续窗口上的相关中位数，再在档内取中位数。高于 0.8 的比例把无定义的格子算作没有超过。",
            "",
            "| 城市 | 日均档 | 1 天 | 3 天 | 5 天 | 7 天 | 14 天 | 30 天 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for code in ("dc", "ny", "chi"):
        for name in ("1–9", "10–49", "不低于 200", "全部"):
            by_n = {item["n"]: item for item in report["cities"][code]["days_needed"][name]}
            cells = " | ".join(
                f"{_fmt(by_n[n]['median_of_cell_medians'])} / {_fmt_pct(by_n[n]['frac_cells_median_gt_0.8'])}"
                for n in (1, 3, 5, 7, 14, 30)
            )
            lines.append(f"| {CITY_NAME[code]} | {name} | {cells} |")
    lines.append("")
    lines.append("单元格是「该档每个格子先取各窗口相关的中位数，再在档内取中位数 / 这个中位数高于 0.8 的格子比例」。无定义算作没有超过 0.8。")
    lines.extend(
        [
            "",
            "## 早晚高峰单独看",
            "",
            "早高峰取 7、8、9、10 点，晚高峰取 16、17、18、19 点，用的是序列自己的小时下标。",
            "相对误差是 |短窗口均值 − 长窗口均值| / 长窗口均值，只在长窗口该段日均至少 0.5 次时计算。",
            "高峰谁更高，要求长窗口这两段合计日均至少 1 次。钟点相差不超过 1，要求长窗口该段日均至少 0.5 次。",
            "",
            "| 城市 | 日均档 | 早高峰相对误差 | 晚高峰相对误差 | 谁更高一致 | 早高峰钟点差 ≤ 1 | 晚高峰钟点差 ≤ 1 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for code in ("dc", "ny", "chi"):
        for row in report["cities"][code]["peak_bands"]:
            if row["stratum"] not in {"1–9", "10–49", "不低于 200", "全部"}:
                continue
            lines.append(
                f"| {CITY_NAME[code]} | {row['stratum']} | {_fmt(row['am_rel_median'])} | {_fmt(row['pm_rel_median'])} | "
                f"{_fmt_pct(row['peak_sign_agree'])} ({row['peak_sign_agree_n']}) | "
                f"{_fmt_pct(row['am_hour_within_1'])} ({row['am_hour_within_1_n']}) | "
                f"{_fmt_pct(row['pm_hour_within_1'])} ({row['pm_hour_within_1_n']}) |"
            )
    lines.extend(
        [
            "",
            "## 改法对照",
            "",
            "对照是原来的 1km 轮廓。平滑用的是以该格为中心的邻域求和，预测格子仍是 1km；邻域超出地图的部分按 0。",
            "合成块只保留刚好铺满的块，边缘不足一块的不进这一行。",
            "进入轮廓的日均，是真正参与这条轮廓的上车量。分档门槛用的仍是格子自己的日均，不是邻域日均。",
            "只给高流量格子分档时，分母就是这些格子里两个窗口都有定义的数量。",
            "",
            "| 做法 | 进入轮廓的日均中位数 | 其中低于 10 | 自身相关中位数 | 纽约→华盛顿 | 按上车量加权 | 芝加哥→华盛顿 | 按上车量加权 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["mitigations"]:
        ny = item["directions"]["ny"]
        chi = item["directions"]["chi"]
        lines.append(
            f"| {item['name']} | {_fmt(item['contour_daily_median'], 1)} | {_fmt_pct(item['frac_contour_daily_below_10'])} | "
            f"{_fmt(ny['self_median'])} | {_ratio(ny)} | {_fmt(ny['trip_weighted_agreement'])} | "
            f"{_ratio(chi)} | {_fmt(chi['trip_weighted_agreement'])} |"
        )
    lines.extend(
        [
            "",
            "原来就已经有定义的格子，平滑或改小时之后是否还在同一档。分母固定在原来的有定义集合里。",
            "",
            "| 做法 | 纽约，原有定义格子 | 其中 1–9 档 | 其中不低于 200 | 芝加哥，原有定义格子 | 其中 1–9 档 | 其中不低于 200 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["mitigations"]:
        if not item["keeps_cells"]:
            continue
        cells = []
        for source in ("ny", "chi"):
            direction = item["directions"][source]
            cells.append(_ratio(direction.get("on_original_defined")))
            bands = {row["stratum"]: row for row in direction["bands"]}
            cells.append(_ratio(bands["1–9"]))
            cells.append(_ratio(bands["不低于 200"]))
        lines.append(f"| {item['name']} | " + " | ".join(cells) + " |")
    lines.extend(["", "## 读法", ""])
    for paragraph in report["reading"]:
        lines.append(paragraph)
        lines.append("")
    lines.extend(
        [
            "重算：",
            "",
            "```bash",
            "python scripts/14_cell_volume.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--out-json", type=Path, default=ROOT / "docs" / "profiles" / "cell_volume.json")
    parser.add_argument("--out-md", type=Path, default=ROOT / "docs" / "profiles" / "cell_volume.md")
    args = parser.parse_args()
    report = build_report(args.processed_dir)
    payload = _jsonable(report)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    dc = payload["cities"]["dc"]["volume"]
    ny = payload["directions"]["ny"]["overall"]
    print(
        f"dc median daily {dc['quantiles']['50']:.2f}; "
        f"ny->dc {ny['n_same']}/{ny['n_both']} weighted {ny['trip_weighted_agreement']:.4f}"
    )
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
