"""Few-shot and train/val/test splits aligned with CrossTReS / STBaT.

Upstream (CrossTReS / STBaT ``utils.split_x_y``):
  val_num = test_num = 60 * 24  # ~2 months of 30-day months
  few-shot: after split, train = train[-data_amount * 24:]

See ``docs/UPSTREAM_MAP.md`` §2 and ``docs/DATA.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SplitIndices:
    train: slice
    val: slice
    test: Optional[slice] = None


def upstream_style_cut_lengths(
    num_timesteps: int,
    few_shot_days: int = 3,
    hours_per_day: int = 24,
    val_hours: int = 60 * 24,
    test_hours: int = 60 * 24,
) -> Dict[str, int]:
    """Lengths matching CrossTReS/STBaT ``split_x_y`` + ``data_amount`` trim.

    Does **not** build lag windows; only reports how many timesteps fall into
    each bucket after the upstream contiguous cut. Use this when porting the
    real loader so configs stay honest about ``60*24`` (not calendar months).
    """
    if few_shot_days < 0:
        raise ValueError("few_shot_days must be >= 0")
    need = val_hours + test_hours
    if num_timesteps < need:
        raise ValueError(
            f"T={num_timesteps} < val+test={need} (upstream uses 60*24 each)"
        )
    remainder_train = num_timesteps - val_hours - test_hours
    few = few_shot_days * hours_per_day
    if few > remainder_train:
        raise ValueError(
            f"few_shot_train={few} > remainder_train={remainder_train}"
        )
    return {
        "total": num_timesteps,
        "remainder_train": remainder_train,
        "few_shot_train": few,
        "val": val_hours,
        "test": test_hours,
    }


LAG_6H: Tuple[int, ...] = (-6, -5, -4, -3, -2, -1)


def split_x_y(
    data: np.ndarray,
    lag: Sequence[int] = LAG_6H,
    val_num: int = 60 * 24,
    test_num: int = 60 * 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Window a ``T×H×W`` grid the way CrossTReS / STBaT ``utils.split_x_y`` does.

    ``lag = [-6, ..., -1]`` and the loop starts at ``i = 6``, so each sample's
    ``x`` is six past hours and ``y`` is hour ``i``. Validation and test are the
    last ``60*24`` prediction hours each. Returns
    ``train_x, train_y, val_x, val_y, test_x, test_y`` with
    ``x`` shaped ``(N, len(lag), H, W)`` and ``y`` shaped ``(N, 1, H, W)``.
    """
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"expected T×H×W, got {data.shape}")
    num_samples = int(data.shape[0])
    need = val_num + test_num
    if num_samples < need:
        raise ValueError(f"T={num_samples} < val+test={need}")
    start = -int(min(lag))
    if start >= num_samples - need:
        raise ValueError(f"lag start {start} leaves no train hours before val/test")

    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []
    for i in range(start, num_samples):
        x_idx = [int(step + i) for step in lag]
        x_ = data[x_idx]
        y_ = data[i : i + 1]
        if i < num_samples - val_num - test_num:
            train_x.append(x_)
            train_y.append(y_)
        elif i < num_samples - test_num:
            val_x.append(x_)
            val_y.append(y_)
        else:
            test_x.append(x_)
            test_y.append(y_)
    return (
        np.stack(train_x, axis=0),
        np.stack(train_y, axis=0),
        np.stack(val_x, axis=0),
        np.stack(val_y, axis=0),
        np.stack(test_x, axis=0),
        np.stack(test_y, axis=0),
    )


def trim_few_shot(
    train_x: np.ndarray,
    train_y: np.ndarray,
    few_shot_days: int,
    hours_per_day: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep the last ``few_shot_days * 24`` train windows (STBaT ``data_amount``)."""
    if few_shot_days < 0:
        raise ValueError("few_shot_days must be >= 0")
    if few_shot_days == 0:
        return train_x, train_y
    n = few_shot_days * hours_per_day
    if n > len(train_x):
        raise ValueError(f"few_shot windows {n} > train windows {len(train_x)}")
    return train_x[-n:], train_y[-n:]


def few_shot_split_indices(
    num_timesteps: int,
    few_shot_days: int = 3,
    hours_per_day: int = 24,
    target_val_months: int = 2,
    target_test_months: int = 2,
    hours_per_month: int = 30 * 24,
) -> SplitIndices:
    """Return index slices for a target-city few-shot protocol.

    **Schema helper** using ``30*24`` month approximations. For paper-faithful
    cuts prefer ``upstream_style_cut_lengths`` (``60*24``) and port
    ``split_x_y`` windowing from upstream ``utils.py``.

    Raises if the timeline is too short for the requested protocol.
    """
    if few_shot_days < 0:
        raise ValueError("few_shot_days must be >= 0")

    test_len = target_test_months * hours_per_month
    val_len = target_val_months * hours_per_month
    train_len = few_shot_days * hours_per_day
    need = test_len + val_len + train_len
    if num_timesteps < need:
        raise ValueError(
            f"T={num_timesteps} < required {need} "
            f"(train={train_len}, val={val_len}, test={test_len}). "
            "Check data length or month approximations."
        )

    test_start = num_timesteps - test_len
    val_start = test_start - val_len
    train_start = val_start - train_len

    return SplitIndices(
        train=slice(train_start, val_start),
        val=slice(val_start, test_start),
        test=slice(test_start, num_timesteps),
    )


def source_train_val_indices(
    num_timesteps: int,
    train_ratio: float = 2 / 3,
) -> SplitIndices:
    """Simple contiguous train/val cut for a source city (STBaT ~2:1)."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    cut = int(num_timesteps * train_ratio)
    if cut <= 0 or cut >= num_timesteps:
        raise ValueError(f"bad cut={cut} for T={num_timesteps}")
    return SplitIndices(train=slice(0, cut), val=slice(cut, num_timesteps), test=None)
