"""Few-shot and train/val/test index helpers (stubs).

Align with STBaT-style target protocol when implementing for real:
  test ≈ last 2 months; val ≈ preceding 2 months; train = few_shot_days before val.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SplitIndices:
    train: slice
    val: slice
    test: Optional[slice] = None


def few_shot_split_indices(
    num_timesteps: int,
    few_shot_days: int = 3,
    hours_per_day: int = 24,
    target_val_months: int = 2,
    target_test_months: int = 2,
    hours_per_month: int = 30 * 24,
) -> SplitIndices:
    """Return index slices for a target-city few-shot protocol.

    This is a **schema helper** only. Month lengths and exact calendar cuts
    must be verified against upstream CrossTReS/STBaT preprocessing before
    claiming paper-faithful splits.

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
