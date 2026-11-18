"""Timestamp-based validation blocks and expanding-prefix rolling rules."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.val_select.protocol import (
    EXPECTED_VAL_HOURS,
    EXPECTED_VAL_START,
    N_VAL_BLOCKS,
)


class IncompleteValError(ValueError):
    """Raised when the cropped val series is not a complete contiguous block."""


def validate_time_index(time_index: np.ndarray, expected_start: int | None = EXPECTED_VAL_START) -> np.ndarray:
    idx = np.asarray(time_index, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        raise IncompleteValError("empty time index")
    if np.any(np.diff(idx) <= 0):
        raise IncompleteValError("time index is not strictly increasing")
    return idx


def require_complete_val(
    time_index: np.ndarray,
    expected_hours: int = EXPECTED_VAL_HOURS,
    expected_start: int = EXPECTED_VAL_START,
) -> np.ndarray:
    """Refuse to silently split an incomplete or gapped 60-day val window."""
    idx = validate_time_index(time_index, expected_start)
    if idx.size != expected_hours:
        raise IncompleteValError(
            f"val has {idx.size} effective hours, expected {expected_hours}; refusing to pad or drop"
        )
    if int(idx[0]) != expected_start:
        raise IncompleteValError(f"val starts at {int(idx[0])}, expected {expected_start}")
    if int(idx[-1]) != expected_start + expected_hours - 1:
        raise IncompleteValError(
            f"val ends at {int(idx[-1])}, expected {expected_start + expected_hours - 1}"
        )
    if not np.array_equal(idx, np.arange(expected_start, expected_start + expected_hours, dtype=np.int64)):
        raise IncompleteValError("val time index has gaps; refusing to invent hours")
    return idx


def four_contiguous_blocks(time_index: np.ndarray) -> list[dict[str, Any]]:
    """Cut four contiguous blocks from cropped label timestamps.

    Boundaries are defined by timestamps, not by forcing 360-row slices.
    If the official series is the complete 1440-hour window, each block
    naturally contains 360 hours; that count is reported, not padded.
    """
    idx = require_complete_val(time_index)
    start = int(idx[0])
    end = int(idx[-1]) + 1
    span = end - start
    if span % N_VAL_BLOCKS != 0:
        raise IncompleteValError(f"timestamp span {span} is not divisible into {N_VAL_BLOCKS} equal blocks")
    width = span // N_VAL_BLOCKS
    blocks = []
    for k in range(N_VAL_BLOCKS):
        lo = start + k * width
        hi = start + (k + 1) * width
        mask = (idx >= lo) & (idx < hi)
        positions = np.flatnonzero(mask)
        if positions.size == 0:
            raise IncompleteValError(f"block {k} [{lo}, {hi}) has no effective hours")
        blocks.append(
            {
                "block": k,
                "t_start": lo,
                "t_end": hi,
                "n_effective": int(positions.size),
                "positions": positions,
            }
        )
    return blocks


def rolling_rounds(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expanding prefix for selection, next block for scoring.

    Round 0: prefix=block0, score=block1
    Round 1: prefix=block0+1, score=block2
    Round 2: prefix=block0+1+2, score=block3
    """
    if len(blocks) != N_VAL_BLOCKS:
        raise IncompleteValError(f"need {N_VAL_BLOCKS} blocks, got {len(blocks)}")
    rounds = []
    for r in range(N_VAL_BLOCKS - 1):
        prefix = blocks[: r + 1]
        score = blocks[r + 1]
        prefix_pos = np.concatenate([b["positions"] for b in prefix])
        rounds.append(
            {
                "round": r,
                "prefix_blocks": [b["block"] for b in prefix],
                "prefix_n": int(sum(b["n_effective"] for b in prefix)),
                "prefix_positions": prefix_pos,
                "score_block": score["block"],
                "score_n": int(score["n_effective"]),
                "score_positions": score["positions"],
            }
        )
    return rounds


def weighted_block_mae(items: list[tuple[float, int]]) -> float:
    """Weight by effective observation counts (not an unweighted mean of blocks)."""
    num = 0.0
    den = 0
    for mae, n in items:
        if n <= 0 or not np.isfinite(mae):
            raise ValueError("cannot weight a block with no effective hours or a non-finite MAE")
        num += float(mae) * int(n)
        den += int(n)
    if den <= 0:
        raise ValueError("no effective observations to weight")
    return num / float(den)
