"""Few-shot profile slice and Pearson tests. No city data and no training."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.pearson_weekday_profile import (
    CALENDAR_START,
    few_shot_label_bounds,
    max_source_pearson,
    paired_profile_pearson,
)


def test_few_shot_labels_are_three_midnights() -> None:
    start, end = few_shot_label_bounds(8784)
    assert (start, end) == (5832, 5904)
    assert start % 24 == 0
    first = CALENDAR_START + timedelta(days=start // 24)
    last = CALENDAR_START + timedelta(days=(end - 1) // 24)
    assert first == date(2016, 8, 31)
    assert last == date(2016, 9, 2)
    assert (last - first).days == 2


def test_paired_pearson_drops_constant_profiles() -> None:
    wave = np.arange(24, dtype=np.float64)
    left = np.vstack([wave, np.zeros(24), wave])
    right = np.vstack([wave, wave, np.ones(24)])
    rho, defined = paired_profile_pearson(left, right)
    assert defined.tolist() == [True, False, False]
    assert rho[0] == pytest.approx(1.0)
    assert np.isnan(rho[1]) and np.isnan(rho[2])


def test_max_source_pearson_ignores_flat_source_rows() -> None:
    wave = np.arange(24, dtype=np.float64)
    other = np.arange(24, dtype=np.float64)[::-1]
    source = np.vstack([np.zeros(24), wave])
    target = np.vstack([wave, np.full(24, 3.0), other])
    rho, defined = max_source_pearson(source, target)
    assert defined.tolist() == [True, False, True]
    assert rho[0] == pytest.approx(1.0)
    assert np.isnan(rho[1])
    assert rho[2] == pytest.approx(float(np.corrcoef(other, wave)[0, 1]))
