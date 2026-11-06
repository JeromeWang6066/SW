"""Calendar and per-cell 24h profile tests. No city data and no training."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.pearson_weekday_profile import (
    CALENDAR_START,
    EXPECTED_SERIES_HOURS,
    TRAIN_PREFIX_HOURS,
    grid_hour_profile,
    hourly_calendar,
)


def test_calendar_start_is_friday_and_full_year_length() -> None:
    assert CALENDAR_START == date(2016, 1, 1)
    assert CALENDAR_START.weekday() == 4
    assert EXPECTED_SERIES_HOURS == 366 * 24
    assert TRAIN_PREFIX_HOURS == 8784 - 2 * 1440
    assert TRAIN_PREFIX_HOURS % 24 == 0

    hour, is_weekday = hourly_calendar(EXPECTED_SERIES_HOURS)
    assert hour.shape == (EXPECTED_SERIES_HOURS,)
    assert is_weekday.shape == (EXPECTED_SERIES_HOURS,)
    assert hour[0] == 0 and hour[23] == 23 and hour[24] == 0
    assert bool(is_weekday[0])
    assert not bool(is_weekday[24])
    last_day = date(2016, 12, 31)
    assert last_day.weekday() == 5
    assert not bool(is_weekday[-1])


def test_calendar_rejects_partial_day() -> None:
    with pytest.raises(ValueError):
        hourly_calendar(24 + 1)


def test_weekday_profile_keeps_zero_cells_and_drops_weekend_spike() -> None:
    hours = 7 * 24
    flow = np.zeros((hours, 2, 2), dtype=np.float64)
    _, is_weekday = hourly_calendar(hours)
    hour = np.arange(hours) % 24
    flow[(is_weekday) & (hour == 8), 0, 0] = 5.0
    flow[(~is_weekday) & (hour == 8), 0, 0] = 9.0

    weekday, weekday_counts = grid_hour_profile(flow, is_weekday)
    weekend, weekend_counts = grid_hour_profile(flow, ~is_weekday)

    assert weekday.shape == (2, 2, 24)
    assert weekend.shape == (2, 2, 24)
    assert weekday[0, 0, 8] == 5.0
    assert weekend[0, 0, 8] == 9.0
    assert np.all(weekday[0, 0, np.arange(24) != 8] == 0.0)
    assert np.all(weekday[0, 1] == 0.0)
    assert np.all(weekend[1, 1] == 0.0)
    assert np.all(weekday_counts == 5)
    assert np.all(weekend_counts == 2)
