"""Weekday versus weekend-plus-holiday labels for the calendar experiment.

Hour 0 is 2016-01-01 00:00. No daylight-saving insertion or deletion.
A label hour is weekend class when its calendar day is Saturday, Sunday, or one
of the 2016 US federal holidays listed below. Labor Day is in the validation
window, so it is not a training label. It is still weekend class at evaluation.

Filtering happens after ``split_x_y``. Window ``j`` of a split has label hour
``label_hours(split)[j]``. The default is no filter: callers that omit
``calendar`` keep every window.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict

import numpy as np

from src.data.dataset import GridFlowDataset
from src.utils.pearson_weekday_profile import (
    CALENDAR_START,
    EXPECTED_SERIES_HOURS,
    LABEL_LAG_HOURS,
    TRAIN_PREFIX_HOURS,
    VAL_OR_TEST_HOURS,
)

# Weekday federal holidays in the 2016 train prefix, plus Labor Day (validation).
US_FEDERAL_HOLIDAYS_2016 = (
    date(2016, 1, 1),
    date(2016, 1, 18),
    date(2016, 2, 15),
    date(2016, 5, 30),
    date(2016, 7, 4),
    date(2016, 9, 5),
)
LABOR_DAY = date(2016, 9, 5)

# Washington labels for this round. Monday and Tuesday between them are unused.
DC_WEEKEND_LABELS = (5736, 5784)
DC_WEEKDAY_LABELS = (5832, 5904)

_TRAIN_MODES = ("all", "weekday", "weekend", "dc_wd3_we2", "dc_wd3", "dc_we2")
_EARLY_MODES = ("all", "weekday", "weekend")


def hour_dates(hours: np.ndarray) -> np.ndarray:
    """Calendar date of each hour index. ``object`` array of ``datetime.date``."""
    hours = np.asarray(hours, dtype=np.int64)
    origin = np.datetime64(CALENDAR_START.isoformat())
    days = hours // 24
    return (origin + days.astype("timedelta64[D]")).astype("O")


def is_weekend_or_holiday(hours: np.ndarray) -> np.ndarray:
    """True when the hour's date is Saturday, Sunday, or a listed 2016 holiday."""
    hours = np.asarray(hours, dtype=np.int64)
    day_index = hours // 24
    weekday = (CALENDAR_START.weekday() + day_index) % 7
    holiday_days = np.array(
        [(day - CALENDAR_START).days for day in US_FEDERAL_HOLIDAYS_2016],
        dtype=np.int64,
    )
    return (weekday >= 5) | np.isin(day_index, holiday_days)


def label_hours(split: str) -> np.ndarray:
    """Raw series index of each window's label hour, before any calendar filter."""
    if split == "train":
        return np.arange(LABEL_LAG_HOURS, TRAIN_PREFIX_HOURS, dtype=np.int64)
    if split == "val":
        start = TRAIN_PREFIX_HOURS
    elif split == "test":
        start = TRAIN_PREFIX_HOURS + VAL_OR_TEST_HOURS
    else:
        raise ValueError(f"unknown split {split!r}")
    return np.arange(start, start + VAL_OR_TEST_HOURS, dtype=np.int64)


def labor_day_hour() -> int:
    """First hour of Labor Day, 2016-09-05 00:00."""
    return (LABOR_DAY - CALENDAR_START).days * 24


def keep_mask(hours: np.ndarray, mode: str) -> np.ndarray:
    """Boolean mask over ``hours``. ``weekend`` means weekend plus holiday."""
    hours = np.asarray(hours, dtype=np.int64)
    if mode == "all":
        return np.ones(hours.shape, dtype=bool)
    weekend = is_weekend_or_holiday(hours)
    if mode == "weekday":
        return ~weekend
    if mode == "weekend":
        return weekend
    if mode == "dc_wd3":
        start, end = DC_WEEKDAY_LABELS
        return (hours >= start) & (hours < end)
    if mode == "dc_we2":
        start, end = DC_WEEKEND_LABELS
        return (hours >= start) & (hours < end)
    if mode == "dc_wd3_we2":
        return keep_mask(hours, "dc_wd3") | keep_mask(hours, "dc_we2")
    raise ValueError(f"unknown calendar mode {mode!r}")


def subset_dataset(dataset: GridFlowDataset, keep: np.ndarray) -> GridFlowDataset:
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != (len(dataset),):
        raise ValueError(f"keep length {keep.shape} != dataset {len(dataset)}")
    if not bool(keep.any()):
        raise ValueError("calendar filter removed every window")
    return GridFlowDataset(dataset.x[keep], dataset.y[keep])


def apply_calendar(prepared: Dict[str, Any], calendar: Dict[str, Any] | None) -> Dict[str, Any]:
    """Filter train windows and choose the validation hours used for early stopping.

    ``prepared['val']`` and ``prepared['test']`` stay the full splits. Early
    stopping reads ``prepared['val_early']``. An empty calendar leaves the
    datasets unchanged and does not attach label hours.
    """
    if not calendar:
        return prepared
    train_mode = str(calendar.get("train_labels", "all"))
    early_mode = str(calendar.get("early_stop", "all"))
    if train_mode not in _TRAIN_MODES:
        raise ValueError(f"calendar.train_labels must be one of {_TRAIN_MODES}")
    if early_mode not in _EARLY_MODES:
        raise ValueError(f"calendar.early_stop must be one of {_EARLY_MODES}")

    train_hours = label_hours("train")
    val_hours = label_hours("val")
    test_hours = label_hours("test")
    if len(prepared["train"]) != len(train_hours):
        raise ValueError(
            "calendar filtering expects the untrimmed train split "
            f"({len(train_hours)} windows), got {len(prepared['train'])}. "
            "Set splits.few_shot_days to 0."
        )
    if len(prepared["val"]) != len(val_hours) or len(prepared["test"]) != len(test_hours):
        raise ValueError("val/test window counts do not match the 1440-hour splits")
    if int(train_hours[-1]) >= EXPECTED_SERIES_HOURS:
        raise ValueError("train label hours run past the series")

    out = dict(prepared)
    out["train"] = subset_dataset(prepared["train"], keep_mask(train_hours, train_mode))
    out["val_early"] = subset_dataset(prepared["val"], keep_mask(val_hours, early_mode))
    out["label_hours"] = {"val": val_hours, "test": test_hours}
    out["calendar"] = {"train_labels": train_mode, "early_stop": early_mode}
    return out


def slice_keeps(hours: np.ndarray) -> Dict[str, np.ndarray]:
    """All hours, weekdays, and weekend-plus-holiday, over one val or test split."""
    weekend = is_weekend_or_holiday(hours)
    return {
        "all": np.ones(hours.shape, dtype=bool),
        "weekday": ~weekend,
        "weekend": weekend,
    }


def describe_counts() -> Dict[str, Any]:
    """Counts the count script prints. Does not load flow arrays."""
    train_hours = label_hours("train")
    train_weekend = is_weekend_or_holiday(train_hours)
    val_hours = label_hours("val")
    labor = labor_day_hour()
    weekend_start, weekend_end = DC_WEEKEND_LABELS
    weekday_start, weekday_end = DC_WEEKDAY_LABELS
    return {
        "train_weekday_hours": int((~train_weekend).sum()),
        "train_weekend_hours": int(train_weekend.sum()),
        "train_hours": int(len(train_hours)),
        "dc_weekend_hours": int(weekend_end - weekend_start),
        "dc_weekday_hours": int(weekday_end - weekday_start),
        "dc_weekend_range": [weekend_start, weekend_end],
        "dc_weekday_range": [weekday_start, weekday_end],
        "dc_weekend_start": str(CALENDAR_START + timedelta(days=weekend_start // 24)),
        "dc_weekend_end": str(CALENDAR_START + timedelta(days=(weekend_end - 1) // 24)),
        "dc_weekday_start": str(CALENDAR_START + timedelta(days=weekday_start // 24)),
        "dc_weekday_end": str(CALENDAR_START + timedelta(days=(weekday_end - 1) // 24)),
        "labor_day_hour": labor,
        "labor_day_in_val": bool(val_hours[0] <= labor < val_hours[-1] + 1),
        "labor_day_is_weekend_class": bool(is_weekend_or_holiday(np.array([labor]))[0]),
        "dc_weekend_all_weekend_class": bool(
            is_weekend_or_holiday(np.arange(weekend_start, weekend_end)).all()
        ),
        "dc_weekday_all_weekday_class": bool(
            (~is_weekend_or_holiday(np.arange(weekday_start, weekday_end))).all()
        ),
    }
