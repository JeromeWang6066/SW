"""Calendar label masks. Old windows stay intact when no calendar is set."""

from datetime import date

import numpy as np

from src.data.calendar_mask import (
    DC_WEEKDAY_LABELS,
    DC_WEEKEND_LABELS,
    apply_calendar,
    describe_counts,
    is_weekend_or_holiday,
    keep_mask,
    label_hours,
    labor_day_hour,
)
from src.data.dataset import GridFlowDataset


def test_listed_holidays_and_labor_day_are_weekend_class():
    holidays = [
        date(2016, 1, 1),
        date(2016, 1, 18),
        date(2016, 2, 15),
        date(2016, 5, 30),
        date(2016, 7, 4),
        date(2016, 9, 5),
    ]
    origin = date(2016, 1, 1)
    hours = np.array([(day - origin).days * 24 + 12 for day in holidays])
    assert is_weekend_or_holiday(hours).all()
    # A plain Wednesday is not.
    wednesday = (date(2016, 8, 31) - origin).days * 24
    assert not bool(is_weekend_or_holiday(np.array([wednesday]))[0])
    assert labor_day_hour() == 5952


def test_washington_segments_match_the_plan():
    counts = describe_counts()
    assert counts["dc_weekend_hours"] == 48
    assert counts["dc_weekday_hours"] == 72
    assert counts["dc_weekend_range"] == list(DC_WEEKEND_LABELS)
    assert counts["dc_weekday_range"] == list(DC_WEEKDAY_LABELS)
    assert counts["dc_weekend_start"] == "2016-08-27"
    assert counts["dc_weekday_end"] == "2016-09-02"
    assert counts["labor_day_in_val"]
    assert counts["train_weekday_hours"] + counts["train_weekend_hours"] == 5898
    train = label_hours("train")
    assert int(keep_mask(train, "dc_wd3_we2").sum()) == 120
    assert int(keep_mask(train, "dc_wd3").sum()) == 72
    assert int(keep_mask(train, "dc_we2").sum()) == 48


def test_empty_calendar_does_not_filter():
    train = GridFlowDataset(np.zeros((2, 6, 1, 1), np.float32), np.zeros((2, 1, 1, 1), np.float32))
    prepared = {"train": train, "val": train, "test": train}
    assert apply_calendar(prepared, None) is prepared
    assert apply_calendar(prepared, {}) is prepared
