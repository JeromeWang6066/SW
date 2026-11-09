"""Calendar windows for the five-day bin agreement. No city arrays."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "calendar_bin_agreement",
    ROOT / "scripts" / "11_calendar_bin_agreement.py",
)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def test_weekend_and_weekday_windows_do_not_overlap() -> None:
    report = mod.window_report()
    assert (report["weekend_start"], report["weekend_end"]) == (5736, 5784)
    assert (report["weekday_start"], report["weekday_end"]) == (5832, 5904)
    assert report["weekend_end"] <= report["weekday_start"]
    assert report["weekend_first_date"] == "2016-08-27"
    assert report["weekend_last_date"] == "2016-08-28"
    assert report["weekday_first_date"] == "2016-08-31"
    assert report["weekday_last_date"] == "2016-09-02"
    assert np.array_equal(report["weekend_hour_counts"], np.full(24, 2))
    assert np.array_equal(report["weekday_hour_counts"], np.full(24, 3))
