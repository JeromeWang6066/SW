"""Print weekday and weekend-plus-holiday label counts. Exit 2 on a mismatch.

The numbers that must match the plan are the Washington segments and Labor Day's
place in the validation window. Source-city counts are printed and only checked
for adding up to the train-label length.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.calendar_mask import describe_counts


def main() -> int:
    counts = describe_counts()
    print(
        "source train labels: "
        f"weekday {counts['train_weekday_hours']}, "
        f"weekend+holiday {counts['train_weekend_hours']}, "
        f"total {counts['train_hours']}"
    )
    print(
        "washington weekend "
        f"{counts['dc_weekend_range']} {counts['dc_weekend_start']}..{counts['dc_weekend_end']} "
        f"hours={counts['dc_weekend_hours']}"
    )
    print(
        "washington weekday "
        f"{counts['dc_weekday_range']} {counts['dc_weekday_start']}..{counts['dc_weekday_end']} "
        f"hours={counts['dc_weekday_hours']}"
    )
    print(
        "labor day hour "
        f"{counts['labor_day_hour']} "
        f"in_val={counts['labor_day_in_val']} "
        f"weekend_class={counts['labor_day_is_weekend_class']}"
    )
    ok = (
        counts["train_weekday_hours"] + counts["train_weekend_hours"] == counts["train_hours"]
        and counts["dc_weekend_hours"] == 48
        and counts["dc_weekday_hours"] == 72
        and counts["dc_weekend_range"] == [5736, 5784]
        and counts["dc_weekday_range"] == [5832, 5904]
        and counts["dc_weekend_start"] == "2016-08-27"
        and counts["dc_weekend_end"] == "2016-08-28"
        and counts["dc_weekday_start"] == "2016-08-31"
        and counts["dc_weekday_end"] == "2016-09-02"
        and counts["dc_weekend_all_weekend_class"]
        and counts["dc_weekday_all_weekday_class"]
        and counts["labor_day_in_val"]
        and counts["labor_day_is_weekend_class"]
        and counts["labor_day_hour"] == 5952
    )
    if not ok:
        print("calendar counts do not match the plan; not training")
        return 2
    print("calendar counts match the plan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
