#!/usr/bin/env python3
"""Write per-cell weekday and weekend 24h profiles for taxi pickups.

Does not train, match cities, or download data. Exits non-zero before writing
if any city array is missing or is not the 8784-hour grid.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    TRAIN_PREFIX_HOURS,
    grid_hour_profile,
    hourly_calendar,
)

CITIES = ("ny", "chi", "dc")


def _period(flow: np.ndarray, is_weekday: np.ndarray) -> dict[str, Any]:
    weekday, weekday_counts = grid_hour_profile(flow, is_weekday)
    weekend, weekend_counts = grid_hour_profile(flow, ~is_weekday)
    return {
        "hours": int(flow.shape[0]),
        "weekday_profile": weekday,
        "weekend_profile": weekend,
        "weekday_hour_counts": weekday_counts,
        "weekend_hour_counts": weekend_counts,
    }


def _city_record(city: str, flow: np.ndarray) -> dict[str, Any]:
    if flow.shape[0] != EXPECTED_SERIES_HOURS:
        raise ValueError(
            f"{city} T={flow.shape[0]} != {EXPECTED_SERIES_HOURS}; "
            "refusing to apply the 2016-01-01 calendar"
        )
    _, is_weekday = hourly_calendar(EXPECTED_SERIES_HOURS, CALENDAR_START)
    train = flow[:TRAIN_PREFIX_HOURS]
    _, train_weekday = hourly_calendar(TRAIN_PREFIX_HOURS, CALENDAR_START)
    valid = valid_region_mask(flow)
    return {
        "city": city,
        "channel": "taxi_pickup",
        "calendar_start": f"{CALENDAR_START.isoformat()}T00:00:00",
        "calendar_assumption": (
            "Hour 0 is 2016-01-01 00:00 local, with no daylight-saving "
            "insertion or deletion. CrossTReS Table 2 states this range for "
            "New York. Chicago and Washington share it because each series "
            "is also 8784 hours."
        ),
        "shape_thw": [int(v) for v in flow.shape],
        "valid_mask": valid,
        "n_valid": int(valid.sum()),
        "full_year": _period(flow, is_weekday),
        "train_prefix": _period(train, train_weekday),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return value.astype(int).tolist()
        if np.issubdtype(value.dtype, np.integer):
            return value.astype(int).tolist()
        return value.astype(float).tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _valid_mean(profile: np.ndarray, valid: np.ndarray) -> np.ndarray:
    chosen = profile[valid]
    if chosen.size == 0:
        return np.zeros(24, dtype=np.float64)
    return chosen.mean(axis=0)


def _fmt_counts(counts: np.ndarray) -> str:
    return ", ".join(str(int(v)) for v in counts)


def _fmt_profile(values: np.ndarray) -> str:
    return ", ".join(f"{float(v):.4f}" for v in values)


def _readme(records: list[dict[str, Any]]) -> str:
    lines = [
        "# 出租车上车量的 24 小时轮廓",
        "",
        "每个格子、每个钟点的均值：",
        "",
        "`profile[r, h] = mean { flow[t, r] | 选中的日期 and hour(t) = h }`",
        "",
        "- 通道是 `Taxi{CITY}_pickup.npy`。下车和单车这次没算。",
        "- 工作日是周一到周五。周末轮廓写在同一份 JSON 里，供对照。",
        "- 无效格（全年上车量之和为 0）保留，轮廓为 0。有效格数量用这个掩码。",
        "- `full_year` 用全部 8784 小时，只供查看。以后若用轮廓选择方法，用 `train_prefix`。",
        f"- `train_prefix` 是去掉验证、测试各 1440 小时后的前 {TRAIN_PREFIX_HOURS} 小时。",
        "- 第 0 小时是 2016-01-01 00:00。不做夏令时插删。纽约的日期来自 CrossTReS 附录 Table 2；芝加哥和华盛顿因为同样是 8784 小时而共用这个起点。",
        "- JSON 是完整精度。下面的 24 个数是有效格平均后再保留四位小数。",
        "- 这里没有城市之间的 Pearson，也没有 0.8 / 0.5。",
        "",
        "展平是行优先：`(H, W, 24)` 变成 `(H*W, 24)`。",
        "",
    ]
    for rec in records:
        city = rec["city"]
        t, h, w = rec["shape_thw"]
        valid = rec["valid_mask"]
        full = rec["full_year"]
        train = rec["train_prefix"]
        lines.extend(
            [
                f"## {city}",
                "",
                f"- 形状 `T×H×W`：{t}×{h}×{w}",
                f"- 有效格：{rec['n_valid']} / {h * w}",
                f"- 全年工作日每个钟点的样本数：{_fmt_counts(full['weekday_hour_counts'])}",
                f"- 全年周末每个钟点的样本数：{_fmt_counts(full['weekend_hour_counts'])}",
                f"- 训练段工作日每个钟点的样本数：{_fmt_counts(train['weekday_hour_counts'])}",
                f"- 训练段周末每个钟点的样本数：{_fmt_counts(train['weekend_hour_counts'])}",
                f"- 全年工作日，有效格平均：{_fmt_profile(_valid_mean(full['weekday_profile'], valid))}",
                f"- 训练段工作日，有效格平均：{_fmt_profile(_valid_mean(train['weekday_profile'], valid))}",
                "",
            ]
        )
    lines.extend(
        [
            "文件：`ny_taxi_pickup.json`、`chi_taxi_pickup.json`、`dc_taxi_pickup.json`。",
            "",
            "重算：",
            "",
            "```bash",
            "python scripts/04_grid_profiles.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_profiles(processed_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    loaded: list[tuple[str, np.ndarray]] = []
    for city in CITIES:
        loaded.append((city, load_city_array(city, processed_dir)))
    records = [_city_record(city, flow) for city, flow in loaded]
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        path = out_dir / f"{rec['city']}_taxi_pickup.json"
        path.write_text(
            json.dumps(_jsonable(rec), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        print(
            f"  {rec['city']} shape={tuple(rec['shape_thw'])} "
            f"valid={rec['n_valid']} -> {path.relative_to(ROOT)}"
        )
    readme = out_dir / "README.md"
    readme.write_text(_readme(records), encoding="utf-8")
    print(f"  summary -> {readme.relative_to(ROOT)}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "profiles")
    args = parser.parse_args()
    try:
        write_profiles(args.processed_dir, args.out_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"profile computation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
