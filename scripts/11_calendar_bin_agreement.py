#!/usr/bin/env python3
"""Bin agreement for the 3-weekday plus 2-weekend Washington labels.

Weekday labels repeat the old 72-hour window and must match its published
agreement. Weekend labels are Saturday and Sunday only. No training.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import load_city_array, valid_region_mask
from src.utils.pearson_weekday_profile import (
    BIN_AGREEMENT_GATE,
    BIN_HIGH,
    BIN_LOW,
    CALENDAR_START,
    EXPECTED_SERIES_HOURS,
    TRAIN_PREFIX_HOURS,
    assign_similarity_bin,
    bin_agreement,
    few_shot_label_bounds,
    grid_hour_profile,
    hourly_calendar,
    max_source_pearson,
)

PAIRS = (("ny", "dc"), ("chi", "dc"))
WEEKEND_LABELS = (5736, 5784)
PUBLISHED_WEEKDAY = {
    ("ny", "dc"): (0.4656, 88, 189),
    ("chi", "dc"): (0.5026, 95, 189),
}
PUBLISHED_ATOL = 1e-4
BIN_NAMES = {0: "le_0.5", 1: "gt_0.5_le_0.8", 2: "gt_0.8"}


def weekday_mask(num_timesteps: int = EXPECTED_SERIES_HOURS) -> np.ndarray:
    _, weekday = hourly_calendar(num_timesteps)
    return weekday


def weekend_mask(num_timesteps: int = EXPECTED_SERIES_HOURS) -> np.ndarray:
    """Saturday and Sunday. Federal holidays that fall on weekdays stay out."""
    return ~weekday_mask(num_timesteps)


def _hour_counts(select: np.ndarray) -> np.ndarray:
    hours = np.arange(select.shape[0], dtype=np.int64) % 24
    return np.bincount(hours[select], minlength=24).astype(np.int64)


def window_report() -> dict[str, Any]:
    """Calendar facts for the two short windows. Does not load flow."""
    weekday_start, weekday_end = few_shot_label_bounds()
    weekend_start, weekend_end = WEEKEND_LABELS
    if (weekday_start, weekday_end) != (5832, 5904):
        raise ValueError(f"weekday labels {(weekday_start, weekday_end)} != (5832, 5904)")
    mask = weekend_mask()
    weekend_slice = mask[weekend_start:weekend_end]
    weekday_slice = weekday_mask()[weekday_start:weekday_end]
    if weekend_start % 24 != 0 or weekday_start % 24 != 0:
        raise ValueError("a short window does not start at midnight")
    if not bool(weekend_slice.all()):
        raise ValueError("weekend labels include a Monday-Friday hour")
    if not bool(weekday_slice.all()):
        raise ValueError("weekday labels include a Saturday or Sunday")
    if weekend_end > weekday_start:
        raise ValueError("weekend and weekday labels overlap")
    weekend_counts = _hour_counts(np.ones(weekend_end - weekend_start, dtype=bool))
    weekday_counts = _hour_counts(weekday_slice)
    return {
        "weekend_start": weekend_start,
        "weekend_end": weekend_end,
        "weekday_start": weekday_start,
        "weekday_end": weekday_end,
        "weekend_first_date": (CALENDAR_START + timedelta(days=weekend_start // 24)).isoformat(),
        "weekend_last_date": (CALENDAR_START + timedelta(days=(weekend_end - 1) // 24)).isoformat(),
        "weekday_first_date": (CALENDAR_START + timedelta(days=weekday_start // 24)).isoformat(),
        "weekday_last_date": (CALENDAR_START + timedelta(days=(weekday_end - 1) // 24)).isoformat(),
        "weekend_hour_counts": weekend_counts,
        "weekday_hour_counts": weekday_counts,
    }


def _counts(bins: np.ndarray) -> dict[str, int]:
    defined = bins[bins >= 0]
    return {
        "n_defined": int(defined.size),
        "n_undefined": int((bins < 0).sum()),
        **{BIN_NAMES[code]: int((defined == code).sum()) for code in BIN_NAMES},
    }


def _pair(source_rows: np.ndarray, target_short: np.ndarray, target_long: np.ndarray) -> dict[str, Any]:
    short_rho, _short_defined = max_source_pearson(source_rows, target_short)
    long_rho, _long_defined = max_source_pearson(source_rows, target_long)
    short_bins = assign_similarity_bin(short_rho)
    long_bins = assign_similarity_bin(long_rho)
    stats = bin_agreement(short_bins, long_bins)
    return {
        "agreement": stats["agreement"],
        "passes_gate": stats["passes_gate"],
        "gate": BIN_AGREEMENT_GATE,
        "n_both_defined": stats["n_both_defined"],
        "n_same": stats["n_same"],
        "n_only_protocol": stats["n_only_protocol"],
        "n_only_control": stats["n_only_control"],
        "protocol_counts": _counts(short_bins),
        "control_counts": _counts(long_bins),
    }


def _check_published(pairs: list[dict[str, Any]]) -> None:
    mismatches = []
    for item in pairs:
        expected = PUBLISHED_WEEKDAY[(item["source"], item["target"])]
        agreement, n_same, n_both = expected
        got = item["agreement"]
        if got is None or abs(got - agreement) > PUBLISHED_ATOL or item["n_same"] != n_same or item["n_both_defined"] != n_both:
            mismatches.append(
                f"{item['source']}->{item['target']} got {got} ({item['n_same']}/{item['n_both_defined']}) "
                f"expected {agreement} ({n_same}/{n_both})"
            )
    if mismatches:
        raise ValueError("weekday agreement does not match the published 72-hour result:\n" + "\n".join(mismatches))


def build_report(processed_dir: Path) -> dict[str, Any]:
    windows = window_report()
    if not np.array_equal(windows["weekend_hour_counts"], np.full(24, 2)):
        raise ValueError("weekend short window is not 2 samples per hour")
    if not np.array_equal(windows["weekday_hour_counts"], np.full(24, 3)):
        raise ValueError("weekday short window is not 3 samples per hour")
    weekday = weekday_mask()
    weekend = weekend_mask()
    flows = {}
    for city in ("ny", "chi", "dc"):
        flow = load_city_array(city, processed_dir)
        if flow.shape[0] != EXPECTED_SERIES_HOURS:
            raise ValueError(f"{city} T={flow.shape[0]} != {EXPECTED_SERIES_HOURS}")
        valid = valid_region_mask(flow)
        flows[city] = {"flow": flow, "valid": valid, "n_valid": int(valid.sum())}

    def rows(city: str, select: np.ndarray, flow_slice: np.ndarray | None = None) -> np.ndarray:
        flow = flows[city]["flow"] if flow_slice is None else flow_slice
        profile, _counts_out = grid_hour_profile(flow, select)
        return profile[flows[city]["valid"]]

    kinds = {}
    for name, short_bounds, select_full in (
        ("weekday", (windows["weekday_start"], windows["weekday_end"]), weekday),
        ("weekend", (windows["weekend_start"], windows["weekend_end"]), weekend),
    ):
        start, end = short_bounds
        short_flow = flows["dc"]["flow"][start:end]
        short_select = np.ones(end - start, dtype=bool) if name == "weekend" else select_full[start:end]
        kinds[name] = {
            "source": {
                city: rows(city, select_full[:TRAIN_PREFIX_HOURS], flows[city]["flow"][:TRAIN_PREFIX_HOURS])
                for city in ("ny", "chi")
            },
            "dc_short": rows("dc", short_select, short_flow),
            "dc_long": rows("dc", select_full[:TRAIN_PREFIX_HOURS], flows["dc"]["flow"][:TRAIN_PREFIX_HOURS]),
        }

    weekday_pairs = []
    weekend_pairs = []
    for source, target in PAIRS:
        for name, bucket in (("weekday", weekday_pairs), ("weekend", weekend_pairs)):
            item = _pair(kinds[name]["source"][source], kinds[name]["dc_short"], kinds[name]["dc_long"])
            item["source"] = source
            item["target"] = target
            item["kind"] = name
            item["n_target_valid"] = flows[target]["n_valid"]
            bucket.append(item)
    _check_published(weekday_pairs)
    return {
        "channel": "taxi_pickup",
        "trained": False,
        "bin_low": BIN_LOW,
        "bin_high": BIN_HIGH,
        "gate": BIN_AGREEMENT_GATE,
        "windows": {key: value for key, value in windows.items() if not str(key).endswith("hour_counts")},
        "weekend_hour_counts": windows["weekend_hour_counts"],
        "weekday_hour_counts": windows["weekday_hour_counts"],
        "weekday_pairs": weekday_pairs,
        "weekend_pairs": weekend_pairs,
    }


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "无定义"
    return f"{value:.4f}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _section(lines: list[str], title: str, pairs: list[dict[str, Any]], short_name: str) -> None:
    lines.extend([f"## {title}", ""])
    for item in pairs:
        protocol = item["protocol_counts"]
        control = item["control_counts"]
        passed = "达到门槛" if item["passes_gate"] else "未达到门槛"
        lines.extend(
            [
                f"### {item['source']} → {item['target']}",
                "",
                f"- 一致比例 {_fmt_ratio(item['agreement'])}，{passed}。",
                f"- 两次都有定义 {item['n_both_defined']}，其中档位相同 {item['n_same']}。",
                f"- 只在 {short_name}有定义 {item['n_only_protocol']}，只在 5904 小时有定义 {item['n_only_control']}。",
                f"- {short_name}分档人数："
                f"不超过 0.5 为 {protocol['le_0.5']}，"
                f"0.5 到 0.8 为 {protocol['gt_0.5_le_0.8']}，"
                f"高于 0.8 为 {protocol['gt_0.8']}，"
                f"无定义 {protocol['n_undefined']}。",
                "- 5904 小时分档人数："
                f"不超过 0.5 为 {control['le_0.5']}，"
                f"0.5 到 0.8 为 {control['gt_0.5_le_0.8']}，"
                f"高于 0.8 为 {control['gt_0.8']}，"
                f"无定义 {control['n_undefined']}。",
                "",
            ]
        )


def _markdown(report: dict[str, Any]) -> str:
    windows = report["windows"]
    lines = [
        "# 五日训练窗口的格子分档一致比例",
        "",
        "通道是出租上车。只看纽约→华盛顿、芝加哥→华盛顿。这一步没有训练。",
        "工作日短窗口是 "
        f"[{windows['weekday_start']}, {windows['weekday_end']})，"
        f"{windows['weekday_first_date']} 至 {windows['weekday_last_date']}，每个钟点 3 个样本。",
        "它和原来的 72 小时是同一段，所以工作日一致比例必须和已发表的结果一致。",
        "周末短窗口是 "
        f"[{windows['weekend_start']}, {windows['weekend_end']})，"
        f"{windows['weekend_first_date']} 至 {windows['weekend_last_date']}，每个钟点 2 个样本。",
        "周末只含周六和周日，不含落到周一到周五的联邦假日。",
        "源城和华盛顿的长窗口都用前 5904 小时里的同一种日期。",
        f"分档是不超过 {report['bin_low']}、大于 {report['bin_low']} 且不超过 {report['bin_high']}、大于 {report['bin_high']}。",
        "一致比例的分母是两次都有定义的格子。",
        f"门槛是 {report['gate']}。这一步即使达到门槛也不训练。",
        "",
    ]
    _section(lines, "工作日", report["weekday_pairs"], "72 小时")
    _section(lines, "周末", report["weekend_pairs"], "48 小时")
    lines.extend(
        [
            "重算：",
            "",
            "```bash",
            "python scripts/11_calendar_bin_agreement.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(processed_dir: Path, out_dir: Path) -> dict[str, Any]:
    report = build_report(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "calendar_bin_agreement.md"
    json_path = out_dir / "calendar_bin_agreement.json"
    json_path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(f"wrote {md_path}")
    for name in ("weekday_pairs", "weekend_pairs"):
        for item in report[name]:
            print(
                f"  {item['kind']} {item['source']}->{item['target']} "
                f"agreement={_fmt_ratio(item['agreement'])} pass={item['passes_gate']}"
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "profiles")
    args = parser.parse_args()
    try:
        write_report(args.processed_dir, args.out_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"calendar bin agreement failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
