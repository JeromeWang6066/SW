#!/usr/bin/env python3
"""City-level weekday Pearson across taxi pickup, taxi dropoff, and bike pickup.

No training. Weekdays are Monday–Friday. A flat city profile is undefined,
not zero. The published taxi-pickup numbers are checked before any file is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from itertools import combinations, permutations
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.pearson_weekday_profile import (
    CALENDAR_START,
    EXPECTED_SERIES_HOURS,
    TRAIN_PREFIX_HOURS,
    few_shot_label_bounds,
    grid_hour_profile,
    hourly_calendar,
    paired_profile_pearson,
)

CITIES = ("ny", "chi", "dc")
CHANNELS = ("taxi_pickup", "taxi_dropoff", "bike_pickup")
GRIDS = {"ny": (20, 23), "chi": (17, 28), "dc": (21, 20)}
EXPECTED_VALID = {
    ("ny", "taxi_pickup"): 458,
    ("chi", "taxi_pickup"): 270,
    ("dc", "taxi_pickup"): 207,
    ("ny", "bike_pickup"): 108,
    ("chi", "bike_pickup"): 229,
    ("dc", "bike_pickup"): 125,
}
PUBLISHED_PROTOCOL = {
    ("ny", "chi"): 0.9359,
    ("ny", "dc"): 0.8598,
    ("chi", "ny"): 0.9322,
    ("chi", "dc"): 0.9444,
    ("dc", "ny"): 0.8519,
    ("dc", "chi"): 0.9643,
}
PUBLISHED_SAME_WINDOW = {
    frozenset(("ny", "chi")): 0.9353,
    frozenset(("ny", "dc")): 0.8746,
    frozenset(("chi", "dc")): 0.9601,
}
PUBLISHED_ATOL = 1e-4
BIN_LOW = 0.5
BIN_HIGH = 0.8


def series_id(city: str, channel: str) -> str:
    return f"{city}:{channel}"


def file_name(city: str, channel: str) -> str:
    kind, flow = channel.split("_")
    prefix = "Taxi" if kind == "taxi" else "Bike"
    return f"{prefix}{city.upper()}_{flow}.npy"


def finite_rho(left: np.ndarray, right: np.ndarray) -> float | None:
    rho, defined = paired_profile_pearson(np.asarray(left, dtype=np.float64).reshape(1, 24), np.asarray(right, dtype=np.float64).reshape(1, 24))
    if not bool(defined[0]):
        return None
    return float(rho[0])


def assign_bin(rho: float | None) -> str:
    if rho is None:
        return "undefined"
    if rho <= BIN_LOW:
        return "le_0.5"
    if rho <= BIN_HIGH:
        return "gt_0.5_le_0.8"
    return "gt_0.8"


def classify_pair(protocol: float | None, same_window: float | None) -> str:
    """Training candidate only when both correlations exist and the long window is at most 0.8.

    A protocol value at or below 0.8 with a long-window value above 0.8 is
    short-window attenuation, not a candidate.
    """
    if protocol is None or same_window is None:
        return "undefined"
    if same_window <= BIN_HIGH:
        return "candidate"
    if protocol <= BIN_HIGH:
        return "attenuated"
    return "both_high"


def short_weekday_counts() -> np.ndarray:
    _, weekday = hourly_calendar(EXPECTED_SERIES_HOURS)
    start, end = few_shot_label_bounds()
    hours = np.arange(end - start, dtype=np.int64) % 24
    selected = weekday[start:end]
    return np.bincount(hours[selected], minlength=24).astype(np.int64)


def _load_flow(processed_dir: Path, city: str, channel: str) -> np.ndarray:
    path = processed_dir / city / "upstream" / file_name(city, channel)
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    flow = np.load(path)
    expected = (EXPECTED_SERIES_HOURS, *GRIDS[city])
    if tuple(flow.shape) != expected:
        raise ValueError(f"{path.name} shape {tuple(flow.shape)} != {expected}")
    return flow


def _profiles(flow: np.ndarray) -> dict[str, Any]:
    valid = flow.sum(axis=0) > 0
    if int(valid.sum()) == 0:
        raise ValueError("valid mask is empty")
    _, weekday = hourly_calendar(flow.shape[0])
    long_profile, _ = grid_hour_profile(flow[:TRAIN_PREFIX_HOURS], weekday[:TRAIN_PREFIX_HOURS])
    start, end = few_shot_label_bounds()
    short_profile, short_counts = grid_hour_profile(flow[start:end], weekday[start:end])
    expected_counts = short_weekday_counts()
    if not np.array_equal(short_counts, expected_counts):
        raise ValueError(f"72-hour counts {short_counts.tolist()} != {expected_counts.tolist()}")
    return {
        "n_valid": int(valid.sum()),
        "long_mean": long_profile[valid].mean(axis=0),
        "short_mean": short_profile[valid].mean(axis=0),
        "short_hour_counts": short_counts,
    }


def _check_published(protocol: list[dict[str, Any]], same_window: list[dict[str, Any]]) -> None:
    mismatches = []
    lookup = {
        (item["source_city"], item["target_city"]): item["rho"]
        for item in protocol
        if item["source_channel"] == "taxi_pickup" and item["target_channel"] == "taxi_pickup"
    }
    for key, expected in PUBLISHED_PROTOCOL.items():
        got = lookup.get(key)
        if got is None or abs(got - expected) > PUBLISHED_ATOL:
            mismatches.append(f"protocol {key[0]}->{key[1]} got {got} expected {expected}")
    same_lookup = {frozenset((item["a_city"], item["b_city"])): item["rho"] for item in same_window if item["a_channel"] == "taxi_pickup" and item["b_channel"] == "taxi_pickup"}
    for key, expected in PUBLISHED_SAME_WINDOW.items():
        got = same_lookup.get(key)
        if got is None or abs(got - expected) > PUBLISHED_ATOL:
            cities = sorted(key)
            mismatches.append(f"same-window {'-'.join(cities)} got {got} expected {expected}")
    if mismatches:
        raise ValueError("published taxi-pickup correlations do not match:\n" + "\n".join(mismatches))


def build_report(processed_dir: Path) -> dict[str, Any]:
    start, end = few_shot_label_bounds()
    series = []
    means: dict[str, dict[str, Any]] = {}
    for city in CITIES:
        for channel in CHANNELS:
            flow = _load_flow(processed_dir, city, channel)
            item = _profiles(flow)
            expected_valid = EXPECTED_VALID.get((city, channel))
            if expected_valid is not None and item["n_valid"] != expected_valid:
                raise ValueError(f"{city} {channel} valid cells {item['n_valid']} != {expected_valid}")
            key = series_id(city, channel)
            means[key] = item
            series.append(
                {
                    "id": key,
                    "city": city,
                    "channel": channel,
                    "shape": [EXPECTED_SERIES_HOURS, *GRIDS[city]],
                    "n_valid": item["n_valid"],
                    "long_mean": item["long_mean"],
                    "short_mean": item["short_mean"],
                }
            )

    protocol = []
    for source, target in permutations(series, 2):
        rho = finite_rho(means[source["id"]]["long_mean"], means[target["id"]]["short_mean"])
        protocol.append(
            {
                "source": source["id"],
                "target": target["id"],
                "source_city": source["city"],
                "source_channel": source["channel"],
                "target_city": target["city"],
                "target_channel": target["channel"],
                "rho": rho,
                "bin": assign_bin(rho),
            }
        )
    same_window = []
    for left, right in combinations(series, 2):
        rho = finite_rho(means[left["id"]]["long_mean"], means[right["id"]]["long_mean"])
        same_window.append(
            {
                "a": left["id"],
                "b": right["id"],
                "a_city": left["city"],
                "a_channel": left["channel"],
                "b_city": right["city"],
                "b_channel": right["channel"],
                "rho": rho,
                "bin": assign_bin(rho),
            }
        )
    if len(protocol) != 72 or len(same_window) != 36:
        raise ValueError(f"pair counts protocol={len(protocol)} same_window={len(same_window)}")
    _check_published(protocol, same_window)

    same_by_pair = {frozenset((item["a"], item["b"])): item for item in same_window}
    judged = []
    for item in protocol:
        same = same_by_pair[frozenset((item["source"], item["target"]))]
        label = classify_pair(item["rho"], same["rho"])
        judged.append({**item, "same_window_rho": same["rho"], "same_window_bin": same["bin"], "label": label})

    return {
        "channels": list(CHANNELS),
        "weekday": "Monday-Friday, no holiday reclassification",
        "few_shot_label_start": start,
        "few_shot_label_end": end,
        "few_shot_first_date": (CALENDAR_START + timedelta(days=start // 24)).isoformat(),
        "few_shot_last_date": (CALENDAR_START + timedelta(days=(end - 1) // 24)).isoformat(),
        "short_weekday_hour_counts": means[series_id("ny", "taxi_pickup")]["short_hour_counts"],
        "series": series,
        "protocol": protocol,
        "same_window": same_window,
        "judged": judged,
        "n_candidate": sum(item["label"] == "candidate" for item in judged),
        "n_attenuated": sum(item["label"] == "attenuated" for item in judged),
        "trained": False,
    }


def _fmt(value: float | None) -> str:
    if value is None:
        return "无定义"
    return f"{value:.4f}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [None if not np.isfinite(float(item)) else float(item) for item in value.astype(float).ravel()]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 模态之间的城市级相关",
        "",
        "通道是出租上车、出租下车、单车上车。单车下车没有算。工作日是周一到周五，不把假日从工作日里拿出去。",
        "源侧用前 5904 小时的工作日。目标侧用最后 72 个训练标签，下标 "
        f"[{report['few_shot_label_start']}, {report['few_shot_label_end']})，"
        f"日期 {report['few_shot_first_date']} 至 {report['few_shot_last_date']}。",
        "这 72 小时每个钟点有 3 个工作日样本。常数轮廓记为无定义，不记成 0。",
        "城市轮廓是该通道自己的有效格上的 24 小时均值。格子不在城市之间对齐。",
        "0.5 和 0.8 只用来分档。同窗口相关不超过 0.8，并且协议内相关也有定义，才是以后的训练候选。",
        "协议内不超过 0.8、同窗口仍高于 0.8 的，记成短窗口把相关压低了，不是候选。这一步没有训练。",
        "",
        f"有向协议内相关 {len(report['protocol'])} 个，无序同窗口相关 {len(report['same_window'])} 个。",
        f"训练候选 {report['n_candidate']} 个。短窗口压低 {report['n_attenuated']} 个。",
        "",
        "## 序列",
        "",
    ]
    for item in report["series"]:
        shape = "×".join(str(v) for v in item["shape"])
        lines.append(f"- {item['id']}：形状 {shape}，有效格 {item['n_valid']}")
    lines.extend(["", "## 协议内相关", "", "源侧是 5904 小时，目标侧是 72 小时。", ""])
    for item in report["judged"]:
        lines.append(
            f"- {item['source']} → {item['target']}：{_fmt(item['rho'])}（{item['bin']}），"
            f"同窗口 {_fmt(item['same_window_rho'])}，{item['label']}"
        )
    lines.extend(["", "## 同窗口相关", "", "两边都是前 5904 小时，一对只有一个数。", ""])
    for item in report["same_window"]:
        lines.append(f"- {item['a']} 与 {item['b']}：{_fmt(item['rho'])}（{item['bin']}）")
    lines.extend(["", "## 读法", ""])
    if report["n_candidate"] == 0:
        lines.append("没有训练候选。同窗口相关不超过 0.8 的有向对数量为 0。停在这里，不训练。")
    else:
        lines.append("训练候选是同窗口相关不超过 0.8 的有向对。这一步仍然不训练。")
        for item in report["judged"]:
            if item["label"] == "candidate":
                lines.append(f"- 候选 {item['source']} → {item['target']}：协议内 {_fmt(item['rho'])}，同窗口 {_fmt(item['same_window_rho'])}")
    if report["n_attenuated"]:
        lines.append("")
        lines.append("下面这些协议内相关落到 0.8 及以下，同窗口仍高于 0.8。")
        lines.append("")
        for item in report["judged"]:
            if item["label"] == "attenuated":
                lines.append(f"- {item['source']} → {item['target']}：协议内 {_fmt(item['rho'])}，同窗口 {_fmt(item['same_window_rho'])}")
    lines.extend(
        [
            "",
            "重算：",
            "",
            "```bash",
            "python scripts/10_modality_similarity.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(processed_dir: Path, out_dir: Path) -> dict[str, Any]:
    report = build_report(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "modality_similarity.json"
    md_path = out_dir / "modality_similarity.md"
    json_path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(f"wrote {md_path}")
    print(f"candidates {report['n_candidate']} attenuated {report['n_attenuated']}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "profiles")
    args = parser.parse_args()
    try:
        write_report(args.processed_dir, args.out_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"modality similarity failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
