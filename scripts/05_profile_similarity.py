#!/usr/bin/env python3
"""Protocol similarity of weekday taxi-pickup profiles. No training.

Source profiles use the first 5904 hours. Target profiles use the last 72
training-label hours. Constant profiles are excluded instead of scored as 0.
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

from src.data.dataset import load_city_array, valid_region_mask
from src.utils.pearson_weekday_profile import (
    CALENDAR_START,
    EXPECTED_SERIES_HOURS,
    TRAIN_PREFIX_HOURS,
    few_shot_label_bounds,
    grid_hour_profile,
    hourly_calendar,
    max_source_pearson,
    paired_profile_pearson,
)

CITIES = ("ny", "chi", "dc")
BIN_LOW = 0.5
BIN_HIGH = 0.8


def _load_saved_train_profile(profile_dir: Path, city: str) -> np.ndarray:
    path = profile_dir / f"{city}_taxi_pickup.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing saved profile {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    saved = np.asarray(payload["train_prefix"]["weekday_profile"], dtype=np.float64)
    return saved


def _weekday_profile(flow: np.ndarray, select: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    profile, counts = grid_hour_profile(flow, select)
    return profile, counts


def _mean_profile(profile: np.ndarray, valid: np.ndarray) -> np.ndarray:
    chosen = profile[valid]
    if chosen.size == 0:
        raise ValueError("no valid cells")
    return chosen.mean(axis=0)


def _rho_summary(rho: np.ndarray, defined: np.ndarray) -> dict[str, Any]:
    values = rho[defined]
    n_defined = int(defined.sum())
    n_excluded = int((~defined).sum())
    if n_defined == 0:
        return {
            "n_defined": 0,
            "n_excluded": n_excluded,
            "median": None,
            "frac_le_0.5": None,
            "frac_gt_0.5_le_0.8": None,
            "frac_gt_0.8": None,
        }
    return {
        "n_defined": n_defined,
        "n_excluded": n_excluded,
        "median": float(np.median(values)),
        "frac_le_0.5": float(np.mean(values <= BIN_LOW)),
        "frac_gt_0.5_le_0.8": float(np.mean((values > BIN_LOW) & (values <= BIN_HIGH))),
        "frac_gt_0.8": float(np.mean(values > BIN_HIGH)),
    }


def _finite_rho(left: np.ndarray, right: np.ndarray) -> float | None:
    rho, defined = paired_profile_pearson(left.reshape(1, 24), right.reshape(1, 24))
    if not bool(defined[0]):
        return None
    return float(rho[0])


def _fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "无定义"
    return f"{value:.4f}"


def _fmt_counts(counts: np.ndarray) -> str:
    return ", ".join(str(int(v)) for v in counts)


def build_similarity(processed_dir: Path, profile_dir: Path) -> dict[str, Any]:
    few_start, few_end = few_shot_label_bounds()
    _, weekday = hourly_calendar(EXPECTED_SERIES_HOURS)
    cities: dict[str, dict[str, Any]] = {}
    for city in CITIES:
        flow = load_city_array(city, processed_dir)
        if flow.shape[0] != weekday.shape[0]:
            raise ValueError(f"{city} T={flow.shape[0]} != {weekday.shape[0]}")
        valid = valid_region_mask(flow)
        long_profile, _ = _weekday_profile(flow[:TRAIN_PREFIX_HOURS], weekday[:TRAIN_PREFIX_HOURS])
        saved = _load_saved_train_profile(profile_dir, city)
        if long_profile.shape != saved.shape or not np.allclose(long_profile, saved, rtol=0, atol=1e-8):
            raise ValueError(f"{city} recomputed train_prefix profile does not match {profile_dir}")
        short_profile, short_counts = _weekday_profile(
            flow[few_start:few_end], weekday[few_start:few_end]
        )
        long_rows = long_profile[valid]
        short_rows = short_profile[valid]
        stability, stability_defined = paired_profile_pearson(short_rows, long_rows)
        city_rho = _finite_rho(_mean_profile(short_profile, valid), _mean_profile(long_profile, valid))
        cities[city] = {
            "n_valid": int(valid.sum()),
            "long_mean": _mean_profile(long_profile, valid),
            "short_mean": _mean_profile(short_profile, valid),
            "long_rows": long_rows,
            "short_rows": short_rows,
            "short_hour_counts": short_counts,
            "stability_city_mean": city_rho,
            "stability_cells": _rho_summary(stability, stability_defined),
        }

    protocol = []
    for source, target in permutations(CITIES, 2):
        rho = _finite_rho(cities[source]["long_mean"], cities[target]["short_mean"])
        protocol.append({"source": source, "target": target, "rho": rho})

    same_window = []
    for left, right in combinations(CITIES, 2):
        rho = _finite_rho(cities[left]["long_mean"], cities[right]["long_mean"])
        same_window.append({"a": left, "b": right, "rho": rho})

    cell_level = []
    for source, target in permutations(CITIES, 2):
        rho, defined = max_source_pearson(cities[source]["long_rows"], cities[target]["short_rows"])
        cell_level.append(
            {"source": source, "target": target, **_rho_summary(rho, defined)}
        )

    first_counts = cities[CITIES[0]]["short_hour_counts"]
    for city in CITIES[1:]:
        if not np.array_equal(cities[city]["short_hour_counts"], first_counts):
            raise ValueError("72-hour weekday counts differ across cities")

    return {
        "channel": "taxi_pickup",
        "few_shot_label_start": few_start,
        "few_shot_label_end": few_end,
        "few_shot_first_date": (CALENDAR_START + timedelta(days=few_start // 24)).isoformat(),
        "few_shot_last_date": (CALENDAR_START + timedelta(days=(few_end - 1) // 24)).isoformat(),
        "short_weekday_hour_counts": first_counts,
        "cities": cities,
        "city_level_protocol": protocol,
        "city_level_same_window_5904": same_window,
        "cell_level": cell_level,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.integer):
            return value.astype(int).tolist()
        return [None if not np.isfinite(v) else float(v) for v in value.astype(float).ravel()] if value.ndim == 1 else value.astype(float).tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items() if key not in {"long_rows", "short_rows", "long_mean", "short_mean"}}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 协议内的轮廓相似度",
        "",
        "通道是出租车上车量的工作日轮廓。源城用前 5904 小时。目标城用最后 72 个训练标签，原始下标 "
        f"[{report['few_shot_label_start']}, {report['few_shot_label_end']})，"
        f"日期 {report['few_shot_first_date']} 至 {report['few_shot_last_date']}。",
        "这 72 小时之前的 6 小时只作为网络输入，不进入轮廓。",
        "",
        f"这 72 小时里每个钟点的工作日样本数：{_fmt_counts(report['short_weekday_hour_counts'])}。",
        "常数轮廓的 Pearson 记为无定义，不记成 0。",
        "0.5 和 0.8 只用来描述格子级最大值落在哪一档，不是已经选定的规则。",
        "协议内的城市级相关随方向改变，因为只有目标城使用 72 小时。两边都用 5904 小时时，三个无序对各有一个对称值。",
        "",
        "## 同一座城：72 小时对 5904 小时",
        "",
    ]
    for city in CITIES:
        item = report["cities"][city]
        cells = item["stability_cells"]
        lines.append(
            f"- {city}：有效格 {item['n_valid']}；城市平均轮廓 {_fmt(item['stability_city_mean'])}；"
            f"格子中位数 {_fmt(cells['median'])}；"
            f"有定义 {cells['n_defined']}，剔除 {cells['n_excluded']}"
        )
    lines.extend(["", "## 城市级，源城 5904 小时对目标城 72 小时", ""])
    for item in report["city_level_protocol"]:
        lines.append(f"- {item['source']} → {item['target']}：{_fmt(item['rho'])}")
    lines.extend(["", "## 城市级，两边都是 5904 小时", ""])
    for item in report["city_level_same_window_5904"]:
        lines.append(f"- {item['a']} 与 {item['b']}：{_fmt(item['rho'])}")
    lines.extend(["", "## 格子级最大相关", ""])
    for item in report["cell_level"]:
        lines.append(
            f"- {item['source']} → {item['target']}：中位数 {_fmt(item['median'])}；"
            f"≤0.5 为 {_fmt(item['frac_le_0.5'])}，(0.5, 0.8] 为 {_fmt(item['frac_gt_0.5_le_0.8'])}，"
            f">0.8 为 {_fmt(item['frac_gt_0.8'])}；"
            f"有定义 {item['n_defined']}，剔除 {item['n_excluded']}"
        )
    lines.extend(
        [
            "",
            "重算：",
            "",
            "```bash",
            "python scripts/05_profile_similarity.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_similarity(processed_dir: Path, profile_dir: Path, out_dir: Path) -> None:
    report = build_similarity(processed_dir, profile_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "similarity.json"
    md_path = out_dir / "similarity.md"
    json_path.write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(f"  similarity -> {md_path.relative_to(ROOT)}")
    for item in report["city_level_protocol"]:
        print(f"  {item['source']}->{item['target']} city rho={_fmt(item['rho'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--profile-dir", type=Path, default=ROOT / "docs" / "profiles")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "profiles")
    args = parser.parse_args()
    try:
        write_similarity(args.processed_dir, args.profile_dir, args.out_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"similarity computation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
