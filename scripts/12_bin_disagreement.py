#!/usr/bin/env python3
"""Why a Washington cell changes similarity bin between the short and long window.

Two setups, taxi pickup only, no training:

- three weekdays: Monday–Friday, holidays kept, short labels [5832, 5904).
- five-day training calendar: weekday excludes the five federal holidays in the
  train prefix; weekend is Saturday, Sunday, and those holidays. Short weekday
  labels stay [5832, 5904). Short weekend labels are [5736, 5784).

For each cell defined on both windows, record whether the best source cell is
shared, how far the two maximum correlations move, and how similar the cell's
own two 24-hour profiles are.
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

from src.data.calendar_mask import is_weekend_or_holiday
from src.data.dataset import load_city_array, valid_region_mask
from src.utils.pearson_weekday_profile import (
    BIN_HIGH,
    BIN_LOW,
    EXPECTED_SERIES_HOURS,
    TRAIN_PREFIX_HOURS,
    assign_similarity_bin,
    bin_agreement,
    few_shot_label_bounds,
    grid_hour_profile,
    hourly_calendar,
    max_source_match,
    paired_profile_pearson,
)

PAIRS = (("ny", "dc"), ("chi", "dc"))
WEEKEND_LABELS = (5736, 5784)
BIN_LABELS = ("不超过 0.5", "0.5 到 0.8", "高于 0.8")
# Descriptive cuts for "the correlations barely moved". Not a training rule.
DELTA_CUTS = (0.02, 0.05, 0.10)
EXPECTED = {
    ("three_weekday", "weekday", "ny"): (88, 189),
    ("three_weekday", "weekday", "chi"): (95, 189),
    ("five_day", "weekday", "ny"): (91, 189),
    ("five_day", "weekday", "chi"): (93, 189),
    ("five_day", "weekend", "ny"): (106, 173),
    ("five_day", "weekend", "chi"): (84, 173),
}
SETUP_TITLE = {
    "three_weekday": "三天工作日",
    "five_day": "三天工作日加两天周末",
}
KIND_TITLE = {
    "weekday": "工作日",
    "weekend": "周末加节假日",
}
CITY = {"ny": "纽约", "chi": "芝加哥", "dc": "华盛顿"}


def _finite_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "n": int(values.size),
            "undefined": int(values.size),
            "gt_0.8": 0,
            "gt_0.5_le_0.8": 0,
            "le_0.5": 0,
            "median": None,
        }
    return {
        "n": int(values.size),
        "undefined": int(values.size - finite.size),
        "gt_0.8": int((finite > BIN_HIGH).sum()),
        "gt_0.5_le_0.8": int(((finite > BIN_LOW) & (finite <= BIN_HIGH)).sum()),
        "le_0.5": int((finite <= BIN_LOW).sum()),
        "median": float(np.median(finite)),
    }


def _delta_stats(delta: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(delta, dtype=np.float64)
    delta = delta[np.isfinite(delta)]
    if delta.size == 0:
        return {
            "n": 0,
            "le_0.02": 0,
            "le_0.05": 0,
            "le_0.10": 0,
            "gt_0.10": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    quantiles = np.quantile(delta, [0.0, 0.25, 0.5, 0.75, 1.0])
    slack = 1e-8
    return {
        "n": int(delta.size),
        "le_0.02": int((delta <= DELTA_CUTS[0] + slack).sum()),
        "le_0.05": int((delta <= DELTA_CUTS[1] + slack).sum()),
        "le_0.10": int((delta <= DELTA_CUTS[2] + slack).sum()),
        "gt_0.10": int((delta > DELTA_CUTS[2] + slack).sum()),
        "min": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "max": float(quantiles[4]),
    }


def _group(
    mask: np.ndarray,
    same_source: np.ndarray,
    delta: np.ndarray,
    self_rho: np.ndarray,
) -> dict[str, Any]:
    chosen = np.asarray(mask, dtype=bool)
    return {
        "n": int(chosen.sum()),
        "same_source": _pack(chosen & same_source, delta, self_rho),
        "different_source": _pack(chosen & ~same_source, delta, self_rho),
        "all": _pack(chosen, delta, self_rho),
    }


def _pack(mask: np.ndarray, delta: np.ndarray, self_rho: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(np.asarray(mask).sum()),
        "delta": _delta_stats(delta[mask]),
        "self_profile": _finite_stats(self_rho[mask]),
    }


def summarize_bin_changes(
    short_rho: np.ndarray,
    long_rho: np.ndarray,
    short_argmax: np.ndarray,
    long_argmax: np.ndarray,
    short_tied: np.ndarray,
    long_tied: np.ndarray,
    self_rho: np.ndarray,
) -> dict[str, Any]:
    """Split cells by whether the bin and the best source cell changed.

    A source cell counts as shared when it is within 1e-8 of the maximum on
    both windows. The first argmax alone is also reported, so a tie is visible.
    """
    short_rho = np.asarray(short_rho, dtype=np.float64)
    long_rho = np.asarray(long_rho, dtype=np.float64)
    if short_tied.shape != long_tied.shape:
        raise ValueError(f"tie shape {short_tied.shape} != {long_tied.shape}")
    if short_rho.shape != long_rho.shape or short_rho.shape[0] != short_tied.shape[0]:
        raise ValueError("rho and tie rows do not match")
    both = np.isfinite(short_rho) & np.isfinite(long_rho)
    short_bins = assign_similarity_bin(short_rho)
    long_bins = assign_similarity_bin(long_rho)
    if np.any(both & ((short_bins < 0) | (long_bins < 0))):
        raise ValueError("a finite correlation was left without a bin")
    shared = (short_tied & long_tied).any(axis=1)
    exact = (short_argmax == long_argmax) & (short_argmax >= 0)
    if np.any(both & exact & ~shared):
        raise ValueError("the first argmax is not inside the tie set")
    delta = np.full(short_rho.shape, np.nan, dtype=np.float64)
    delta[both] = np.abs(short_rho[both] - long_rho[both])
    diff = both & (short_bins != long_bins)
    same_bin = both & (short_bins == long_bins)
    transitions = np.zeros((3, 3), dtype=np.int64)
    for left, right in zip(short_bins[both], long_bins[both]):
        transitions[int(left), int(right)] += 1
    short_ties = short_tied.sum(axis=1) > 1
    long_ties = long_tied.sum(axis=1) > 1
    return {
        "n_both_defined": int(both.sum()),
        "n_same_bin": int(same_bin.sum()),
        "n_diff_bin": int(diff.sum()),
        "n_shared_source_but_argmax_differs": int((diff & shared & ~exact).sum()),
        "n_either_window_tied": int((diff & (short_ties | long_ties)).sum()),
        "transitions_short_by_long": transitions.tolist(),
        "different_bin": _group(diff, shared, delta, self_rho),
        "same_bin": _group(same_bin, shared, delta, self_rho),
    }


def _rows(flow: np.ndarray, select: np.ndarray, valid: np.ndarray) -> np.ndarray:
    profile, _counts = grid_hour_profile(flow, select)
    return profile[valid]


def _check_expected(setup: str, kind: str, source: str, agreement: dict[str, Any]) -> None:
    n_same, n_both = EXPECTED[(setup, kind, source)]
    if agreement["n_same"] != n_same or agreement["n_both_defined"] != n_both:
        raise ValueError(
            f"{setup} {kind} {source} got {agreement['n_same']}/{agreement['n_both_defined']} "
            f"expected {n_same}/{n_both}"
        )


def _one_pair(
    source_rows: np.ndarray,
    target_short: np.ndarray,
    target_long: np.ndarray,
) -> dict[str, Any]:
    short_rho, short_defined, short_argmax, short_tied = max_source_match(source_rows, target_short)
    long_rho, long_defined, long_argmax, long_tied = max_source_match(source_rows, target_long)
    short_bins = assign_similarity_bin(short_rho)
    long_bins = assign_similarity_bin(long_rho)
    agreement = bin_agreement(short_bins, long_bins)
    self_rho, _self_defined = paired_profile_pearson(target_short, target_long)
    if short_defined.shape != long_defined.shape:
        raise ValueError("short and long defined masks differ in shape")
    summary = summarize_bin_changes(
        short_rho,
        long_rho,
        short_argmax,
        long_argmax,
        short_tied,
        long_tied,
        self_rho,
    )
    if summary["n_same_bin"] != agreement["n_same"] or summary["n_both_defined"] != agreement["n_both_defined"]:
        raise ValueError("disagreement summary does not match bin agreement")
    return {"agreement": agreement, "summary": summary}


def build_report(processed_dir: Path) -> dict[str, Any]:
    weekday_start, weekday_end = few_shot_label_bounds()
    weekend_start, weekend_end = WEEKEND_LABELS
    if (weekday_start, weekday_end) != (5832, 5904):
        raise ValueError(f"weekday labels {(weekday_start, weekday_end)} != (5832, 5904)")
    hours = np.arange(EXPECTED_SERIES_HOURS, dtype=np.int64)
    monday_friday = hourly_calendar(EXPECTED_SERIES_HOURS)[1]
    holiday_weekend = is_weekend_or_holiday(hours)
    holiday_hours = int((monday_friday[:TRAIN_PREFIX_HOURS] & holiday_weekend[:TRAIN_PREFIX_HOURS]).sum())
    if holiday_hours != 120:
        raise ValueError(f"holiday hours in the train prefix: {holiday_hours}")
    masks = {
        "three_weekday": {"weekday": monday_friday},
        "five_day": {"weekday": ~holiday_weekend, "weekend": holiday_weekend},
    }
    short_bounds = {
        "weekday": (weekday_start, weekday_end),
        "weekend": (weekend_start, weekend_end),
    }
    flows = {}
    for city in ("ny", "chi", "dc"):
        flow = load_city_array(city, processed_dir)
        if flow.shape[0] != EXPECTED_SERIES_HOURS:
            raise ValueError(f"{city} T={flow.shape[0]} != {EXPECTED_SERIES_HOURS}")
        flows[city] = {"flow": flow, "valid": valid_region_mask(flow)}

    comparisons = []
    for setup, kinds in masks.items():
        for kind, select in kinds.items():
            start, end = short_bounds[kind]
            if not bool(select[start:end].all()):
                raise ValueError(f"{setup} {kind} short window is not entirely that class")
            dc_short = _rows(flows["dc"]["flow"][start:end], select[start:end], flows["dc"]["valid"])
            dc_long = _rows(
                flows["dc"]["flow"][:TRAIN_PREFIX_HOURS],
                select[:TRAIN_PREFIX_HOURS],
                flows["dc"]["valid"],
            )
            for source, target in PAIRS:
                source_rows = _rows(
                    flows[source]["flow"][:TRAIN_PREFIX_HOURS],
                    select[:TRAIN_PREFIX_HOURS],
                    flows[source]["valid"],
                )
                item = _one_pair(source_rows, dc_short, dc_long)
                _check_expected(setup, kind, source, item["agreement"])
                item["setup"] = setup
                item["kind"] = kind
                item["source"] = source
                item["target"] = target
                comparisons.append(item)
    return {
        "channel": "taxi_pickup",
        "trained": False,
        "bin_low": BIN_LOW,
        "bin_high": BIN_HIGH,
        "delta_cuts": list(DELTA_CUTS),
        "tie_atol": 1e-8,
        "holiday_hours_in_prefix": holiday_hours,
        "comparisons": comparisons,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _fmt_ratio(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.4f}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _source_table(group: dict[str, Any]) -> list[str]:
    lines = [
        "| 最大源格 | 格子数 | 自身轮廓相关的中位数 | 自身轮廓仍高于 0.8 | 相似度差的中位数 | 相似度差不超过 0.02 | 不超过 0.05 | 不超过 0.10 | 超过 0.10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("same_source", "相同"),
        ("different_source", "不同"),
        ("all", "合计"),
    ):
        block = group[key]
        delta = block["delta"]
        self_profile = block["self_profile"]
        lines.append(
            "| {label} | {n} | {self_median} | {self_high} | {delta_median} | {d02} | {d05} | {d10} | {drest} |".format(
                label=label,
                n=block["n"],
                self_median=_fmt(self_profile["median"]),
                self_high=self_profile["gt_0.8"],
                delta_median=_fmt(delta["median"]),
                d02=delta["le_0.02"],
                d05=delta["le_0.05"],
                d10=delta["le_0.10"],
                drest=delta["gt_0.10"],
            )
        )
    return lines


def _conclusion(report: dict[str, Any]) -> list[str]:
    lines = ["## 结论", ""]
    for item in report["comparisons"]:
        diff = item["summary"]["different_bin"]
        same = item["summary"]["same_bin"]["all"]["self_profile"]["median"]
        lines.append(
            "{setup}，{city}，{kind}：档位不同 {n} 个，其中最大源格不同 {changed} 个、相同 {kept} 个。"
            "源格不同的格子，相似度差中位数 {delta}，不超过 0.05 的有 {small} 个，自身轮廓仍高于 0.8 的有 {stable} 个。"
            "档位相同的格子，自身轮廓相关中位数 {same_self}。".format(
                setup=SETUP_TITLE[item["setup"]],
                city=CITY[item["source"]],
                kind=KIND_TITLE[item["kind"]],
                n=diff["n"],
                changed=diff["different_source"]["n"],
                kept=diff["same_source"]["n"],
                delta=_fmt(diff["different_source"]["delta"]["median"]),
                small=diff["different_source"]["delta"]["le_0.05"],
                stable=diff["different_source"]["self_profile"]["gt_0.8"],
                same_self=_fmt(same),
            )
        )
    lines.extend(
        [
            "",
            "六组都是同一件事：档位不同的格子里，最大源格相同的只有 1 到 3 个。",
            "这几个格子里，相似度差不超过 0.05 的只有纽约周末的 1 个，其余都更大。",
            "其余格子换了最匹配的源格。它们的相似度差中位数在 0.18 到 0.23，自身短轮廓和长轮廓的相关中位数在 0.40 到 0.49。",
            "不是同一个源格只差一点点、刚好掉到相邻档。",
            "工作日里，档位变化主要是短窗口落在 0.5 到 0.8、长窗口高于 0.8。",
            "档位相同不等于轮廓没变。周末短窗口只有两天，档位相同的格子自身相关中位数可以低到 0.45。",
            "",
        ]
    )
    return lines


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 档位不同的格子差在哪里",
        "",
        "通道是出租上车。只看纽约→华盛顿、芝加哥→华盛顿。这一步没有训练。",
        "每个华盛顿格子先在源城有效格里找皮尔逊相关最大的 24 小时轮廓，再按不超过 0.5、大于 0.5 且不超过 0.8、大于 0.8 分档。",
        "短窗口和长窗口各得一个档。这里只解释两边都有定义、但档位不同的格子。",
        "",
        "最大源格相同，指两个窗口的最大相关源格里至少有一个相同。差在 1e-8 以内的并列最大都算。",
        "相似度差是两个窗口各自最大相关的绝对值差。不超过 0.02、0.05、0.10 这三档是累计人数，用来看有多少格子只是擦着切点掉档。这不是新的训练门槛。",
        "自身轮廓相关是同一个华盛顿格子的短轮廓和长轮廓之间的皮尔逊相关。高于 0.8 表示这条曲线自己仍然很像。",
        "",
        "三天工作日沿用原来的周一到周五，假日留在工作日里。短窗口是 [5832, 5904)。",
        "五日窗口和源城训练用同一套分类：工作日去掉前 5904 小时里的 5 个联邦假日，共 120 小时；周末是周六、周日加上这些假日。",
        "工作日短窗口仍是 [5832, 5904)，周末短窗口是 [5736, 5784)。这两段里没有联邦假日。",
        "长窗口都是前 5904 小时上的同一种分类，不把选中的小时拼成一条新序列。",
        "",
    ]
    current = None
    for item in report["comparisons"]:
        if item["setup"] != current:
            current = item["setup"]
            lines.extend(["", f"## {SETUP_TITLE[current]}", ""])
        summary = item["summary"]
        agreement = item["agreement"]
        lines.extend(
            [
                f"### {CITY[item['source']]} → 华盛顿，{KIND_TITLE[item['kind']]}",
                "",
                (
                    f"两边都有定义 {summary['n_both_defined']}，档位相同 {summary['n_same_bin']}，"
                    f"档位不同 {summary['n_diff_bin']}。"
                    f"一致比例 {_fmt_ratio(agreement['agreement'])}。"
                ),
                (
                    f"档位不同的格子里，有 {summary['n_shared_source_but_argmax_differs']} 个因为并列最大而和单独取第一个源格的结论不同。"
                    f"有 {summary['n_either_window_tied']} 个在至少一个窗口上存在并列最大。"
                ),
                "",
                "档位不同：",
                "",
                *_source_table(summary["different_bin"]),
                "",
                "档位相同，用来对照：",
                "",
                *_source_table(summary["same_bin"]),
                "",
                "短窗口档位（行）对长窗口档位（列）。人数含档位相同的格子。",
                "",
                "| 短 \\ 长 | 不超过 0.5 | 0.5 到 0.8 | 高于 0.8 |",
                "|---|---:|---:|---:|",
            ]
        )
        for row, label in enumerate(BIN_LABELS):
            counts = summary["transitions_short_by_long"][row]
            lines.append(f"| {label} | {counts[0]} | {counts[1]} | {counts[2]} |")
        lines.append("")
    lines.extend(_conclusion(report))
    lines.extend(
        [
            "重算：",
            "",
            "```bash",
            "python scripts/12_bin_disagreement.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(processed_dir: Path, out_dir: Path) -> dict[str, Any]:
    report = build_report(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "bin_disagreement.md"
    json_path = out_dir / "bin_disagreement.json"
    json_path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(f"wrote {md_path}")
    for item in report["comparisons"]:
        diff = item["summary"]["different_bin"]
        print(
            f"  {item['setup']} {item['kind']} {item['source']}->{item['target']} "
            f"diff={diff['n']} same_source={diff['same_source']['n']} "
            f"different_source={diff['different_source']['n']}"
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
        print(f"bin disagreement failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
