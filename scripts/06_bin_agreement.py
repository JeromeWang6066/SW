#!/usr/bin/env python3
"""Compare 72-hour and 5904-hour similarity bins. No training.

For New York to Washington and Chicago to Washington, each Washington cell
gets a max weekday-profile Pearson against the source. The protocol target
profile uses the last 72 training labels. The control uses the 5904-hour
train prefix. Agreement is the share of cells defined both ways that land
in the same bin.
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
    BIN_AGREEMENT_GATE,
    BIN_HIGH,
    BIN_LOW,
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
BIN_NAMES = {
    0: "le_0.5",
    1: "gt_0.5_le_0.8",
    2: "gt_0.8",
}


def _profile(flow: np.ndarray, select: np.ndarray) -> np.ndarray:
    profile, _counts = grid_hour_profile(flow, select)
    return profile


def _counts(bins: np.ndarray) -> dict[str, int]:
    defined = bins[bins >= 0]
    return {
        "n_defined": int(defined.size),
        "n_undefined": int((bins < 0).sum()),
        **{BIN_NAMES[code]: int((defined == code).sum()) for code in BIN_NAMES},
    }


def _pair(
    source_rows: np.ndarray,
    target_short: np.ndarray,
    target_long: np.ndarray,
) -> dict[str, Any]:
    short_rho, _short_defined = max_source_pearson(source_rows, target_short)
    long_rho, _long_defined = max_source_pearson(source_rows, target_long)
    short_bins = assign_similarity_bin(short_rho)
    long_bins = assign_similarity_bin(long_rho)
    stats = bin_agreement(short_bins, long_bins)
    medium = (short_bins == 1).astype(int)
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
        "protocol_bins": short_bins,
        "medium_mask": medium,
    }


def build_agreement(processed_dir: Path) -> dict[str, Any]:
    few_start, few_end = few_shot_label_bounds()
    _, weekday = hourly_calendar(EXPECTED_SERIES_HOURS)
    flows = {}
    profiles = {}
    raw = {}
    for city in ("ny", "chi", "dc"):
        flow = load_city_array(city, processed_dir)
        if flow.shape[0] != EXPECTED_SERIES_HOURS:
            raise ValueError(f"{city} T={flow.shape[0]} != {EXPECTED_SERIES_HOURS}")
        valid = valid_region_mask(flow)
        long_profile = _profile(flow[:TRAIN_PREFIX_HOURS], weekday[:TRAIN_PREFIX_HOURS])
        raw[city] = flow
        flows[city] = {"valid": valid, "n_valid": int(valid.sum())}
        profiles[city] = {"long": long_profile[valid]}
    dc_short = _profile(raw["dc"][few_start:few_end], weekday[few_start:few_end])
    profiles["dc"]["short"] = dc_short[flows["dc"]["valid"]]
    pairs = []
    for source, target in PAIRS:
        item = _pair(profiles[source]["long"], profiles[target]["short"], profiles[target]["long"])
        item["source"] = source
        item["target"] = target
        item["n_target_valid"] = flows[target]["n_valid"]
        pairs.append(item)
    return {
        "channel": "taxi_pickup",
        "few_shot_label_start": few_start,
        "few_shot_label_end": few_end,
        "bin_low": BIN_LOW,
        "bin_high": BIN_HIGH,
        "gate": BIN_AGREEMENT_GATE,
        "pairs": pairs,
    }


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


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "无定义"
    return f"{value:.4f}"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 格子分档是否一致",
        "",
        "只看纽约→华盛顿、芝加哥→华盛顿。源城用前 5904 小时的工作日轮廓。",
        "协议内的华盛顿轮廓是最后 72 个训练标签，"
        f"下标 [{report['few_shot_label_start']}, {report['few_shot_label_end']})。",
        "对照把华盛顿也换成前 5904 小时。",
        f"分档是不超过 {report['bin_low']}、大于 {report['bin_low']} 且不超过 {report['bin_high']}、大于 {report['bin_high']}。",
        "一致比例的分母是两次都有定义的格子。只在一次里有定义的格子不进分母。",
        f"门槛是 {report['gate']}。达到门槛的方向才可以对「有一点像」的格子做蒸馏。",
        "",
    ]
    for item in report["pairs"]:
        protocol = item["protocol_counts"]
        control = item["control_counts"]
        passed = "达到门槛" if item["passes_gate"] else "未达到门槛"
        lines.extend(
            [
                f"## {item['source']} → {item['target']}",
                "",
                f"- 一致比例 {_fmt_ratio(item['agreement'])}，{passed}。",
                f"- 两次都有定义 {item['n_both_defined']}，其中档位相同 {item['n_same']}。",
                f"- 只在 72 小时有定义 {item['n_only_protocol']}，只在 5904 小时有定义 {item['n_only_control']}。",
                "- 72 小时分档人数："
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
    lines.extend(
        [
            "重算：",
            "",
            "```bash",
            "python scripts/06_bin_agreement.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_agreement(processed_dir: Path, out_dir: Path) -> dict[str, Any]:
    report = build_agreement(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "bin_agreement.json"
    md_path = out_dir / "bin_agreement.md"
    json_path.write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(f"  agreement -> {md_path.relative_to(ROOT)}")
    for item in report["pairs"]:
        print(
            f"  {item['source']}->{item['target']} "
            f"agreement={_fmt_ratio(item['agreement'])} pass={item['passes_gate']}"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "profiles")
    args = parser.parse_args()
    try:
        write_agreement(args.processed_dir, args.out_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"bin agreement failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
