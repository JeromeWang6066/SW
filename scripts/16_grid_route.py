#!/usr/bin/env python3
"""Replay seed-42 Washington cells under two grid rules. No training.

The forecast stays the cached 1 km output. One rule bins a cell only when its
own three-weekday pickup total is at least 150. The other bins every cell from
a Washington 3x3 sum, still matched to the unsmoothed source profile. Cuts
0.5, 0.8, and 150 are fixed. A mismatch with the published replay stops
before any report is written.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import load_city_array, prepare_city_windows, valid_region_mask
from src.utils.pearson_weekday_profile import (
    assign_similarity_bin,
    bin_agreement,
    max_source_match,
)

GATE = 150
GATE_TEST_MAE = 1.4737
GATE_TEST_ATOL = 5e-4
BOX_SAME = {"ny": 150, "chi": 169}
BOX_BOTH = 207
ROUTE_MAE = {
    "ny": {"val": "2.0483", "test": "1.7353"},
    "chi": {"val": "1.8637", "test": "1.6309"},
}
RULES = ("finetune", "gate", "box")
SPLITS = ("val", "test")
SOURCES = ("ny", "chi")
MODELS = ("scratch", "ny_finetune", "chi_finetune", "ny_kd", "chi_kd")
CITY = {"ny": "纽约", "chi": "芝加哥"}
RULE_LABEL = {
    "finetune": "一律微调",
    "gate": "三天不少于 150 才按档",
    "box": "华盛顿 3×3 轮廓按档",
}
MODEL_LABEL = {"finetune": "微调", "kd": "蒸馏", "scratch": "从零"}


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scripts/{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def passes_gate(three_total: np.ndarray, gate: float = GATE) -> np.ndarray:
    """True where the protocol three-day pickup total reaches ``gate``."""
    return np.asarray(three_total, dtype=np.float64) >= float(gate)


def choices_from_bins(bins: np.ndarray, eligible: np.ndarray | None = None) -> np.ndarray:
    """Map bins to models. Cells outside ``eligible`` stay on fine-tune."""
    route = _load_module("bin_route_choices", "13_bin_route.py")
    choices = np.array(route.route_choices(np.asarray(bins)), copy=True)
    if eligible is not None:
        keep = np.asarray(eligible, dtype=bool)
        if keep.shape != choices.shape:
            raise ValueError(f"eligible {keep.shape} != bins {choices.shape}")
        choices[~keep] = "finetune"
    return choices


def rule_choices(
    bins_1km: np.ndarray,
    bins_box: np.ndarray,
    three_total: np.ndarray,
    gate: float = GATE,
) -> dict[str, np.ndarray]:
    """Three fixed rules. The 3x3 rule does not look at the volume gate."""
    bins_1km = np.asarray(bins_1km)
    bins_box = np.asarray(bins_box)
    three_total = np.asarray(three_total, dtype=np.float64)
    if bins_1km.shape != bins_box.shape or bins_1km.shape != three_total.shape:
        raise ValueError("1 km bins, 3x3 bins, and three-day totals differ in shape")
    eligible = passes_gate(three_total, gate)
    return {
        "finetune": np.full(bins_1km.shape, "finetune", dtype=object),
        "gate": choices_from_bins(bins_1km, eligible),
        "box": choices_from_bins(bins_box, None),
    }


def _choice_counts(choices: np.ndarray) -> dict[str, int]:
    return {name: int((choices == name).sum()) for name in ("finetune", "kd", "scratch")}


def _bin_counts(bins: np.ndarray) -> dict[str, int]:
    values = np.asarray(bins)
    defined = values[values >= 0]
    return {
        "le_0.5": int((defined == 0).sum()),
        "gt_0.5_le_0.8": int((defined == 1).sum()),
        "gt_0.8": int((defined == 2).sum()),
        "n_undefined": int((values < 0).sum()),
    }


def _require_cache(route) -> dict[str, dict[str, dict[str, Any]]]:
    scored: dict[str, dict[str, dict[str, Any]]] = {}
    missing = []
    for name in MODELS:
        scored[name] = {}
        for split in SPLITS:
            loaded = route._load_cached(name, split)
            if loaded is None:
                missing.append(f"{name}_{split}.npz")
            else:
                scored[name][split] = loaded
    if missing:
        raise FileNotFoundError(
            "bin-route cache missing: "
            + ", ".join(missing)
            + ". Run python scripts/13_bin_route.py first. This script does not load checkpoints."
        )
    return scored


def _profiles(volume, processed_dir: Path) -> dict[str, Any]:
    flows = {city: load_city_array(city, processed_dir) for city in ("ny", "chi", "dc")}
    dc_valid = valid_region_mask(flows["dc"])
    base = volume._prepare(flows["dc"], dc_valid)
    boxed = volume._prepare(volume.box_sum(flows["dc"], 3), dc_valid)
    if base["short"].shape != boxed["short"].shape:
        raise ValueError("3x3 profiles do not keep the Washington valid cells")
    sources = {city: volume._prepare(flows[city])["long"] for city in ("ny", "chi")}
    three_total = base["three_sum"].sum(axis=1)
    if three_total.shape[0] != int(dc_valid.sum()):
        raise ValueError("three-day totals are not aligned with the valid mask")
    return {
        "valid": dc_valid,
        "sources": sources,
        "short": base["short"],
        "long": base["long"],
        "box_short": boxed["short"],
        "box_long": boxed["long"],
        "three_total": three_total,
    }


def _matched_bins(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    rho, _, _, _ = max_source_match(source, target)
    return assign_similarity_bin(rho)


def _slice_metrics(route, stitched: dict[str, Any], mask: np.ndarray, scale: float) -> dict[str, float]:
    selected = np.asarray(mask, dtype=bool)
    if int(selected.sum()) <= 0:
        raise ValueError("subset is empty")
    return route.metrics_from_cells(
        stitched["se"][selected],
        stitched["ae"][selected],
        stitched["ape"][selected],
        stitched["mape_points"][selected],
        stitched["hours"],
        scale,
    )


def _check_models(route, scored: dict[str, dict[str, dict[str, Any]]], scale: float, n_cells: int) -> None:
    for name in MODELS:
        for split in SPLITS:
            totals = scored[name][split]
            if totals["ae"].shape != (n_cells,):
                raise ValueError(f"{name} {split} cells {totals['ae'].shape} != {(n_cells,)}")
            replay = route._scalar_metrics(totals["scalar"], scale)
            route._check_published(name, split, replay)
            cell_metrics = route.metrics_from_cells(
                totals["se"], totals["ae"], totals["ape"], totals["mape_points"], totals["hours"], scale
            )
            if (
                abs(cell_metrics["mae"] - replay["mae"]) > route.CELL_ATOL
                or abs(cell_metrics["rmse"] - replay["rmse"]) > route.CELL_ATOL
            ):
                raise ValueError(
                    f"{name} {split} per-cell {cell_metrics['mae']:.6f}/{cell_metrics['rmse']:.6f} "
                    f"vs replay {replay['mae']:.6f}/{replay['rmse']:.6f}"
                )
            totals["metrics"] = cell_metrics


def _check_old_route(
    route,
    scored: dict[str, dict[str, dict[str, Any]]],
    bins: dict[str, np.ndarray],
    scale: float,
) -> None:
    for source, protocol in bins.items():
        counts = _bin_counts(protocol)
        expected = route.BIN_COUNTS[source]
        for key in ("le_0.5", "gt_0.5_le_0.8", "gt_0.8", "n_undefined"):
            if counts[key] != expected[key]:
                raise ValueError(f"{source} 1 km {key}={counts[key]} != {expected[key]}")
        choices = choices_from_bins(protocol, None)
        for split in SPLITS:
            bundle = {
                "scratch": scored["scratch"][split],
                "finetune": scored[f"{source}_finetune"][split],
                "kd": scored[f"{source}_kd"][split],
            }
            routed = route.stitch_cells(bundle, choices)
            metrics = route.metrics_from_cells(
                routed["se"], routed["ae"], routed["ape"], routed["mape_points"], routed["hours"], scale
            )
            got = f"{metrics['mae']:.4f}"
            if got != ROUTE_MAE[source][split]:
                raise ValueError(f"{source} {split} ungated route MAE {got} != {ROUTE_MAE[source][split]}")


def _check_box(profiles: dict[str, Any]) -> dict[str, dict[str, int]]:
    checked = {}
    for source, long_profile in profiles["sources"].items():
        short_bins = _matched_bins(long_profile, profiles["box_short"])
        long_bins = _matched_bins(long_profile, profiles["box_long"])
        stats = bin_agreement(short_bins, long_bins)
        if stats["n_same"] != BOX_SAME[source] or stats["n_both_defined"] != BOX_BOTH:
            raise ValueError(
                f"{source} 3x3 agreement {stats['n_same']}/{stats['n_both_defined']} "
                f"!= {BOX_SAME[source]}/{BOX_BOTH}"
            )
        checked[source] = {"n_same": int(stats["n_same"]), "n_both": int(stats["n_both_defined"])}
    return checked


def build_report(processed_dir: Path) -> dict[str, Any]:
    route = _load_module("bin_route", "13_bin_route.py")
    volume = _load_module("cell_volume", "14_cell_volume.py")
    scored = _require_cache(route)
    prepared = prepare_city_windows("dc", processed_dir, few_shot_days=3)
    scale = float(prepared["max_val"] - prepared["min_val"])
    profiles = _profiles(volume, processed_dir)
    n_cells = int(profiles["valid"].sum())
    _check_models(route, scored, scale, n_cells)

    bins_1km = {
        source: _matched_bins(long_profile, profiles["short"])
        for source, long_profile in profiles["sources"].items()
    }
    bins_box = {
        source: _matched_bins(long_profile, profiles["box_short"])
        for source, long_profile in profiles["sources"].items()
    }
    _check_old_route(route, scored, bins_1km, scale)
    gate = passes_gate(profiles["three_total"])
    n_gate = int(gate.sum())
    if n_gate != 50:
        raise ValueError(f"cells with three-day total >= {GATE}: {n_gate} != 50")
    box_check = _check_box(profiles)

    cities = []
    for source in SOURCES:
        choices = rule_choices(bins_1km[source], bins_box[source], profiles["three_total"])
        splits = {}
        for split in SPLITS:
            bundle = {
                "scratch": scored["scratch"][split],
                "finetune": scored[f"{source}_finetune"][split],
                "kd": scored[f"{source}_kd"][split],
            }
            stitched = {name: route.stitch_cells(bundle, picked) for name, picked in choices.items()}
            methods = {}
            for name in RULES:
                metrics = route.metrics_from_cells(
                    stitched[name]["se"],
                    stitched[name]["ae"],
                    stitched[name]["ape"],
                    stitched[name]["mape_points"],
                    stitched[name]["hours"],
                    scale,
                )
                methods[name] = {**metrics, **_choice_counts(choices[name])}
            subsets = {}
            for label, mask in (("gate", gate), ("below", ~gate)):
                subsets[label] = {
                    "n": int(mask.sum()),
                    "finetune_mae": _slice_metrics(route, bundle["finetune"], mask, scale)["mae"],
                    "rules": {name: _slice_metrics(route, stitched[name], mask, scale)["mae"] for name in RULES},
                }
            splits[split] = {"methods": methods, "subsets": subsets}
        cities.append({"source": source, "splits": splits})

    chi_gate = next(item for item in cities if item["source"] == "chi")["splits"]["test"]["methods"]["gate"]["mae"]
    if abs(chi_gate - GATE_TEST_MAE) > GATE_TEST_ATOL:
        raise ValueError(f"Chicago test gate MAE {chi_gate:.6f} != {GATE_TEST_MAE}")

    return {
        "channel": "taxi_pickup",
        "seed": 42,
        "trained": False,
        "gate": GATE,
        "n_cells": n_cells,
        "n_gate": n_gate,
        "box_agreement": box_check,
        "cities": cities,
    }


def _fmt(value: float) -> str:
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


def _method_table(methods: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| 规则 | MAE | RMSE | MAPE | 微调 | 蒸馏 | 从零 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in RULES:
        item = methods[key]
        lines.append(
            f"| {RULE_LABEL[key]} | {_fmt(item['mae'])} | {_fmt(item['rmse'])} | {_fmt(item['mape'])} | "
            f"{item['finetune']} | {item['kd']} | {item['scratch']} |"
        )
    return lines


def _subset_table(split: dict[str, Any]) -> list[str]:
    lines = [
        "| 子集 | 格子 | 规则 | MAE | 一律微调 MAE |",
        "|---|---:|---|---:|---:|",
    ]
    labels = {"gate": "三天不少于 150", "below": "三天少于 150"}
    for subset, title in labels.items():
        item = split["subsets"][subset]
        for key in RULES:
            lines.append(
                f"| {title} | {item['n']} | {RULE_LABEL[key]} | {_fmt(item['rules'][key])} | "
                f"{_fmt(item['finetune_mae'])} |"
            )
    return lines


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 网格分流重放",
        "",
        "通道是出租上车。种子 42。没有训练，也没有改权重。",
        "预测仍是原来的 1km 输出。0.5、0.8 和 150 都没有按验证集或测试集改。",
        "一律微调：每个有效格用该源城的纯微调。",
        "上车量门：2016-08-31 至 09-02 该格上车合计不少于 150，才按未平滑的 1km 短轮廓分档。",
        "没过门的格子用该源城微调。",
        "3×3：只把华盛顿短轮廓换成以该格为中心的 3×3 求和，再和源城未平滑的长期轮廓比。",
        "这一组不看 150 这道门。",
        "高于 0.8 用微调，高于 0.5 且不超过 0.8 用蒸馏，不超过 0.5 或无定义用从零。",
        "",
        "写这张表之前核对过已有数字。",
        "五个纯模型的验证集和测试集 MAE、RMSE 与 `scripts/13_bin_route.py` 的已发表值相差不超过 0.0001。",
        "不做门的 1km 拼接 MAE 是纽约验证 2.0483、测试 1.7353，芝加哥验证 1.8637、测试 1.6309。",
        f"三天不少于 {report['gate']} 的格子是 {report['n_gate']} 个。",
        (
            "只平滑华盛顿的 3×3，短轮廓和长轮廓同档的格子是纽约 "
            f"{report['box_agreement']['ny']['n_same']}/{report['box_agreement']['ny']['n_both']}，芝加哥 "
            f"{report['box_agreement']['chi']['n_same']}/{report['box_agreement']['chi']['n_both']}。"
        ),
        "芝加哥测试集上车量门的 MAE 与已经写过的 1.4737 相差不超过 0.0005。这个核对没有用来改 150。",
        "",
    ]
    for city in report["cities"]:
        lines.extend([f"## {CITY[city['source']]} → 华盛顿", "", "验证集：", ""])
        lines.extend(_method_table(city["splits"]["val"]["methods"]))
        lines.extend(["", "测试集：", ""])
        lines.extend(_method_table(city["splits"]["test"]["methods"]))
        lines.extend(["", "过门和没过门的格子。", "", "验证集：", ""])
        lines.extend(_subset_table(city["splits"]["val"]))
        lines.extend(["", "测试集：", ""])
        lines.extend(_subset_table(city["splits"]["test"]))
        lines.append("")
    lines.extend(
        [
            "重算：",
            "",
            "```bash",
            "python scripts/16_grid_route.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(processed_dir: Path, out_dir: Path) -> dict[str, Any]:
    report = build_report(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "grid_route.json"
    md_path = out_dir / "grid_route.md"
    json_path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(f"wrote {md_path}")
    for city in report["cities"]:
        for split in SPLITS:
            methods = city["splits"][split]["methods"]
            print(
                f"  {city['source']} {split} finetune {_fmt(methods['finetune']['mae'])} "
                f"gate {_fmt(methods['gate']['mae'])} box {_fmt(methods['box']['mae'])}"
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "profiles")
    args = parser.parse_args()
    try:
        write_report(args.processed_dir, args.out_dir)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"grid route failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
