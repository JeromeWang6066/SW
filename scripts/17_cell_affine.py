#!/usr/bin/env python3
"""Per-cell scale and shift on an existing fine-tune. No new training.

For each Washington cell with at least 150 raw pickups in the 72-hour label
window, fit ``p' = a * p + b`` on that window's denormalized fine-tune
predictions. The ridge pulls ``a`` toward 1 and ``b`` toward 0 with lambda
equal to the number of hours. Every other cell keeps ``a = 1`` and ``b = 0``.
Lambda and the gate are fixed before validation or test are scored.
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
from src.utils.config import load_config

GATE = 150
LAM = 72.0
PUBLISHED_ATOL = 1e-3
SOURCES = ("ny", "chi")
SPLITS = ("train", "val", "test")
CITY = {"ny": "纽约", "chi": "芝加哥"}
CHECKPOINTS = {"ny": "finetune_ny2dc.pt", "chi": "finetune_chi2dc.pt"}


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scripts/{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def solve_ab(prediction: np.ndarray, target: np.ndarray, lam: float) -> tuple[float, float]:
    """Ridge ``target ~ a * prediction + b`` toward ``a = 1``, ``b = 0``."""
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred.shape != truth.shape or pred.size == 0:
        raise ValueError("prediction and target must be the same non-empty length")
    if lam < 0:
        raise ValueError("lambda must be non-negative")
    gram = np.array(
        [
            [float(np.dot(pred, pred)) + lam, float(pred.sum())],
            [float(pred.sum()), float(pred.size) + lam],
        ],
        dtype=np.float64,
    )
    rhs = np.array(
        [float(np.dot(pred, truth)) + lam, float(truth.sum())],
        dtype=np.float64,
    )
    scale, shift = np.linalg.solve(gram, rhs)
    return float(scale), float(shift)


def fit_affine(
    prediction: np.ndarray,
    target: np.ndarray,
    eligible: np.ndarray,
    lam: float = LAM,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one line per eligible cell. Other cells stay at ``a = 1``, ``b = 0``."""
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    keep = np.asarray(eligible, dtype=bool)
    if pred.ndim != 2 or pred.shape != truth.shape or keep.shape != (pred.shape[1],):
        raise ValueError("expected prediction/target (hours, cells) and one flag per cell")
    scale = np.ones(pred.shape[1], dtype=np.float64)
    shift = np.zeros(pred.shape[1], dtype=np.float64)
    for index in np.flatnonzero(keep):
        scale[index], shift[index] = solve_ab(pred[:, index], truth[:, index], lam)
    return scale, shift


def apply_affine(prediction: np.ndarray, scale: np.ndarray, shift: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float64)
    if pred.ndim != 2 or scale.shape != (pred.shape[1],) or shift.shape != scale.shape:
        raise ValueError("scale and shift must match the cell axis")
    return pred * scale.reshape(1, -1) + shift.reshape(1, -1)


def score_arrays(prediction: np.ndarray, target: np.ndarray, value_scale: float) -> dict[str, float]:
    """MAE, denormalized RMSE, and MAPE. Same reduction as training evaluation."""
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError("prediction and target shapes differ")
    if value_scale <= 0:
        raise ValueError("value scale must be positive")
    hours, n_cells = pred.shape
    valid_points = float(hours * n_cells)
    if valid_points <= 0:
        raise ValueError("no points to score")
    gap = pred - truth
    positive = truth > 0
    mape_points = float(positive.sum())
    mape = float("nan") if mape_points <= 0 else float(np.abs(gap[positive] / truth[positive]).sum()) / mape_points * 100.0
    se_norm = float(np.sum((gap / float(value_scale)) ** 2))
    return {
        "mae": float(np.abs(gap).sum()) / valid_points,
        "rmse": (se_norm / valid_points) ** 0.5 * float(value_scale),
        "mape": mape,
        "hours": float(hours),
        "n_cells": int(n_cells),
    }


def _collect(net, loader, mask, device, value_scale: float, min_val: float) -> tuple[np.ndarray, np.ndarray]:
    import torch

    preds = []
    trues = []
    valid = mask.reshape(-1).bool()
    for inputs, target in loader:
        inputs = inputs.to(device)
        target = target.to(device)
        out = net(inputs, spatial_mask=mask)
        y_valid = target.reshape(target.shape[0], 1, -1)[:, :, valid]
        preds.append((out * value_scale + min_val).detach().double().cpu().numpy()[:, 0, :])
        trues.append((y_valid * value_scale + min_val).detach().double().cpu().numpy()[:, 0, :])
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def _subset_mae(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool)
    if int(selected.sum()) <= 0:
        raise ValueError("subset is empty")
    gap = prediction[:, selected] - target[:, selected]
    return float(np.abs(gap).sum()) / float(gap.size)


def build_report(processed_dir: Path, checkpoint_dir: Path, device_name: str) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    route = _load_module("bin_route_affine", "13_bin_route.py")
    volume = _load_module("cell_volume_affine", "14_cell_volume.py")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    prepared = prepare_city_windows("dc", processed_dir, few_shot_days=3)
    raw = load_city_array("dc", processed_dir)
    raw_valid = valid_region_mask(raw)
    if not np.array_equal(raw_valid, np.asarray(prepared["mask"])):
        raise ValueError("raw valid mask does not match the prepared windows")
    three_total = volume._prepare(raw, raw_valid)["three_sum"].sum(axis=1)
    eligible = three_total >= GATE
    if int(eligible.sum()) != 50:
        raise ValueError(f"cells with raw three-day total >= {GATE}: {int(eligible.sum())} != 50")

    value_scale = float(prepared["max_val"] - prepared["min_val"])
    min_val = float(prepared["min_val"])
    mask = torch.as_tensor(prepared["mask"])
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    if device_name == "cpu":
        torch.set_num_threads(4)
    device = torch.device(device_name)
    loaders = {
        "train": DataLoader(prepared["train"], batch_size=32, shuffle=False),
        "val": DataLoader(prepared["val"], batch_size=32, shuffle=False),
        "test": DataLoader(prepared["test"], batch_size=32, shuffle=False),
    }
    model_cfg = cfg.get("model", {})
    cities = []
    for source in SOURCES:
        path = checkpoint_dir / CHECKPOINTS[source]
        print(f"scoring {source}", flush=True)
        net = route._load_net(path, mask, device, model_cfg)
        series = {}
        for split, loader in loaders.items():
            prediction, target = _collect(net, loader, mask, device, value_scale, min_val)
            if prediction.shape[1] != int(eligible.shape[0]):
                raise ValueError(f"{source} {split} cells {prediction.shape[1]} != {eligible.shape[0]}")
            series[split] = {"prediction": prediction, "target": target}
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if series["train"]["prediction"].shape[0] != int(LAM):
            raise ValueError(f"{source} train hours {series['train']['prediction'].shape[0]} != {int(LAM)}")
        for split in ("val", "test"):
            baseline = score_arrays(series[split]["prediction"], series[split]["target"], value_scale)
            published_mae, published_rmse = route.PUBLISHED[f"{source}_finetune"][split]
            if abs(baseline["mae"] - published_mae) > PUBLISHED_ATOL or abs(baseline["rmse"] - published_rmse) > PUBLISHED_ATOL:
                raise ValueError(
                    f"{source} {split} fine-tune {baseline['mae']:.6f}/{baseline['rmse']:.6f} "
                    f"!= {published_mae:.4f}/{published_rmse:.4f}"
                )
        scale, shift = fit_affine(
            series["train"]["prediction"],
            series["train"]["target"],
            eligible,
            LAM,
        )
        window_total = series["train"]["target"].sum(axis=0)
        splits = {}
        for split in SPLITS:
            base_pred = series[split]["prediction"]
            target = series[split]["target"]
            corrected = apply_affine(base_pred, scale, shift)
            splits[split] = {
                "finetune": score_arrays(base_pred, target, value_scale),
                "affine": score_arrays(corrected, target, value_scale),
                "subsets": {
                    "gate": {
                        "n": int(eligible.sum()),
                        "finetune_mae": _subset_mae(base_pred, target, eligible),
                        "affine_mae": _subset_mae(corrected, target, eligible),
                    },
                    "below": {
                        "n": int((~eligible).sum()),
                        "finetune_mae": _subset_mae(base_pred, target, ~eligible),
                        "affine_mae": _subset_mae(corrected, target, ~eligible),
                    },
                },
            }
        gated = eligible
        cities.append(
            {
                "source": source,
                "n_gate": int(gated.sum()),
                "scale_median": float(np.median(scale[gated])),
                "shift_median": float(np.median(shift[gated])),
                "scale_p10": float(np.quantile(scale[gated], 0.1)),
                "scale_p90": float(np.quantile(scale[gated], 0.9)),
                "shift_p10": float(np.quantile(shift[gated], 0.1)),
                "shift_p90": float(np.quantile(shift[gated], 0.9)),
                "gate_disagrees_with_model_target": int((eligible != (window_total >= GATE)).sum()),
                "splits": splits,
            }
        )
    return {
        "channel": "taxi_pickup",
        "seed": 42,
        "trained": False,
        "gate": GATE,
        "lambda": LAM,
        "device": device_name,
        "n_cells": int(eligible.shape[0]),
        "n_gate": int(eligible.sum()),
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


def _method_table(split: dict[str, Any]) -> list[str]:
    lines = [
        "| 方法 | MAE | RMSE | MAPE |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (("finetune", "一律微调"), ("affine", "按格 a、b")):
        item = split[key]
        lines.append(f"| {label} | {_fmt(item['mae'])} | {_fmt(item['rmse'])} | {_fmt(item['mape'])} |")
    return lines


def _subset_table(split: dict[str, Any]) -> list[str]:
    lines = [
        "| 子集 | 格子 | 一律微调 MAE | 按格 a、b MAE |",
        "|---|---:|---:|---:|",
    ]
    labels = {"gate": "三天不少于 150", "below": "三天少于 150"}
    for key, title in labels.items():
        item = split["subsets"][key]
        lines.append(f"| {title} | {item['n']} | {_fmt(item['finetune_mae'])} | {_fmt(item['affine_mae'])} |")
    return lines


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 按格水平和位移",
        "",
        "通道是出租上车。种子 42。没有新训练。",
        "起点是已经训好的整城微调。每个过门格子在三天窗口上拟合 `p' = a * p + b`。",
        f"岭回归的 λ 是 {report['lambda']:.0f}，把 `a` 拉向 1、把 `b` 拉向 0。",
        f"三天原始上车合计少于 {report['gate']} 的格子固定 `a = 1`、`b = 0`。",
        "λ 和 150 都在看验证集、测试集之前定下，之后没有改。",
        "校正用的预测和标签是反归一化后的上车量。门槛用原始计数。",
        "",
        f"有效格 {report['n_cells']} 个，过门 {report['n_gate']} 个。",
        "未校正的验证集和测试集 MAE、RMSE 与已发表的微调结果相差不超过 0.001，才写这张表。",
        "",
    ]
    for city in report["cities"]:
        lines.extend(
            [
                f"## {CITY[city['source']]} → 华盛顿",
                "",
                (
                    f"过门格子的 `a` 中位数 {_fmt(city['scale_median'])}，"
                    f"10 到 90 分位 {_fmt(city['scale_p10'])} 到 {_fmt(city['scale_p90'])}。"
                ),
                (
                    f"`b` 中位数 {_fmt(city['shift_median'])}，"
                    f"10 到 90 分位 {_fmt(city['shift_p10'])} 到 {_fmt(city['shift_p90'])}。"
                ),
                f"原始门槛和模型标签门槛不一致的格子有 {city['gate_disagrees_with_model_target']} 个。",
                "",
                "三天训练窗：",
                "",
                *_method_table(city["splits"]["train"]),
                "",
                "验证集：",
                "",
                *_method_table(city["splits"]["val"]),
                "",
                *_subset_table(city["splits"]["val"]),
                "",
                "测试集：",
                "",
                *_method_table(city["splits"]["test"]),
                "",
                *_subset_table(city["splits"]["test"]),
                "",
            ]
        )
    lines.extend(
        [
            "重算：",
            "",
            "```bash",
            "python scripts/17_cell_affine.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(processed_dir: Path, checkpoint_dir: Path, out_dir: Path, device_name: str) -> dict[str, Any]:
    report = build_report(processed_dir, checkpoint_dir, device_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "cell_affine.json"
    md_path = out_dir / "cell_affine.md"
    json_path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(f"wrote {md_path}")
    for city in report["cities"]:
        for split in ("val", "test"):
            base = city["splits"][split]["finetune"]["mae"]
            done = city["splits"][split]["affine"]["mae"]
            print(f"  {city['source']} {split} finetune {_fmt(base)} affine {_fmt(done)}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "profiles")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    try:
        write_report(args.processed_dir, args.checkpoint_dir, args.out_dir, args.device)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"cell affine failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
