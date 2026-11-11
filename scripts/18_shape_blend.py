#!/usr/bin/env python3
"""Blend a fine-tune forecast with a scaled source-cell shape. No training.

For each Washington cell with at least 150 raw pickups in the 72-hour window,
average the long weekday profiles of source cells whose Pearson is within 0.02
of that cell's best and at least 0.8. Scale the average to the cell's own
three-day level. The forecast is ``(1 - w) * finetune + w * shape``. One ``w``
is chosen on the validation MAE. A source city whose validation MAE is not
strictly lower than fine-tune does not get a test score.
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
from src.utils.pearson_weekday_profile import (
    PROFILE_STD_EPS,
    TRAIN_PREFIX_HOURS,
    _centered_norm,
)

GATE = 150
MARGIN = 0.02
MIN_RHO = 0.8
WEIGHTS = tuple(float(f"{step:.2f}") for step in np.linspace(0.0, 1.0, 21))
PUBLISHED_ATOL = 1e-3
SOURCES = ("ny", "chi")
CITY = {"ny": "纽约", "chi": "芝加哥"}
CHECKPOINTS = {"ny": "finetune_ny2dc.pt", "chi": "finetune_chi2dc.pt"}


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scripts/{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def correlation_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pearson of every target row against every source row. Flat rows stay nan."""
    source_c, source_n = _centered_norm(source)
    target_c, target_n = _centered_norm(target)
    rho = np.full((target_c.shape[0], source_c.shape[0]), np.nan, dtype=np.float64)
    ok_source = source_n >= PROFILE_STD_EPS
    ok_target = target_n >= PROFILE_STD_EPS
    if np.any(ok_source) and np.any(ok_target):
        rho[np.ix_(ok_target, ok_source)] = (target_c[ok_target] / target_n[ok_target, None]) @ (
            source_c[ok_source] / source_n[ok_source, None]
        ).T
    return rho


def donor_mask(rho: np.ndarray, margin: float = MARGIN, min_rho: float = MIN_RHO) -> np.ndarray:
    """Source cells close to the best correlation, and only when that best is high."""
    values = np.asarray(rho, dtype=np.float64)
    finite = np.isfinite(values)
    chosen = np.zeros(values.shape, dtype=bool)
    if not np.any(finite):
        return chosen
    best = float(values[finite].max())
    if best < min_rho:
        return chosen
    chosen[finite & (values >= best - margin) & (values >= min_rho)] = True
    return chosen


def scaled_shape(profiles: np.ndarray, three_total: float) -> np.ndarray:
    """Mean profile scaled so its 24 hours sum to the cell's three-day daily level."""
    mean = np.asarray(profiles, dtype=np.float64).mean(axis=0)
    if mean.shape != (24,):
        raise ValueError(f"expected a 24-hour profile, got {mean.shape}")
    daily = float(mean.sum())
    if daily <= 0:
        raise ValueError("source profile has no pickups")
    return mean * (float(three_total) / 3.0 / daily)


def blend(prediction: np.ndarray, shape: np.ndarray, weight: float) -> np.ndarray:
    """Mix the forecast with the shape. NaN shape cells stay on the forecast."""
    pred = np.asarray(prediction, dtype=np.float64)
    prior = np.asarray(shape, dtype=np.float64)
    if pred.shape != prior.shape:
        raise ValueError("prediction and shape shapes differ")
    if not 0.0 <= float(weight) <= 1.0:
        raise ValueError("weight must be between 0 and 1")
    mixed = np.array(pred, copy=True)
    use = np.isfinite(prior)
    mixed[use] = (1.0 - float(weight)) * pred[use] + float(weight) * prior[use]
    return mixed


def _shape_grid(
    rho: np.ndarray,
    source_long: np.ndarray,
    three_total: np.ndarray,
    eligible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_cells = int(eligible.shape[0])
    shape = np.full((n_cells, 24), np.nan, dtype=np.float64)
    donors = np.zeros(n_cells, dtype=np.int64)
    for index in np.flatnonzero(eligible):
        chosen = donor_mask(rho[index])
        donors[index] = int(chosen.sum())
        if donors[index] == 0:
            continue
        shape[index] = scaled_shape(source_long[chosen], float(three_total[index]))
    return shape, donors


def _repeat_by_hour(shape: np.ndarray, start_hour: int, n_hours: int) -> np.ndarray:
    clock = (np.arange(n_hours, dtype=np.int64) + int(start_hour)) % 24
    grid = np.full((n_hours, shape.shape[0]), np.nan, dtype=np.float64)
    known = np.isfinite(shape).all(axis=1)
    if np.any(known):
        grid[:, known] = shape[known][:, clock].T
    return grid


def _best_weight(prediction: np.ndarray, target: np.ndarray, shape: np.ndarray) -> tuple[float, float, float]:
    base = float(np.abs(prediction - target).sum()) / float(prediction.size)
    best_weight = 0.0
    best_mae = base
    for weight in WEIGHTS:
        mixed = blend(prediction, shape, weight)
        mae = float(np.abs(mixed - target).sum()) / float(mixed.size)
        if mae < best_mae:
            best_weight = float(weight)
            best_mae = mae
    return best_weight, best_mae, base


def build_report(processed_dir: Path, checkpoint_dir: Path, device_name: str) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    affine = _load_module("cell_affine_shape", "17_cell_affine.py")
    route = _load_module("bin_route_shape", "13_bin_route.py")
    volume = _load_module("cell_volume_shape", "14_cell_volume.py")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    prepared = prepare_city_windows("dc", processed_dir, few_shot_days=3)
    raw = {city: load_city_array(city, processed_dir) for city in ("ny", "chi", "dc")}
    dc_valid = valid_region_mask(raw["dc"])
    if not np.array_equal(dc_valid, np.asarray(prepared["mask"])):
        raise ValueError("raw valid mask does not match the prepared windows")
    dc = volume._prepare(raw["dc"], dc_valid)
    three_total = dc["three_sum"].sum(axis=1)
    eligible = three_total >= GATE
    if int(eligible.sum()) != 50:
        raise ValueError(f"gated cells {int(eligible.sum())} != 50")
    sources = {city: volume._prepare(raw[city]) for city in SOURCES}

    value_scale = float(prepared["max_val"] - prepared["min_val"])
    min_val = float(prepared["min_val"])
    mask = torch.as_tensor(prepared["mask"])
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    if device_name == "cpu":
        torch.set_num_threads(4)
    device = torch.device(device_name)
    loaders = {
        split: DataLoader(prepared[split], batch_size=32, shuffle=False)
        for split in ("train", "val", "test")
    }
    model_cfg = cfg.get("model", {})
    cities = []
    for source in SOURCES:
        print(f"scoring {source}", flush=True)
        net = route._load_net(checkpoint_dir / CHECKPOINTS[source], mask, device, model_cfg)
        series = {
            split: affine._collect(net, loader, mask, device, value_scale, min_val)
            for split, loader in loaders.items()
        }
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for split in ("val", "test"):
            baseline = affine.score_arrays(series[split][0], series[split][1], value_scale)
            published_mae, published_rmse = route.PUBLISHED[f"{source}_finetune"][split]
            if (
                abs(baseline["mae"] - published_mae) > PUBLISHED_ATOL
                or abs(baseline["rmse"] - published_rmse) > PUBLISHED_ATOL
            ):
                raise ValueError(
                    f"{source} {split} fine-tune {baseline['mae']:.6f}/{baseline['rmse']:.6f} "
                    f"!= {published_mae:.4f}/{published_rmse:.4f}"
                )
        rho = correlation_matrix(sources[source]["long"], dc["short"])
        cell_shape, donor_counts = _shape_grid(rho, sources[source]["long"], three_total, eligible)
        val_hours = series["val"][0].shape[0]
        test_hours = series["test"][0].shape[0]
        val_shape = _repeat_by_hour(cell_shape, TRAIN_PREFIX_HOURS, val_hours)
        weight, val_mae, base_mae = _best_weight(series["val"][0], series["val"][1], val_shape)
        improved = val_mae < base_mae
        val_prediction = blend(series["val"][0], val_shape, weight if improved else 0.0)
        record = {
            "source": source,
            "n_with_shape": int(np.isfinite(cell_shape).all(axis=1).sum()),
            "donor_median": float(np.median(donor_counts[donor_counts > 0])) if np.any(donor_counts > 0) else None,
            "weight": weight if improved else 0.0,
            "validation_improved": improved,
            "splits": {
                "val": {
                    "finetune": affine.score_arrays(series["val"][0], series["val"][1], value_scale),
                    "blend": affine.score_arrays(val_prediction, series["val"][1], value_scale),
                }
            },
        }
        if improved:
            test_shape = _repeat_by_hour(cell_shape, TRAIN_PREFIX_HOURS + val_hours, test_hours)
            test_prediction = blend(series["test"][0], test_shape, weight)
            record["splits"]["test"] = {
                "finetune": affine.score_arrays(series["test"][0], series["test"][1], value_scale),
                "blend": affine.score_arrays(test_prediction, series["test"][1], value_scale),
            }
        cities.append(record)
        print(
            f"  {source} val {base_mae:.4f} -> {val_mae:.4f} weight {record['weight']:.2f} "
            f"test={'yes' if improved else 'no'}",
            flush=True,
        )
    return {
        "channel": "taxi_pickup",
        "seed": 42,
        "trained": False,
        "gate": GATE,
        "margin": MARGIN,
        "min_rho": MIN_RHO,
        "device": device_name,
        "cities": cities,
    }


def _fmt(value: float | None) -> str:
    if value is None or not np.isfinite(value):
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


def _method_table(split: dict[str, Any]) -> list[str]:
    lines = [
        "| 方法 | MAE | RMSE | MAPE |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (("finetune", "一律微调"), ("blend", "形状混合")):
        item = split[key]
        lines.append(f"| {label} | {_fmt(item['mae'])} | {_fmt(item['rmse'])} | {_fmt(item['mape'])} |")
    return lines


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 相似格子的形状混合",
        "",
        "通道是出租上车。种子 42。没有新训练。",
        "过门格子用三天轮廓和源城长期轮廓做 Pearson。",
        f"只保留相关不低于 {report['min_rho']}、并且离最高相关不超过 {report['margin']} 的源城格子。",
        "这些长期轮廓先平均，再缩放到该格三天的日均上车量。",
        "预报是 `(1 - w) * 微调 + w * 形状`。`w` 在 0 到 1 之间每隔 0.05 取一个，只按验证集全城 MAE 选取。",
        "验证集没有严格低于一律微调的方向，不计算测试集。",
        "门槛、0.02 和 0.8 在看验证集之前定下，之后没有改。",
        "",
    ]
    for city in report["cities"]:
        tested = "计算了测试集" if city["validation_improved"] else "没有计算测试集"
        lines.extend(
            [
                f"## {CITY[city['source']]} → 华盛顿",
                "",
                (
                    f"有形状的格子 {city['n_with_shape']} 个。"
                    f"这些格子的源城格子数中位数是 {_fmt(city['donor_median'])}。"
                ),
                f"验证集选中的权重是 {_fmt(city['weight'])}。{tested}。",
                "",
                "验证集：",
                "",
                *_method_table(city["splits"]["val"]),
                "",
            ]
        )
        if "test" in city["splits"]:
            lines.extend(["测试集：", "", *_method_table(city["splits"]["test"]), ""])
    lines.extend(["重算：", "", "```bash", "python scripts/18_shape_blend.py", "```", ""])
    return "\n".join(lines)


def write_report(processed_dir: Path, checkpoint_dir: Path, out_dir: Path, device_name: str) -> dict[str, Any]:
    report = build_report(processed_dir, checkpoint_dir, device_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "shape_blend.json"
    md_path = out_dir / "shape_blend.md"
    json_path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(f"wrote {md_path}")
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
        print(f"shape blend failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
