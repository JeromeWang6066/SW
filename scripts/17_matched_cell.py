#!/usr/bin/env python3
"""Per-cell transfer from the best-matching source cell. No spatial neighbors.

Only Washington cells with a 3-day pickup total of at least 150 and a weekday
profile correlation above 0.8 are included. A ridge model learns six-hour lag
shape from that source cell, after the source series is scaled to the
Washington cell's 3-day mean. Two affine coefficients then recalibrate the
level on the 72 Washington label hours. Validation is compared with the
existing whole-city fine-tune on the same cells. The test split is scored
only when that validation MAE is strictly lower.
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

from src.data.dataset import load_city_array, min_max_normalize, valid_region_mask
from src.utils.pearson_weekday_profile import max_source_match

LAG = 6
LABEL_START = 5832
TRAIN_END = 5904
VAL_END = 7344
SERIES_HOURS = 8784
GATE = 150
HIGH = 0.8
ALPHA = 1.0
EXPECTED_N = 46
VAL_HOURS = VAL_END - TRAIN_END
TEST_HOURS = SERIES_HOURS - VAL_END
SOURCES = ("ny", "chi")
CITY = {"ny": "纽约", "chi": "芝加哥"}
PUBLISHED_VAL_MAE = {"ny": 2.0548, "chi": 1.7130}


def eligible_mask(
    rho: np.ndarray,
    total: np.ndarray,
    gate: float = GATE,
    high: float = HIGH,
) -> np.ndarray:
    """Cells whose 3-day total and short-window correlation clear the fixed cuts."""
    values = np.asarray(rho, dtype=np.float64)
    counts = np.asarray(total, dtype=np.float64)
    if values.shape != counts.shape:
        raise ValueError(f"rho {values.shape} != total {counts.shape}")
    return np.isfinite(values) & (values > high) & (counts >= gate)


def scale_to_level(source: np.ndarray, dc_level: float, src_level: float) -> np.ndarray:
    """Multiply a source series so its mean matches the Washington 3-day mean."""
    if not np.isfinite(src_level) or src_level <= 0:
        raise ValueError("source level must be positive")
    if not np.isfinite(dc_level) or dc_level < 0:
        raise ValueError("Washington level must be non-negative")
    return np.asarray(source, dtype=np.float64) * (float(dc_level) / float(src_level))


def lag_windows(series: np.ndarray, start: int, end: int, lag: int = LAG) -> tuple[np.ndarray, np.ndarray]:
    """Targets ``series[start:end]`` with the ``lag`` preceding hours as columns."""
    values = np.asarray(series, dtype=np.float64)
    if start < lag or end > len(values) or end < start:
        raise ValueError(f"bad window [{start}, {end}) for length {len(values)} lag {lag}")
    idx = np.arange(start, end)
    lags = np.column_stack([values[idx - k] for k in range(lag, 0, -1)])
    return lags, values[idx]


def fit_shape(x: np.ndarray, y: np.ndarray, alpha: float = ALPHA) -> dict[str, Any]:
    """Ridge on six standardized lags. The penalty is fixed, not chosen on val."""
    features = np.asarray(x, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    if features.ndim != 2 or len(features) != len(target):
        raise ValueError("shape design and target differ")
    mu = features.mean(axis=0)
    sig = features.std(axis=0)
    sig = np.where(sig < 1e-8, 1.0, sig)
    y_mu = float(target.mean())
    y_sig = float(target.std())
    if y_sig < 1e-8:
        y_sig = 1.0
    xz = (features - mu) / sig
    yz = (target - y_mu) / y_sig
    gram = xz.T @ xz + float(alpha) * np.eye(features.shape[1])
    beta = np.linalg.solve(gram, xz.T @ yz)
    return {"beta": beta, "mu": mu, "sig": sig, "y_mu": y_mu, "y_sig": y_sig}


def shape_predict(x: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    features = np.asarray(x, dtype=np.float64)
    xz = (features - model["mu"]) / model["sig"]
    return xz @ model["beta"] * model["y_sig"] + model["y_mu"]


def fit_level(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Affine recalibration ``a + c * prediction`` on the 72 label hours."""
    prediction = np.asarray(pred, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    if len(prediction) != len(target) or len(target) == 0:
        raise ValueError("level design and target differ")
    design = np.column_stack([np.ones(len(prediction)), prediction])
    coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return np.asarray(coef, dtype=np.float64)


def apply_level(pred: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Apply the affine map and keep pickup forecasts non-negative."""
    prediction = np.asarray(pred, dtype=np.float64)
    return np.maximum(coef[0] + coef[1] * prediction, 0.0)


def opens_test(candidate: float, baseline: float) -> bool:
    """Test is scored only when validation MAE is strictly below fine-tune."""
    return float(candidate) < float(baseline)


def _load_volume():
    spec = importlib.util.spec_from_file_location("cell_volume", ROOT / "scripts" / "14_cell_volume.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load scripts/14_cell_volume.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clipped(flow: np.ndarray) -> np.ndarray:
    """Percentile-clipped counts, the same target unit as the ST-Net replay."""
    normalized, max_val, min_val = min_max_normalize(flow)
    return np.asarray(normalized, dtype=np.float64) * (max_val - min_val) + min_val


def _valid_series(flow: np.ndarray, valid: np.ndarray) -> np.ndarray:
    flat = np.asarray(flow, dtype=np.float64).reshape(flow.shape[0], -1)
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    if flat.shape[1] != mask.shape[0]:
        raise ValueError("mask does not match the grid")
    return flat[:, mask]


def _cache_ae(name: str, split: str) -> tuple[np.ndarray, float]:
    path = ROOT / "logs" / "bin_route_cache" / f"{name}_{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path.name}. Run python scripts/13_bin_route.py first."
        )
    blob = np.load(path)
    return np.asarray(blob["ae"], dtype=np.float64), float(blob["hours"])


def _subset_mae(ae: np.ndarray, hours: float, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool)
    n_cells = int(selected.sum())
    if n_cells <= 0 or hours <= 0:
        raise ValueError("empty subset")
    return float(ae[selected].sum()) / (float(hours) * n_cells)


def _fit_cell(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    src_level = float(np.mean(source[:LABEL_START]))
    dc_level = float(np.mean(target[LABEL_START:TRAIN_END]))
    scaled = scale_to_level(source[:LABEL_START], dc_level, src_level)
    src_x, src_y = lag_windows(scaled, LAG, LABEL_START)
    shape = fit_shape(src_x, src_y, ALPHA)
    dc_x, dc_y = lag_windows(target, LABEL_START, TRAIN_END)
    level = fit_level(shape_predict(dc_x, shape), dc_y)
    local = fit_shape(dc_x, dc_y, ALPHA)
    return {"shape": shape, "level": level, "local": local, "scale": dc_level / src_level}


def _predict_cell(model: dict[str, Any], series: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = lag_windows(series, start, end)
    transferred = apply_level(shape_predict(x, model["shape"]), model["level"])
    local = np.maximum(shape_predict(x, model["local"]), 0.0)
    return transferred, local, y


def _score(models: list[dict[str, Any]], series: np.ndarray, mask: np.ndarray, start: int, end: int) -> dict[str, Any]:
    rows = np.flatnonzero(mask)
    transferred = np.empty((len(rows), end - start), dtype=np.float64)
    local = np.empty_like(transferred)
    target = np.empty_like(transferred)
    for row, cell in enumerate(rows):
        transferred[row], local[row], target[row] = _predict_cell(models[row], series[:, cell], start, end)
    hours = float(end - start)
    return {
        "transferred_mae": float(np.abs(transferred - target).sum()) / (hours * len(rows)),
        "local_mae": float(np.abs(local - target).sum()) / (hours * len(rows)),
        "transferred_cell_mae": np.abs(transferred - target).mean(axis=1),
        "hours": hours,
    }


def build_report() -> dict[str, Any]:
    volume = _load_volume()
    raw = {city: load_city_array(city) for city in ("ny", "chi", "dc")}
    for city, flow in raw.items():
        if flow.shape[0] != SERIES_HOURS:
            raise ValueError(f"{city} T={flow.shape[0]} != {SERIES_HOURS}")
    clipped = {city: _clipped(flow) for city, flow in raw.items()}
    dc_valid = valid_region_mask(raw["dc"])
    dc_profile = volume._prepare(raw["dc"], dc_valid)
    dc_series = _valid_series(clipped["dc"], dc_valid)
    total = dc_profile["three_sum"].sum(axis=1)
    if dc_series.shape[1] != len(total):
        raise ValueError("clipped cells are not aligned with the profiles")

    cities = []
    for source in SOURCES:
        prepared = volume._prepare(raw[source])
        rho, _, argmax, _tied = max_source_match(prepared["long"], dc_profile["short"])
        mask = eligible_mask(rho, total)
        n_cells = int(mask.sum())
        if n_cells != EXPECTED_N:
            raise ValueError(f"{source} eligible cells {n_cells} != {EXPECTED_N}")
        ae, hours = _cache_ae(f"{source}_finetune", "val")
        if ae.shape != total.shape:
            raise ValueError(f"{source} cache cells {ae.shape} != {total.shape}")
        if hours != VAL_HOURS:
            raise ValueError(f"{source} val hours {hours} != {VAL_HOURS}")
        city_ae, _ = _cache_ae(f"{source}_finetune", "val")
        city_mae = float(city_ae.sum()) / (hours * len(city_ae))
        if abs(city_mae - PUBLISHED_VAL_MAE[source]) > 5e-4:
            raise ValueError(f"{source} val fine-tune MAE {city_mae:.6f} drifted")
        baseline = _subset_mae(ae, hours, mask)
        donor = argmax[mask]
        if np.any(donor < 0):
            raise ValueError(f"{source} has an eligible cell with no donor")
        source_series = _valid_series(clipped[source], valid_region_mask(raw[source]))
        models = [_fit_cell(source_series[:, int(index)], dc_series[:, cell]) for cell, index in zip(np.flatnonzero(mask), donor)]
        val = _score(models, dc_series, mask, TRAIN_END, VAL_END)
        opened = opens_test(val["transferred_mae"], baseline)
        test = None
        if opened:
            scored = _score(models, dc_series, mask, VAL_END, SERIES_HOURS)
            test_ae, test_hours = _cache_ae(f"{source}_finetune", "test")
            if test_hours != TEST_HOURS:
                raise ValueError(f"{source} test hours {test_hours} != {TEST_HOURS}")
            test = {
                "transferred_mae": scored["transferred_mae"],
                "local_mae": scored["local_mae"],
                "finetune_mae": _subset_mae(test_ae, test_hours, mask),
                "hours": scored["hours"],
            }
        cell_base = ae[mask] / hours
        cities.append(
            {
                "source": source,
                "n_cells": n_cells,
                "n_donors": int(np.unique(donor).size),
                "scale_median": float(np.median([model["scale"] for model in models])),
                "finetune_val_mae": baseline,
                "transferred_val_mae": val["transferred_mae"],
                "local_val_mae": val["local_mae"],
                "n_better_than_finetune": int(np.sum(val["transferred_cell_mae"] < cell_base)),
                "opens_test": opened,
                "test": test,
            }
        )
    return {
        "channel": "taxi_pickup",
        "seed": None,
        "trained_stnet": False,
        "gate": GATE,
        "high": HIGH,
        "alpha": ALPHA,
        "lag": LAG,
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


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 匹配格子的单格时间模型",
        "",
        "通道是出租上车。没有改整城 ST-Net，也没有看邻域。",
        "格子要同时满足两件事先定下的事：2016-08-31 至 09-02 上车合计不少于 150，并且和源城长期轮廓的最大相关高于 0.8。",
        "源城序列用的是该城 99.9 分位截断后的计数，小时范围是 [0, 5832)。",
        "它先乘上一个常数，使这段的均值等于华盛顿这一格在 [5832, 5904) 的均值。",
        "形状是 6 小时滞后的岭回归，惩罚系数 1，在标准化之后固定，不在验证集上选。",
        "水平是那 72 个华盛顿标签小时上的 `a + c * 形状预测`。预测小于 0 的记为 0。",
        "对照有两个：同一批格子上已经训好的整城微调，以及只用这 72 小时、不用源城格子的岭回归。",
        "验证集平均 MAE 严格低于整城微调，才计算测试集。两条源城分开判断。",
        "",
        "| 源城 | 格子 | 不同的源城格子 | 缩放倍数中位数 | 整城微调 | 匹配格子 | 只用三天 | 低于微调的格子 | 测试集 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for city in report["cities"]:
        if city["opens_test"]:
            test = (
                f"匹配 {_fmt(city['test']['transferred_mae'])}，"
                f"微调 {_fmt(city['test']['finetune_mae'])}，"
                f"三天 {_fmt(city['test']['local_mae'])}"
            )
        else:
            test = "未计算"
        lines.append(
            f"| {CITY[city['source']]} | {city['n_cells']} | {city['n_donors']} | {_fmt(city['scale_median'])} | "
            f"{_fmt(city['finetune_val_mae'])} | {_fmt(city['transferred_val_mae'])} | {_fmt(city['local_val_mae'])} | "
            f"{city['n_better_than_finetune']} | {test} |"
        )
    lines.extend(
        [
            "",
            "整城微调、匹配格子、只用三天这三列是验证集。测试集只在匹配格子的验证误差严格低于整城微调时计算。",
            "",
            "重算：",
            "",
            "```bash",
            "python scripts/17_matched_cell.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(out_dir: Path) -> dict[str, Any]:
    report = build_report()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "matched_cell.json"
    md_path = out_dir / "matched_cell.md"
    json_path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(f"wrote {md_path}")
    for city in report["cities"]:
        print(
            f"  {city['source']} val transfer {_fmt(city['transferred_val_mae'])} "
            f"finetune {_fmt(city['finetune_val_mae'])} local {_fmt(city['local_val_mae'])} "
            f"test={'open' if city['opens_test'] else 'closed'}"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "profiles")
    args = parser.parse_args()
    try:
        write_report(args.out_dir)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"matched cell failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
