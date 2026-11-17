#!/usr/bin/env python3
"""Centered per-cell affine versus the published lambda=72 line. No training.

The protocol is docs/CELL_AFFINE_V2_PLAN.md. Stages are split on purpose:
predict writes denormalized fine-tune caches, fit reads only the 72 training
hours and freezes kappa, score reads that freeze and reports val and test.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

N_HOURS = 72
DAY_HOURS = 24
GATE = 150
N_CELLS = 207
N_GATE = 50
LABEL_START = 5832
GRID = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, math.inf)
TIE_ATOL = 1e-9
BOOT_REPS = 2000
BOOT_SEED = 0
BASELINE_ATOL = 1e-3
LABEL_ATOL = 1e-3
O72_ATOL = 1e-5
SOURCES = ("ny", "chi")
SEEDS = (42, 43, 44)
CITY = {"ny": "纽约", "chi": "芝加哥"}
ARM_ORDER = ("B0", "O72", "Ostar", "S", "N1", "Nstar", "N0")
SECONDARY_PAIRS = (
    ("Nstar", "Ostar", "训练窗选出的新公式对训练窗选出的旧公式"),
    ("S", "O72", "只改尺度对 λ=72"),
    ("N1", "S", "κ=1 对只改尺度"),
    ("N0", "N1", "κ=0 对 κ=1"),
    ("N1", "B0", "κ=1 对一律微调"),
    ("O72", "B0", "λ=72 对一律微调"),
)
ARM_LABEL = {
    "B0": "一律微调",
    "O72": "旧公式 λ=72",
    "Ostar": "旧公式，训练窗选出的倍数",
    "S": "只改尺度",
    "N1": "新公式 κ=1",
    "Nstar": "新公式，训练窗选出的 κ",
    "N0": "新公式 κ=0",
}
PUBLISHED_VAL_42 = {
    "ny": (2.0548, 6.1937),
    "chi": (1.7130, 5.5809),
}
PUBLISHED_TEST = {
    ("ny", 42): (1.7153, 5.2786),
    ("ny", 43): (1.6344, 5.1229),
    ("ny", 44): (1.6242, 5.1752),
    ("chi", 42): (1.4773, 5.0578),
    ("chi", 43): (1.5319, 5.0045),
    ("chi", 44): (1.5422, 5.1079),
}
PUBLISHED_O72_MAE = {
    "ny": {"train": 1.4845197415876903, "val": 2.036014593704619, "test": 1.7054192513504478},
    "chi": {"train": 1.4103618944218177, "val": 1.6863351164205165, "test": 1.4415295244856239},
}
PUBLISHED_O72_MEDIAN = {"ny": 0.9889594710260354, "chi": 0.9390470088621887}
LOG_NAME = {
    ("ny", 43): "G1C-NY2DC_seed43.json",
    ("ny", 44): "G1C-NY2DC_seed44.json",
    ("chi", 43): "G1C-CHI2DC_seed43.json",
    ("chi", 44): "G1C-CHI2DC_seed44.json",
}


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scripts/{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _vec(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def solve_old(prediction: np.ndarray, target: np.ndarray, mult: float) -> tuple[float, float]:
    """Published ridge. ``mult`` scales lambda, and lambda = mult * n."""
    if math.isinf(mult):
        return 1.0, 0.0
    if not math.isfinite(mult) or mult < 0:
        raise ValueError("multiplier must be non-negative")
    pred = _vec(prediction)
    truth = _vec(target)
    if pred.shape != truth.shape or pred.size == 0:
        raise ValueError("prediction and target must be the same non-empty length")
    lam = float(mult) * float(pred.size)
    gram = np.array(
        [[float(pred @ pred) + lam, float(pred.sum())], [float(pred.sum()), float(pred.size) + lam]],
        dtype=np.float64,
    )
    rhs = np.array([float(pred @ truth) + lam, float(truth.sum())], dtype=np.float64)
    scale, shift = np.linalg.solve(gram, rhs)
    return float(scale), float(shift)


def solve_scale_only(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Same line, but the scale penalty uses p·p instead of a fixed lambda."""
    pred = _vec(prediction)
    truth = _vec(target)
    if pred.shape != truth.shape or pred.size == 0:
        raise ValueError("prediction and target must be the same non-empty length")
    energy = float(pred @ pred)
    gram = np.array(
        [[2.0 * energy, float(pred.sum())], [float(pred.sum()), 2.0 * float(pred.size)]],
        dtype=np.float64,
    )
    rhs = np.array([float(pred @ truth) + energy, float(truth.sum())], dtype=np.float64)
    scale, shift = np.linalg.solve(gram, rhs)
    return float(scale), float(shift)


def solve_centered(prediction: np.ndarray, target: np.ndarray, kappa: float) -> tuple[float, float]:
    """Penalize (a-1) * mean(p) and the level correction on a common scale."""
    if math.isinf(kappa):
        return 1.0, 0.0
    if not math.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be non-negative")
    pred = _vec(prediction)
    truth = _vec(target)
    if pred.shape != truth.shape or pred.size == 0:
        raise ValueError("prediction and target must be the same non-empty length")
    n = int(pred.size)
    center = float(pred.mean())
    centered = pred - center
    residual = truth - pred
    denom = float(centered @ centered) + float(kappa) * n * center * center
    deviation = 0.0 if denom <= 0.0 else float(centered @ residual) / denom
    level = float(residual.sum()) / (n * (1.0 + float(kappa)))
    return 1.0 + deviation, level - deviation * center


def fit_cells(
    prediction: np.ndarray,
    target: np.ndarray,
    eligible: np.ndarray,
    solver: Callable[..., tuple[float, float]],
    strength: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    keep = np.asarray(eligible, dtype=bool)
    if pred.ndim != 2 or pred.shape != truth.shape or keep.shape != (pred.shape[1],):
        raise ValueError("expected prediction/target (hours, cells) and one flag per cell")
    scale = np.ones(pred.shape[1], dtype=np.float64)
    shift = np.zeros(pred.shape[1], dtype=np.float64)
    for index in np.flatnonzero(keep):
        if strength is None:
            found = solver(pred[:, index], truth[:, index])
        else:
            found = solver(pred[:, index], truth[:, index], strength)
        scale[index], shift[index] = found
    return scale, shift


def apply_affine(prediction: np.ndarray, scale: np.ndarray, shift: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float64)
    if pred.ndim != 2 or scale.shape != (pred.shape[1],) or shift.shape != scale.shape:
        raise ValueError("scale and shift must match the cell axis")
    return pred * scale.reshape(1, -1) + shift.reshape(1, -1)


def lodo_curve(
    pred_train: np.ndarray,
    target_train: np.ndarray,
    eligible: np.ndarray,
    solver: Callable[..., tuple[float, float]],
    grid: tuple[float, ...] = GRID,
) -> list[dict[str, float]]:
    """Leave-one-day MAE on the training window. Val and test are not arguments."""
    hours = int(pred_train.shape[0])
    if hours != N_HOURS:
        raise ValueError(f"train hours {hours} != {N_HOURS}")
    if hours % DAY_HOURS:
        raise ValueError("train window is not a whole number of days")
    folds = [np.arange(day * DAY_HOURS, (day + 1) * DAY_HOURS) for day in range(hours // DAY_HOURS)]
    cells = np.flatnonzero(np.asarray(eligible, dtype=bool))
    if cells.size == 0:
        raise ValueError("no eligible cells")
    curve = []
    for strength in grid:
        abs_sum = 0.0
        sq_sum = 0.0
        count = 0
        for hold in folds:
            keep = np.setdiff1d(np.arange(hours), hold)
            for index in cells:
                scale, shift = solver(pred_train[keep, index], target_train[keep, index], strength)
                gap = scale * pred_train[hold, index] + shift - target_train[hold, index]
                abs_sum += float(np.abs(gap).sum())
                sq_sum += float(np.square(gap).sum())
                count += int(gap.size)
        curve.append({"strength": float(strength), "mae": abs_sum / count, "rmse": (sq_sum / count) ** 0.5})
    return curve


def pick_strength(curve: list[dict[str, float]]) -> float:
    best = min(item["mae"] for item in curve)
    tied = [item["strength"] for item in curve if item["mae"] - best < TIE_ATOL]
    return max(tied)


def day_block_bootstrap(
    pred_arm: np.ndarray,
    pred_ref: np.ndarray,
    target: np.ndarray,
    n_rep: int = BOOT_REPS,
    seed: int = BOOT_SEED,
) -> tuple[float, float, float]:
    hours, cells = target.shape
    if hours % DAY_HOURS:
        raise ValueError("hours must be whole days")
    days = hours // DAY_HOURS
    arm = np.abs(pred_arm - target).reshape(days, DAY_HOURS, cells).sum(axis=(1, 2))
    ref = np.abs(pred_ref - target).reshape(days, DAY_HOURS, cells).sum(axis=(1, 2))
    drawn = np.random.default_rng(seed).integers(0, days, size=(n_rep, days))
    diff = (arm[drawn].sum(axis=1) - ref[drawn].sum(axis=1)) / (days * DAY_HOURS * cells)
    point = float((arm.sum() - ref.sum()) / (days * DAY_HOURS * cells))
    return point, float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))


def _checkpoint(directory: Path, source: str, seed: int) -> Path:
    stem = "finetune_{0}2dc".format(source) if seed == 42 else "finetune_{0}2dc_seed{1}".format(source, seed)
    path = directory / f"{stem}.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"checkpoint missing: {path}. The name comes from train_finetune.py; do not substitute another file."
        )
    return path


def _cache(cache_dir: Path, source: str, seed: int, split: str) -> Path:
    return cache_dir / f"{source}_seed{seed}_{split}.npz"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.floating):
        return _jsonable(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        if math.isnan(value):
            return None
        return value
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _strength(value: Any) -> float:
    if value == "inf":
        return math.inf
    return float(value)


def _fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _fmt_strength(value: float) -> str:
    if math.isinf(value):
        return "∞"
    text = f"{value:.4g}"
    return text


def _sign(value: float, tol: float = 1e-8) -> int:
    if abs(value) <= tol:
        return 0
    return 1 if value > 0 else -1


def _quantile_row(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("no finite values")
    return {
        "median": float(np.median(finite)),
        "p10": float(np.quantile(finite, 0.1)),
        "p90": float(np.quantile(finite, 0.9)),
    }


def _collect(net, loader, mask, device, value_scale: float, min_val: float) -> tuple[np.ndarray, np.ndarray]:
    import torch

    preds = []
    trues = []
    valid = mask.reshape(-1).bool()
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device)
            target = target.to(device)
            out = net(inputs, spatial_mask=mask)
            y_valid = target.reshape(target.shape[0], 1, -1)[:, :, valid]
            preds.append((out * value_scale + min_val).detach().double().cpu().numpy()[:, 0, :])
            trues.append((y_valid * value_scale + min_val).detach().double().cpu().numpy()[:, 0, :])
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def _log_metrics(log_dir: Path, source: str, seed: int) -> dict[str, float] | None:
    name = LOG_NAME.get((source, seed))
    if name is None:
        return None
    path = log_dir / name
    if not path.is_file():
        return None
    payload = _read_json(path)
    return {
        "val_mae": float(payload["val"]["mae"]),
        "val_rmse": float(payload["val"]["rmse"]),
        "test_mae": float(payload["test"]["mae"]),
        "test_rmse": float(payload["test"]["rmse"]),
    }


def stage_predict(processed_dir: Path, checkpoint_dir: Path, cache_dir: Path, log_dir: Path, device_name: str, seeds: tuple[int, ...]) -> int:
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import load_city_array, min_max_normalize, prepare_city_windows, valid_region_mask
    from src.utils.config import load_config

    route = _load_module("bin_route_v2", "13_bin_route.py")
    volume = _load_module("cell_volume_v2", "14_cell_volume.py")
    published = _load_module("cell_affine_published_score", "17_cell_affine.py")
    raw = load_city_array("dc", processed_dir)
    raw_valid = valid_region_mask(raw)
    prepared = prepare_city_windows("dc", processed_dir, few_shot_days=3)
    if not np.array_equal(raw_valid, np.asarray(prepared["mask"])):
        raise ValueError("raw valid mask does not match the prepared windows")
    normalized, max_val, min_val = min_max_normalize(raw)
    if abs(float(prepared["max_val"]) - max_val) > 1e-6 or abs(float(prepared["min_val"]) - min_val) > 1e-6:
        raise ValueError("min-max statistics differ between the two loads")
    clipped = normalized.astype(np.float64) * (max_val - min_val) + min_val
    valid_flat = raw_valid.reshape(-1)
    expected = clipped.reshape(clipped.shape[0], -1)[LABEL_START : LABEL_START + N_HOURS, valid_flat]
    three_total = volume._prepare(raw, raw_valid)["three_sum"].sum(axis=1)
    eligible = np.asarray(three_total >= GATE)
    if int(raw_valid.sum()) != N_CELLS or int(eligible.sum()) != N_GATE:
        raise ValueError(f"cells {int(raw_valid.sum())}/{int(eligible.sum())} != {N_CELLS}/{N_GATE}")
    rows, cols = np.nonzero(raw_valid)
    value_scale = float(prepared["max_val"] - prepared["min_val"])
    min_value = float(prepared["min_val"])
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    if device_name == "cpu":
        torch.set_num_threads(4)
    device = torch.device(device_name)
    mask = torch.as_tensor(prepared["mask"])
    loaders = {
        "train": DataLoader(prepared["train"], batch_size=32, shuffle=False),
        "val": DataLoader(prepared["val"], batch_size=32, shuffle=False),
        "test": DataLoader(prepared["test"], batch_size=32, shuffle=False),
    }
    model_cfg = load_config(ROOT / "configs" / "default.yaml").get("model", {})
    cache_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    comparisons = []
    for source in SOURCES:
        for seed in seeds:
            path = _checkpoint(checkpoint_dir, source, seed)
            print(f"predict {source} seed {seed}", flush=True)
            net = route._load_net(path, mask, device, model_cfg)
            series = {}
            for split, loader in loaders.items():
                prediction, target = _collect(net, loader, mask, device, value_scale, min_value)
                series[split] = (prediction, target)
            del net
            if device.type == "cuda":
                torch.cuda.empty_cache()
            train_pred, train_true = series["train"]
            if train_pred.shape != (N_HOURS, N_CELLS):
                raise ValueError(f"{source} seed {seed} train shape {train_pred.shape}")
            label_err = float(np.max(np.abs(train_true - expected)))
            disagree = int((eligible != (train_true.sum(axis=0) >= GATE)).sum())
            if label_err > LABEL_ATOL:
                failures.append(f"{source} seed {seed} label max abs {label_err:.6g} > {LABEL_ATOL}")
            if disagree != 0:
                failures.append(f"{source} seed {seed} gate disagreements {disagree}")
            for split in ("val", "test"):
                prediction, target = series[split]
                if prediction.shape != (1440, N_CELLS):
                    raise ValueError(f"{source} seed {seed} {split} shape {prediction.shape}")
                metrics = published.score_arrays(prediction, target, value_scale)
                published_mae, published_rmse = _published_pair(source, seed, split, log_dir)
                record = {
                    "source": source,
                    "seed": seed,
                    "split": split,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "published_mae": published_mae,
                    "published_rmse": published_rmse,
                    "log_checked": published_mae is not None,
                }
                if published_mae is None:
                    record["status"] = "未核对"
                else:
                    err_mae = abs(metrics["mae"] - published_mae)
                    err_rmse = abs(metrics["rmse"] - published_rmse)
                    record["abs_mae"] = err_mae
                    record["abs_rmse"] = err_rmse
                    record["status"] = "通过" if err_mae <= BASELINE_ATOL and err_rmse <= BASELINE_ATOL else "失败"
                    if record["status"] != "通过":
                        failures.append(
                            f"{source} seed {seed} {split} {metrics['mae']:.6f}/{metrics['rmse']:.6f} "
                            f"!= {published_mae:.6f}/{published_rmse:.6f}"
                        )
                comparisons.append(record)
            for split, (prediction, target) in series.items():
                payload = {"prediction": prediction, "target": target, "value_scale": np.float64(value_scale)}
                if split == "train":
                    payload.update(
                        {
                            "eligible": eligible,
                            "rows": rows.astype(np.int16),
                            "cols": cols.astype(np.int16),
                            "label_max_abs_err": np.float64(label_err),
                            "gate_disagree": np.int64(disagree),
                        }
                    )
                np.savez_compressed(_cache(cache_dir, source, seed, split), **payload)
    checks = {
        "n_cells": N_CELLS,
        "n_gate": int(eligible.sum()),
        "gate": GATE,
        "device": device_name,
        "comparisons": comparisons,
        "failures": failures,
        "passed": not failures,
    }
    _write_json(cache_dir / "checks.json", checks)
    for item in comparisons:
        print(
            f"  {item['source']} seed {item['seed']} {item['split']} "
            f"{item['mae']:.4f}/{item['rmse']:.4f} {item['status']}",
            flush=True,
        )
    if failures:
        print("predict checks failed:", file=sys.stderr)
        for item in failures:
            print(f"  {item}", file=sys.stderr)
        return 1
    print("predict checks passed", flush=True)
    return 0


def _published_pair(source: str, seed: int, split: str, log_dir: Path) -> tuple[float | None, float | None]:
    if seed == 42 and split == "val":
        return PUBLISHED_VAL_42[source]
    if split == "test":
        logged = _log_metrics(log_dir, source, seed)
        if logged is not None:
            return logged["test_mae"], logged["test_rmse"]
        return PUBLISHED_TEST[(source, seed)]
    if split == "val":
        logged = _log_metrics(log_dir, source, seed)
        if logged is None:
            return None, None
        return logged["val_mae"], logged["val_rmse"]
    return None, None


def _train_stats(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = _vec(prediction)
    n = pred.size
    center = float(pred.mean())
    centered = pred - center
    energy = float(centered @ centered)
    dot = float(pred @ pred)
    if abs(center) <= 1e-12:
        cv2 = math.inf if energy > 0 else 0.0
    else:
        cv2 = energy / (n * center * center)
    scale, shift = solve_centered(pred, target, 0.0)
    return {
        "p_bar": center,
        "cv2": cv2,
        "p_dot_p": dot,
        "lambda_over_pp": N_HOURS / dot,
        "a_ols": scale,
        "b_ols": shift,
    }


def _arm_gated(scale: np.ndarray, shift: np.ndarray, stats: list[dict[str, float]]) -> dict[str, Any]:
    order = []
    for coef_a, coef_b, item in zip(scale, shift, stats):
        level_from_scale = abs((coef_a - 1.0) * item["p_bar"])
        denom = level_from_scale + abs(coef_b)
        share = None if denom <= 0 else level_from_scale / denom
        order.append(
            {
                "a": float(coef_a),
                "b": float(coef_b),
                "a_dev": float(coef_a - 1.0),
                "c": float(coef_b + (coef_a - 1.0) * item["p_bar"]),
                "level_share": share,
                "flipped": bool(
                    _sign(coef_a - 1.0) != 0
                    and _sign(item["a_ols"] - 1.0) != 0
                    and _sign(coef_a - 1.0) != _sign(item["a_ols"] - 1.0)
                ),
            }
        )
    shares = [item["level_share"] for item in order if item["level_share"] is not None]
    coefficients = np.array([item["a"] for item in order], dtype=np.float64)
    return {
        "cells": order,
        "n_flipped": int(sum(item["flipped"] for item in order)),
        "level_share_median": None if not shares else float(np.median(shares)),
        "a_min": float(coefficients.min()),
        "n_a_below_0_5": int(np.sum(coefficients < 0.5)),
    }


def stage_fit(cache_dir: Path, out_dir: Path, seeds: tuple[int, ...]) -> int:
    checks_path = cache_dir / "checks.json"
    if not checks_path.is_file():
        print("fit refused: predict checks are missing", file=sys.stderr)
        return 1
    checks = _read_json(checks_path)
    if not checks.get("passed"):
        print("fit refused: predict checks failed", file=sys.stderr)
        return 1
    runs = []
    reference_eligible = None
    for source in SOURCES:
        for seed in seeds:
            train_path = _cache(cache_dir, source, seed, "train")
            if not train_path.is_file():
                print(f"fit refused: missing {train_path}", file=sys.stderr)
                return 1
            # Only the training cache is opened. Val and test stay unread until score.
            with np.load(train_path) as blob:
                prediction = np.asarray(blob["prediction"], dtype=np.float64)
                target = np.asarray(blob["target"], dtype=np.float64)
                eligible = np.asarray(blob["eligible"], dtype=bool)
                rows = np.asarray(blob["rows"])
                cols = np.asarray(blob["cols"])
            if prediction.shape != (N_HOURS, N_CELLS) or int(eligible.sum()) != N_GATE:
                raise ValueError(f"{source} seed {seed} train cache is not the 72-hour gated problem")
            if reference_eligible is None:
                reference_eligible = eligible.copy()
            elif not np.array_equal(eligible, reference_eligible):
                raise ValueError(f"{source} seed {seed} gate does not match the other runs")
            print(f"fit {source} seed {seed}", flush=True)
            old_curve = lodo_curve(prediction, target, eligible, solve_old)
            new_curve = lodo_curve(prediction, target, eligible, solve_centered)
            old_choice = pick_strength(old_curve)
            new_choice = pick_strength(new_curve)
            gated = np.flatnonzero(eligible)
            stats = [_train_stats(prediction[:, index], target[:, index]) for index in gated]
            identity_scale, identity_shift = fit_cells(prediction, target, eligible, solve_centered, 0.0)
            old_zero_scale, old_zero_shift = fit_cells(prediction, target, eligible, solve_old, 0.0)
            gap = max(
                float(np.max(np.abs(identity_scale - old_zero_scale))),
                float(np.max(np.abs(identity_shift - old_zero_shift))),
            )
            if gap > 1e-8:
                print(f"fit refused: kappa=0 and m=0 differ by {gap:.3e} on {source} seed {seed}", file=sys.stderr)
                return 1
            infinite_scale, infinite_shift = fit_cells(prediction, target, eligible, solve_centered, math.inf)
            if not np.allclose(infinite_scale, 1.0) or not np.allclose(infinite_shift, 0.0):
                print(f"fit refused: infinite kappa is not identity on {source} seed {seed}", file=sys.stderr)
                return 1
            specifications: list[tuple[str, Callable[..., tuple[float, float]], float | None]] = [
                ("B0", solve_centered, math.inf),
                ("O72", solve_old, 1.0),
                ("Ostar", solve_old, old_choice),
                ("S", solve_scale_only, None),
                ("N1", solve_centered, 1.0),
                ("Nstar", solve_centered, new_choice),
                ("N0", solve_centered, 0.0),
            ]
            arms = {}
            for name, solver, strength in specifications:
                scale, shift = fit_cells(prediction, target, eligible, solver, strength)
                if np.any(scale[~eligible] != 1.0) or np.any(shift[~eligible] != 0.0):
                    raise ValueError(f"{name} changed a cell below the gate")
                summary = _arm_gated(scale[gated], shift[gated], stats)
                summary["scale"] = scale
                summary["shift"] = shift
                summary["strength"] = None if strength is None else float(strength)
                arms[name] = summary
            runs.append(
                {
                    "source": source,
                    "seed": seed,
                    "n_gate": N_GATE,
                    "cv_old": {"curve": old_curve, "chosen": old_choice},
                    "cv_centered": {"curve": new_curve, "chosen": new_choice},
                    "identity_max_abs": gap,
                    "gated_train": {
                        "index": gated.astype(int),
                        "row": rows[gated].astype(int),
                        "col": cols[gated].astype(int),
                        **{key: [item[key] for item in stats] for key in ("p_bar", "cv2", "p_dot_p", "lambda_over_pp", "a_ols", "b_ols")},
                    },
                    "arms": arms,
                }
            )
            print(f"  chosen m={_fmt_strength(old_choice)} kappa={_fmt_strength(new_choice)}", flush=True)
    payload = {
        "channel": "taxi_pickup",
        "trained": False,
        "gate": GATE,
        "n_hours": N_HOURS,
        "grid": list(GRID),
        "selection": "leave one of the three training days out; the smallest CV MAE wins, ties take the larger strength",
        "runs": runs,
    }
    _write_json(out_dir / "cell_affine_v2_fit.json", payload)
    print(f"wrote {out_dir / 'cell_affine_v2_fit.json'}", flush=True)
    return 0


def _metrics(prediction: np.ndarray, target: np.ndarray, value_scale: float) -> dict[str, float]:
    published = _load_module("cell_affine_score_fn", "17_cell_affine.py")
    return published.score_arrays(prediction, target, value_scale)


def _subset_mae(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool)
    gap = prediction[:, selected] - target[:, selected]
    return float(np.abs(gap).sum()) / float(gap.size)


def _load_split(cache_dir: Path, source: str, seed: int, split: str) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(_cache(cache_dir, source, seed, split)) as blob:
        return (
            np.asarray(blob["prediction"], dtype=np.float64),
            np.asarray(blob["target"], dtype=np.float64),
            float(blob["value_scale"]),
        )


def _calendar_slices(target_rows: int) -> dict[str, np.ndarray]:
    from src.data.calendar_mask import is_weekend_or_holiday, label_hours

    hours = label_hours("test")
    if hours.shape[0] != target_rows:
        raise ValueError(f"test calendar {hours.shape[0]} != {target_rows}")
    weekend = np.asarray(is_weekend_or_holiday(hours), dtype=bool)
    return {"weekday": ~weekend, "weekend": weekend}


def stage_score(cache_dir: Path, out_dir: Path, seeds: tuple[int, ...]) -> int:
    fit_path = out_dir / "cell_affine_v2_fit.json"
    fit_bytes = fit_path.read_bytes()
    fit = _read_json(fit_path)
    checks = _read_json(cache_dir / "checks.json")
    by_key = {(item["source"], int(item["seed"])): item for item in fit["runs"]}
    slices = None
    scored_runs = []
    failures = []
    for source in SOURCES:
        for seed in seeds:
            item = by_key[(source, seed)]
            print(f"score {source} seed {seed}", flush=True)
            splits = {}
            value_scale = None
            loaded = {}
            for split in ("train", "val", "test"):
                prediction, target, scale = _load_split(cache_dir, source, seed, split)
                loaded[split] = (prediction, target)
                value_scale = scale
            assert value_scale is not None
            eligible = np.zeros(N_CELLS, dtype=bool)
            eligible[np.asarray(item["gated_train"]["index"], dtype=int)] = True
            if slices is None:
                slices = _calendar_slices(loaded["test"][0].shape[0])
            arm_rows = {}
            for name in ARM_ORDER:
                arm = item["arms"][name]
                scale = np.asarray(arm["scale"], dtype=np.float64)
                shift = np.asarray(arm["shift"], dtype=np.float64)
                below = ~eligible
                arm_splits = {}
                corrected_splits = {}
                for split, (prediction, target) in loaded.items():
                    corrected = apply_affine(prediction, scale, shift)
                    corrected_splits[split] = corrected
                    if float(np.max(np.abs(corrected[:, below] - prediction[:, below]))) != 0.0:
                        failures.append(f"{source} seed {seed} {name} changed a below-gate cell on {split}")
                    arm_splits[split] = {
                        "all": _metrics(corrected, target, value_scale),
                        "gate_mae": _subset_mae(corrected, target, eligible),
                        "below_mae": _subset_mae(corrected, target, below),
                    }
                test_pred = corrected_splits["test"]
                test_true = loaded["test"][1]
                arm_splits["test"]["negative_points"] = int(np.sum(test_pred < 0))
                arm_splits["test"]["calendar"] = {
                    key: _metrics(test_pred[mask], test_true[mask], value_scale)["mae"]
                    for key, mask in slices.items()
                }
                arm_rows[name] = {"splits": arm_splits, "corrected_test": test_pred, "corrected_val": corrected_splits["val"]}
            if seed == 42:
                for split, expected in PUBLISHED_O72_MAE[source].items():
                    got = arm_rows["O72"]["splits"][split]["all"]["mae"]
                    if abs(got - expected) > O72_ATOL:
                        failures.append(f"{source} O72 {split} MAE {got:.8f} != {expected:.8f}")
                median = float(np.median(np.asarray(item["arms"]["O72"]["scale"], dtype=np.float64)[eligible]))
                if abs(median - PUBLISHED_O72_MEDIAN[source]) > O72_ATOL:
                    failures.append(f"{source} O72 scale median {median:.8f} != {PUBLISHED_O72_MEDIAN[source]:.8f}")
            comparisons = {}
            for name in ARM_ORDER:
                if name == "O72":
                    continue
                point, low, high = day_block_bootstrap(
                    arm_rows[name]["corrected_test"],
                    arm_rows["O72"]["corrected_test"],
                    loaded["test"][1],
                )
                base_point, base_low, base_high = day_block_bootstrap(
                    arm_rows[name]["corrected_test"],
                    arm_rows["B0"]["corrected_test"],
                    loaded["test"][1],
                )
                val_point, val_low, val_high = day_block_bootstrap(
                    arm_rows[name]["corrected_val"],
                    arm_rows["O72"]["corrected_val"],
                    loaded["val"][1],
                )
                comparisons[name] = {
                    "test_vs_o72": {"point": point, "low": low, "high": high},
                    "test_vs_b0": {"point": base_point, "low": base_low, "high": base_high},
                    "val_vs_o72": {"point": val_point, "low": val_low, "high": val_high},
                }
            secondary = {}
            for left, right, _title in SECONDARY_PAIRS:
                point, low, high = day_block_bootstrap(
                    arm_rows[left]["corrected_test"],
                    arm_rows[right]["corrected_test"],
                    loaded["test"][1],
                )
                secondary[f"{left}_vs_{right}"] = {"point": point, "low": low, "high": high, "left": left, "right": right}
            sensitivity = {
                "old": _sensitivity(loaded["train"], loaded["val"], eligible, solve_old, value_scale),
                "centered": _sensitivity(loaded["train"], loaded["val"], eligible, solve_centered, value_scale),
            }
            for name in ARM_ORDER:
                arm_rows[name].pop("corrected_test")
                arm_rows[name].pop("corrected_val")
            scored_runs.append(
                {
                    "source": source,
                    "seed": seed,
                    "chosen_m": _strength(item["cv_old"]["chosen"]),
                    "chosen_kappa": _strength(item["cv_centered"]["chosen"]),
                    "arms": arm_rows,
                    "comparisons": comparisons,
                    "secondary": secondary,
                    "sensitivity_val": sensitivity,
                }
            )
    if fit_path.read_bytes() != fit_bytes:
        raise RuntimeError("score modified the fit file")
    if failures:
        print("score checks failed:", file=sys.stderr)
        for item in failures:
            print(f"  {item}", file=sys.stderr)
        return 1
    report = {
        "channel": "taxi_pickup",
        "trained": False,
        "seeds": list(seeds),
        "bootstrap": {"block": "day", "reps": BOOT_REPS, "seed": BOOT_SEED},
        "checks": checks,
        "fit_identity_max_abs": {f"{item['source']}_seed{item['seed']}": item["identity_max_abs"] for item in fit["runs"]},
        "runs": scored_runs,
        "verdict": _verdict(scored_runs),
        "next_default_kappa": _next_default(fit["runs"]),
    }
    _write_json(out_dir / "cell_affine_v2.json", report)
    md_path = out_dir / "cell_affine_v2.md"
    md_path.write_text(_markdown(report, fit, slices), encoding="utf-8", newline="\n")
    print(f"wrote {md_path}", flush=True)
    print(report["verdict"]["primary"], flush=True)
    return 0


def _sensitivity(train, val, eligible, solver, value_scale: float) -> list[dict[str, float]]:
    rows = []
    pred_train, true_train = train
    pred_val, true_val = val
    for strength in GRID:
        scale, shift = fit_cells(pred_train, true_train, eligible, solver, strength)
        corrected = apply_affine(pred_val, scale, shift)
        metrics = _metrics(corrected, true_val, value_scale)
        rows.append({"strength": float(strength), "mae": metrics["mae"], "rmse": metrics["rmse"]})
    return rows


def _pair(runs: list[dict[str, Any]], source: str, seed: int, arm: str) -> dict[str, Any]:
    found = next(item for item in runs if item["source"] == source and int(item["seed"]) == seed)
    comparison = found["comparisons"][arm]["test_vs_o72"]
    rmse_arm = found["arms"][arm]["splits"]["test"]["all"]["rmse"]
    rmse_ref = found["arms"]["O72"]["splits"]["test"]["all"]["rmse"]
    return {**comparison, "rmse_arm": rmse_arm, "rmse_ref": rmse_ref, "source": source, "seed": seed}


def _verdict(runs: list[dict[str, Any]]) -> dict[str, Any]:
    primary_rows = [_pair(runs, source, 42, "N1") for source in SOURCES]
    lower = all(row["high"] < 0 and row["rmse_arm"] <= row["rmse_ref"] + 1e-12 for row in primary_rows)
    higher = any(row["low"] > 0 for row in primary_rows)
    if lower:
        primary = "更低"
    elif higher:
        primary = "更高"
    else:
        primary = "无法区分"
    signs = []
    for seed in SEEDS:
        for source in SOURCES:
            signs.append(float(np.sign(_pair(runs, source, seed, "N1")["point"])))
    stable = len(set(signs)) == 1 and signs[0] != 0
    return {"primary": primary, "primary_rows": primary_rows, "stable": stable, "signs": signs}


def _next_default(fit_runs: list[dict[str, Any]]) -> dict[str, Any]:
    chosen = [_strength(item["cv_centered"]["chosen"]) for item in fit_runs]
    counts: dict[str, int] = {}
    for value in chosen:
        key = "inf" if math.isinf(value) else str(value)
        counts[key] = counts.get(key, 0) + 1
    top = max(counts, key=lambda key: (counts[key], _strength(key)))
    use = top if counts[top] >= 5 and _strength(top) != 1.0 else "1.0"
    return {"chosen": ["inf" if math.isinf(value) else value for value in chosen], "counts": counts, "next_default": use}


def _markdown(report: dict[str, Any], fit: dict[str, Any], slices: dict[str, np.ndarray]) -> str:
    fit_runs = {(item["source"], int(item["seed"])): item for item in fit["runs"]}
    lines = [
        "# 按格仿射第二版：中心化收缩对照",
        "",
        "通道是出租上车。没有新训练。种子 42、43、44。",
        "起点是已经训好的整城微调。过门规则仍是 72 小时原始上车不少于 150，过门 50 格，其余格子 a=1、b=0。",
        "旧公式是 λ = m×n，已发表结果是 m=1。新公式把 (a−1) 乘上训练窗预测均值，再和水平修正一起惩罚，主设定 κ=1。",
        "m 和 κ 的网格是 0、0.125、0.25、0.5、1、2、4、8、16、∞。留一天交叉验证只用 8 月 31 日、9 月 1 日、9 月 2 日，平局取更大的强度。",
        "验证集和测试集不参与选择。主比较是种子 42 上新公式 κ=1 对旧公式 λ=72。",
        "",
        f"主比较结论：{report['verdict']['primary']}。方向{'稳定' if report['verdict']['stable'] else '不稳定'}。",
        "",
        "## 核对",
        "",
    ]
    for item in report["checks"]["comparisons"]:
        if item["published_mae"] is None:
            lines.append(f"- {CITY[item['source']]} 种子 {item['seed']} {item['split']}：未核对。")
        else:
            lines.append(
                f"- {CITY[item['source']]} 种子 {item['seed']} {item['split']}："
                f"{item['status']}，MAE {_fmt(item['mae'])}，RMSE {_fmt(item['rmse'])}。"
            )
    lines.append("- 训练标签与截断后计数、门槛不一致格数、κ=0 与 m=0、O72 复现已发表 MAE，均在打分前通过，否则本页不会写出。")
    lines.extend(["", "## 训练窗里选出的强度", ""])
    for source in SOURCES:
        for seed in SEEDS:
            item = fit_runs[(source, seed)]
            lines.append(f"### {CITY[source]}，种子 {seed}")
            lines.append("")
            lines.append("| 强度 | 旧公式 CV MAE | 旧公式 CV RMSE | 新公式 CV MAE | 新公式 CV RMSE |")
            lines.append("|---:|---:|---:|---:|---:|")
            for old, new in zip(item["cv_old"]["curve"], item["cv_centered"]["curve"]):
                lines.append(
                    f"| {_fmt_strength(_strength(old['strength']))} | {_fmt(old['mae'])} | {_fmt(old['rmse'])} | {_fmt(new['mae'])} | {_fmt(new['rmse'])} |"
                )
            lines.append("")
            lines.append(
                f"选出的倍数 m = {_fmt_strength(_strength(item['cv_old']['chosen']))}，"
                f"κ = {_fmt_strength(_strength(item['cv_centered']['chosen']))}。"
            )
            lines.append("")
    lines.extend(["## 全城结果", ""])
    for source in SOURCES:
        for seed in SEEDS:
            run = next(item for item in report["runs"] if item["source"] == source and int(item["seed"]) == seed)
            lines.append(f"### {CITY[source]}，种子 {seed}")
            lines.append("")
            lines.append("| 臂 | 强度 | train MAE | val MAE | test MAE | test RMSE | test MAPE | 过门 50 格 test MAE | 相对 O72 的 test MAE 差 [2.5%, 97.5%] |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
            for name in ARM_ORDER:
                arm = run["arms"][name]
                strength = _arm_strength_text(name, run)
                diff = "—" if name == "O72" else _interval(run["comparisons"][name]["test_vs_o72"])
                test = arm["splits"]["test"]["all"]
                lines.append(
                    f"| {ARM_LABEL[name]} | {strength} | {_fmt(arm['splits']['train']['all']['mae'])} | "
                    f"{_fmt(arm['splits']['val']['all']['mae'])} | {_fmt(test['mae'])} | {_fmt(test['rmse'])} | "
                    f"{_fmt(test['mape'])} | {_fmt(arm['splits']['test']['gate_mae'])} | {diff} |"
                )
            lines.append("")
    lines.extend(
        [
            "## 三种子均值",
            "",
            "| 臂 | 纽约 test MAE | 纽约 test RMSE | 芝加哥 test MAE | 芝加哥 test RMSE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ARM_ORDER:
        cells = []
        for source in SOURCES:
            group = [item for item in report["runs"] if item["source"] == source]
            mae = np.mean([item["arms"][name]["splits"]["test"]["all"]["mae"] for item in group])
            rmse = np.mean([item["arms"][name]["splits"]["test"]["all"]["rmse"] for item in group])
            cells.extend([_fmt(float(mae)), _fmt(float(rmse))])
        lines.append(f"| {ARM_LABEL[name]} | {' | '.join(cells)} |")
    lines.extend(["", "MAPE 不参与判定。三种子 test MAPE 均值：", ""])
    lines.append("| 臂 | 纽约 | 芝加哥 |")
    lines.append("|---|---:|---:|")
    for name in ARM_ORDER:
        cells = []
        for source in SOURCES:
            group = [item for item in report["runs"] if item["source"] == source]
            mape = np.mean([item["arms"][name]["splits"]["test"]["all"]["mape"] for item in group])
            cells.append(_fmt(float(mape)))
        lines.append(f"| {ARM_LABEL[name]} | {' | '.join(cells)} |")
    primary = report["verdict"]["primary"]
    detail = "；".join(
        f"{CITY[row['source']]} {_interval(row)}" for row in report["verdict"]["primary_rows"]
    )
    lines.extend(
        [
            "",
            "## 判定",
            "",
            f"主比较，种子 42，新公式 κ=1 对旧公式 λ=72：{primary}。{detail}。",
            f"六个源城×种子的 test MAE 差，符号{'全部相同' if report['verdict']['stable'] else '不完全相同'}，因此方向{'稳定' if report['verdict']['stable'] else '不稳定'}。",
            "",
            "次要比较只列区间，不改上面的主结论。差是前者减后者，负值表示前者 MAE 更小。",
            "",
            "| 比较 | 纽约 test MAE 差 [2.5%, 97.5%] | 芝加哥 test MAE 差 [2.5%, 97.5%] |",
            "|---|---|---|",
        ]
    )
    for left, _right, title in SECONDARY_PAIRS:
        ny = _secondary_interval(report, "ny", 42, left, _right)
        chi = _secondary_interval(report, "chi", 42, left, _right)
        lines.append(f"| {title} | {ny} | {chi} |")
    lines.extend(["", "## 系数和训练窗统计", ""])
    for source in SOURCES:
        for seed in SEEDS:
            item = fit_runs[(source, seed)]
            lines.append(f"### {CITY[source]}，种子 {seed}")
            lines.append("")
            lines.append("| 臂 | a 中位数 | a 10–90 | b 中位数 | b 10–90 | 反向格数 | 水平由 a 承担的比例中位数 | a 最小值 | a<0.5 的格数 |")
            lines.append("|---|---:|---|---:|---|---:|---:|---:|---:|")
            for name in ARM_ORDER:
                cells = item["arms"][name]["cells"]
                a_stats = _quantile_row([cell["a"] for cell in cells])
                b_stats = _quantile_row([cell["b"] for cell in cells])
                share = item["arms"][name]["level_share_median"]
                lines.append(
                    f"| {ARM_LABEL[name]} | {_fmt(a_stats['median'])} | {_fmt(a_stats['p10'])} 到 {_fmt(a_stats['p90'])} | "
                    f"{_fmt(b_stats['median'])} | {_fmt(b_stats['p10'])} 到 {_fmt(b_stats['p90'])} | "
                    f"{item['arms'][name]['n_flipped']} | {_fmt(share) if share is not None else '—'} | "
                    f"{_fmt(item['arms'][name]['a_min'])} | {item['arms'][name]['n_a_below_0_5']} |"
                )
            train = item["gated_train"]
            lines.append("")
            for key, title, digits in (
                ("p_bar", "预测均值", 4),
                ("cv2", "CV²", 4),
                ("p_dot_p", "p·p", 1),
                ("lambda_over_pp", "72/(p·p)", 6),
            ):
                stats = _quantile_row(train[key])
                lines.append(
                    f"{title}：中位数 {_fmt(stats['median'], digits)}，10 到 90 分位 {_fmt(stats['p10'], digits)} 到 {_fmt(stats['p90'], digits)}。"
                )
            lines.append("")
    weekend_hours = int(np.sum(slices["weekend"]))
    weekday_hours = int(np.sum(slices["weekday"]))
    lines.extend(
        [
            "## 测试段的周中和周末",
            "",
            f"切分用 `is_weekend_or_holiday`。测试段 {weekday_hours} 个周一到周五小时，{weekend_hours} 个周六日小时。",
            "这个函数的节日表只到劳动节，11 月和 12 月的节日没有另标，所以这里的周末就是周六和周日。",
            "",
            "| 源城 | 种子 | 臂 | 周一到周五 MAE | 周六日 MAE |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for source in SOURCES:
        for seed in SEEDS:
            run = next(item for item in report["runs"] if item["source"] == source and int(item["seed"]) == seed)
            for name in ("B0", "O72", "S", "N1", "Nstar"):
                calendar = run["arms"][name]["splits"]["test"]["calendar"]
                lines.append(
                    f"| {CITY[source]} | {seed} | {ARM_LABEL[name]} | {_fmt(calendar['weekday'])} | {_fmt(calendar['weekend'])} |"
                )
    lines.extend(
        [
            "",
            "## 验证集强度曲线（敏感性，未用于选择）",
            "",
            "每一行都是在全部 72 个训练小时上拟合，再在验证集上打分。选择 κ 和 m 时没有读这些数。",
            "",
        ]
    )
    for source in SOURCES:
        run = next(item for item in report["runs"] if item["source"] == source and int(item["seed"]) == 42)
        lines.append(f"### {CITY[source]}，种子 42")
        lines.append("")
        lines.append("| 强度 | 旧公式 val MAE | 新公式 val MAE |")
        lines.append("|---:|---:|---:|")
        for old, new in zip(run["sensitivity_val"]["old"], run["sensitivity_val"]["centered"]):
            lines.append(f"| {_fmt_strength(_strength(old['strength']))} | {_fmt(old['mae'])} | {_fmt(new['mae'])} |")
        lines.append("")
    nxt = report["next_default_kappa"]
    lines.extend(
        [
            "## 局限",
            "",
            "72 小时上的预测是样本内预测，微调就是在这些标签上训练的。",
            "交叉验证只有三折，留出的都是同一周的工作日，看不到 11–12 月测试段的漂移，所以选出的强度可能偏松。",
            "种子 43 和 44 的旧公式 λ=72 是这次第一次计算，种子 42 的那一行对照了已发表数字。",
            f"六个 κ 的取值是 {', '.join(_fmt_strength(_strength(value)) for value in nxt['chosen'])}。按事先写定的规则，下一轮默认 κ 仍记为 { _fmt_strength(_strength(nxt['next_default'])) }。本轮主结论用的是 κ=1。",
            "",
            "重算：",
            "",
            "```bash",
            "python scripts/19_cell_affine_v2.py --stage predict --seeds 42 43 44",
            "python scripts/19_cell_affine_v2.py --stage fit --seeds 42 43 44",
            "python scripts/19_cell_affine_v2.py --stage score --seeds 42 43 44",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _arm_strength_text(name: str, run: dict[str, Any]) -> str:
    if name == "B0":
        return "不校正"
    if name == "O72":
        return "m=1"
    if name == "Ostar":
        return f"m={_fmt_strength(run['chosen_m'])}"
    if name == "S":
        return "p·p"
    if name == "N1":
        return "κ=1"
    if name == "Nstar":
        return f"κ={_fmt_strength(run['chosen_kappa'])}"
    if name == "N0":
        return "κ=0"
    raise KeyError(name)


def _interval(row: dict[str, float]) -> str:
    return f"{_fmt(row['point'])} [{_fmt(row['low'])}, {_fmt(row['high'])}]"


def _secondary_interval(report: dict[str, Any], source: str, seed: int, left: str, right: str) -> str:
    run = next(item for item in report["runs"] if item["source"] == source and int(item["seed"]) == seed)
    return _interval(run["secondary"][f"{left}_vs_{right}"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("predict", "fit", "score"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "logs" / "cell_affine_v2_cache")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "profiles")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    unknown = [seed for seed in seeds if seed not in SEEDS]
    if unknown:
        print(f"seeds must be among {SEEDS}, got {unknown}", file=sys.stderr)
        return 1
    try:
        if args.stage == "predict":
            return stage_predict(args.processed_dir, args.checkpoint_dir, args.cache_dir, args.log_dir, args.device, seeds)
        if args.stage == "fit":
            return stage_fit(args.cache_dir, args.out_dir, seeds)
        return stage_score(args.cache_dir, args.out_dir, seeds)
    except (FileNotFoundError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        print(f"cell affine v2 failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
