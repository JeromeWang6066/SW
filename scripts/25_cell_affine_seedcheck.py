#!/usr/bin/env python3
"""Weekday check of the hour bias on two new fine-tune seeds.

Protocol: docs/CELL_AFFINE_SEEDCHECK_PLAN.md. Seeds and the formula are fixed.
Weekends are left out of the decision.
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

N_HOURS = 72
N_CELLS = 207
N_GATE = 50
TEST_HOURS = 1440
DAY_HOURS = 24
HOUR_KAPPA = 3.0
LOG_ATOL = 1e-3
BOOT_REPS = 2000
BOOT_SEED = 0
SOURCES = ("ny", "chi")
SEEDS = (45, 46)
CITY = {"ny": "纽约", "chi": "芝加哥"}
TASKS = tuple((source, seed) for source in SOURCES for seed in SEEDS)
LOG_NAME = {
    ("ny", 45): "G1C-NY2DC_seed45.json",
    ("ny", 46): "G1C-NY2DC_seed46.json",
    ("chi", 45): "G1C-CHI2DC_seed45.json",
    ("chi", 46): "G1C-CHI2DC_seed46.json",
}


def _load_script(name: str, filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weekday_hours(n_hours: int) -> np.ndarray:
    robust = _load_script("cell_affine_robust_seedcheck", "24_cell_affine_robust.py")
    return robust.weekday_mask(n_hours)


def slice_mae(prediction: np.ndarray, target: np.ndarray, keep: np.ndarray) -> float:
    if keep.shape[0] != prediction.shape[0] or prediction.shape != target.shape:
        raise ValueError("mask, prediction, and target lengths must match")
    if int(keep.sum()) == 0:
        raise ValueError("the scored slice is empty")
    return float(np.mean(np.abs(prediction[keep] - target[keep])))


def weekday_bootstrap(
    candidate: np.ndarray,
    reference: np.ndarray,
    target: np.ndarray,
    keep_hours: np.ndarray,
    reps: int = BOOT_REPS,
    seed: int = BOOT_SEED,
) -> dict[str, float]:
    """Resample whole weekday days. Weekend days stay out of the interval."""
    hours, cells = target.shape
    if hours % DAY_HOURS or keep_hours.shape[0] != hours:
        raise ValueError("expected one mask entry per hour and whole days")
    days = hours // DAY_HOURS
    day_keep = keep_hours.reshape(days, DAY_HOURS).all(axis=1)
    if not np.array_equal(day_keep, keep_hours.reshape(days, DAY_HOURS).any(axis=1)):
        raise ValueError("weekday mask must keep or drop whole days")
    diff = (
        np.abs(candidate - target).reshape(days, DAY_HOURS, cells).sum(axis=(1, 2))
        - np.abs(reference - target).reshape(days, DAY_HOURS, cells).sum(axis=(1, 2))
    )
    chosen = diff[day_keep]
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, chosen.size, size=(reps, chosen.size))
    scale = chosen.size * DAY_HOURS * cells
    samples = chosen[drawn].sum(axis=1) / scale
    return {
        "point": float(chosen.sum() / scale),
        "low": float(np.quantile(samples, 0.025)),
        "high": float(np.quantile(samples, 0.975)),
        "n_days": int(chosen.size),
    }


def _checkpoint(directory: Path, source: str, seed: int) -> Path:
    return directory / f"finetune_{source}2dc_seed{seed}.pt"


def _cache(cache_dir: Path, source: str, seed: int, split: str) -> Path:
    return cache_dir / f"{source}_seed{seed}_{split}.npz"


def export_predictions(
    processed_dir: Path,
    checkpoint_dir: Path,
    cache_dir: Path,
    reference_mask: np.ndarray,
) -> None:
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import load_city_array, min_max_normalize, prepare_city_windows, valid_region_mask
    from src.utils.config import load_config

    published = _load_script("cell_affine_v2_seedcheck", "19_cell_affine_v2.py")
    route = published._load_module("bin_route_seedcheck", "13_bin_route.py")
    volume = published._load_module("cell_volume_seedcheck", "14_cell_volume.py")
    raw = load_city_array("dc", processed_dir)
    raw_valid = valid_region_mask(raw)
    prepared = prepare_city_windows("dc", processed_dir, few_shot_days=3)
    three_total = volume._prepare(raw, raw_valid)["three_sum"].sum(axis=1)
    eligible = np.asarray(three_total >= published.GATE)
    if not np.array_equal(eligible, reference_mask):
        raise RuntimeError("the 150-count gate does not match the seed-42 mask")
    value_scale = float(prepared["max_val"] - prepared["min_val"])
    min_value = float(prepared["min_val"])
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cpu":
        torch.set_num_threads(4)
    device = torch.device(device_name)
    mask = torch.as_tensor(prepared["mask"])
    loaders = {
        split: DataLoader(prepared[split], batch_size=32, shuffle=False) for split in ("train", "val", "test")
    }
    model_cfg = load_config(ROOT / "configs" / "default.yaml").get("model", {})
    rows, cols = np.nonzero(raw_valid)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for source, seed in TASKS:
        path = _checkpoint(checkpoint_dir, source, seed)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint missing: {path}")
        print(f"predict {source} seed {seed}", flush=True)
        net = route._load_net(path, mask, device, model_cfg)
        for split, loader in loaders.items():
            prediction, target = published._collect(net, loader, mask, device, value_scale, min_value)
            payload = {"prediction": prediction, "target": target, "value_scale": np.float64(value_scale)}
            if split == "train":
                payload["eligible"] = eligible
                payload["rows"] = rows.astype(np.int16)
                payload["cols"] = cols.astype(np.int16)
            np.savez_compressed(_cache(cache_dir, source, seed, split), **payload)
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _load_split(cache_dir: Path, source: str, seed: int, split: str) -> dict[str, np.ndarray]:
    with np.load(_cache(cache_dir, source, seed, split)) as blob:
        loaded = {
            "prediction": np.asarray(blob["prediction"], dtype=np.float64),
            "target": np.asarray(blob["target"], dtype=np.float64),
        }
        if split == "train":
            loaded["eligible"] = np.asarray(blob["eligible"], dtype=bool)
    return loaded


def _score_task(train: dict[str, np.ndarray], test: dict[str, np.ndarray], solve_old, hour_offsets) -> dict[str, Any]:
    robust = _load_script("cell_affine_robust_score", "24_cell_affine_robust.py")
    scale, shift, residual = robust.fit_line_and_residual(train["prediction"], train["target"], train["eligible"], solve_old)
    offsets = robust.offsets_from_residual(residual, train["eligible"], HOUR_KAPPA)
    for index in np.flatnonzero(train["eligible"]):
        helper = hour_offsets(residual[:, index], HOUR_KAPPA)
        if not np.allclose(helper, offsets[:, index]):
            raise RuntimeError("hour offsets disagree with the published helper")
    base = test["prediction"] * scale.reshape(1, -1) + shift.reshape(1, -1)
    corrected = robust.apply_correction(test["prediction"], scale, shift, offsets)
    keep = weekday_hours(test["prediction"].shape[0])
    return {
        "weekday": {
            "B0": slice_mae(test["prediction"], test["target"], keep),
            "O72": slice_mae(base, test["target"], keep),
            "O72H": slice_mae(corrected, test["target"], keep),
        },
        "city_mae": float(np.mean(np.abs(test["prediction"] - test["target"]))),
        "n_weekday_hours": int(keep.sum()),
        "n_gate": int(train["eligible"].sum()),
        "diff": weekday_bootstrap(corrected, base, test["target"], keep),
    }


def _fmt(value: float) -> str:
    return f"{float(value):.4f}"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# 新种子上的工作日核对",
        "",
        "种子 45 和 46。周末不计入。协议见 [CELL_AFFINE_SEEDCHECK_PLAN.md](../CELL_AFFINE_SEEDCHECK_PLAN.md)。",
        "",
        "## 核对",
        "",
    ]
    if payload["failures"]:
        lines.append("训练日志没有对上，下面的通过与否不采用。")
        lines.extend(f"- {item}" for item in payload["failures"])
    else:
        lines.append("四个新检查点的全市测试 MAE 对上了各自的训练日志。")
    lines.extend(["", "## 判定", ""])
    if payload["failures"]:
        lines.append("未判定。")
    elif payload["passed"]:
        lines.append("四个新任务的工作日测试 MAE 都低于旧公式 λ=72。")
    else:
        lines.append("四个新任务没有都更低。")
    lines.extend(
        [
            "",
            "## 工作日测试 MAE",
            "",
            "| 臂 | 纽约 45 | 纽约 46 | 芝加哥 45 | 芝加哥 46 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    if payload["results"]:
        for arm, label in (("B0", "一律微调"), ("O72", "旧公式 λ=72"), ("O72H", "旧公式再加钟点偏差")):
            cells = [_fmt(payload["results"][f"{source}-seed{seed}"]["weekday"][arm]) for source, seed in TASKS]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.extend(["", "## 相对旧公式的工作日差", "", "差是本方法减旧公式。负值表示本方法更低。区间不作为通过条件。", ""])
        lines.append("| 任务 | 差 | 2.5% | 97.5% | 工作日天数 |")
        lines.append("|---|---:|---:|---:|---:|")
        for source, seed in TASKS:
            diff = payload["results"][f"{source}-seed{seed}"]["diff"]
            lines.append(
                f"| {CITY[source]} {seed} | {diff['point']:.4f} | {diff['low']:.4f} | {diff['high']:.4f} | {diff['n_days']} |"
            )
    lines.extend(
        [
            "",
            "## 局限",
            "",
            "这次不看周末。上一份检查里，周末六个旧任务都比旧公式高。",
            "新种子的测试小时同样不是为了这次公式事先锁住的未见数据。",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    processed_dir: Path,
    checkpoint_dir: Path,
    log_dir: Path,
    cache_dir: Path,
    reference_train: Path,
    out_json: Path,
    out_md: Path,
) -> int:
    with np.load(reference_train) as blob:
        reference_mask = np.asarray(blob["eligible"], dtype=bool)
    if int(reference_mask.sum()) != N_GATE:
        raise RuntimeError("reference gate is not 50 cells")
    missing = [str(path) for source, seed in TASKS if not (path := _checkpoint(checkpoint_dir, source, seed)).is_file()]
    if missing:
        raise FileNotFoundError("missing checkpoints: " + ", ".join(missing))
    if any(not _cache(cache_dir, source, seed, split).is_file() for source, seed in TASKS for split in ("train", "test")):
        export_predictions(processed_dir, checkpoint_dir, cache_dir, reference_mask)
    v2 = _load_script("cell_affine_v2_seedcheck_score", "19_cell_affine_v2.py")
    previous = _load_script("cell_affine_next_seedcheck", "23_cell_affine_next.py")
    failures: list[str] = []
    results: dict[str, Any] = {}
    for source, seed in TASKS:
        train = _load_split(cache_dir, source, seed, "train")
        test = _load_split(cache_dir, source, seed, "test")
        if train["prediction"].shape != (N_HOURS, N_CELLS) or test["prediction"].shape != (TEST_HOURS, N_CELLS):
            raise RuntimeError(f"{source} seed {seed} cache shape is unexpected")
        if not np.array_equal(train["eligible"], reference_mask):
            raise RuntimeError(f"{source} seed {seed} gate mask changed")
        record = _score_task(train, test, v2.solve_old, previous.hour_offsets)
        log_path = log_dir / LOG_NAME[(source, seed)]
        logged = json.loads(log_path.read_text(encoding="utf-8"))
        if abs(record["city_mae"] - float(logged["test"]["mae"])) > LOG_ATOL:
            failures.append(
                f"{source} seed {seed} city MAE {record['city_mae']:.6f} != log {float(logged['test']['mae']):.6f}"
            )
        results[f"{source}-seed{seed}"] = record
        print(f"scored {source} seed {seed}", flush=True)
    passed = not failures and all(
        results[f"{source}-seed{seed}"]["weekday"]["O72H"] < results[f"{source}-seed{seed}"]["weekday"]["O72"]
        for source, seed in TASKS
    )
    payload = {"failures": failures, "passed": passed, "results": results}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    text = _markdown(payload)
    out_md.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--reference-train", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=ROOT / "docs" / "profiles" / "cell_affine_seedcheck.json")
    parser.add_argument("--out-md", type=Path, default=ROOT / "docs" / "profiles" / "cell_affine_seedcheck.md")
    args = parser.parse_args()
    return run(
        args.processed_dir,
        args.checkpoint_dir,
        args.log_dir,
        args.cache_dir,
        args.reference_train,
        args.out_json,
        args.out_md,
    )


if __name__ == "__main__":
    raise SystemExit(main())
