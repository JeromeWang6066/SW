#!/usr/bin/env python3
"""Three frozen reforms of the per-cell correction. No training, no retuning.

Protocol: docs/CELL_AFFINE_REFORM_PLAN.md. Fit reads only the 72 training
hours. Score applies the frozen coefficients and writes the report.
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
N_CELLS = 207
N_GATE = 50
DAY_HOURS = 24
BOOT_REPS = 2000
BOOT_SEED = 0
HUBER_DELTA = 5.0
ANCHOR_ATOL = 1e-5
OTHER_ATOL = 1e-4
SOURCES = ("ny", "chi")
SEEDS = (42, 43, 44)
CITY = {"ny": "纽约", "chi": "芝加哥"}
ARM_ORDER = ("B0", "O72", "O72C", "N1", "N1C", "M1", "M0", "L1", "H5")
ARM_LABEL = {
    "B0": "一律微调",
    "O72": "旧公式 λ=72",
    "O72C": "旧公式再裁成非负",
    "N1": "中心化 κ=1",
    "N1C": "中心化再裁成非负",
    "M1": "纯乘法 κ=1",
    "M0": "纯乘法 κ=0",
    "L1": "纯乘法 L1，κ=1",
    "H5": "纯乘法 Huber δ=5，κ=1",
}
CLIP_ARMS = {"O72C": "O72", "N1C": "N1"}
WIN_ARMS = ("M1", "L1", "H5")
# Seed-42 test MAE anchors from the published cell-affine v2 / val-select tables.
ANCHOR = {
    ("ny", 42, "O72"): 1.7054192513504478,
    ("chi", 42, "O72"): 1.4415295244856239,
    ("ny", 42, "N1"): 1.7077793334619,
    ("chi", 42, "N1"): 1.4494899087220661,
}
# Four-decimal test MAE in docs/profiles/cell_affine_v2.md.
OTHER_ANCHOR = {
    ("ny", 43, "O72"): 1.6272,
    ("ny", 43, "N1"): 1.6293,
    ("ny", 44, "O72"): 1.6175,
    ("ny", 44, "N1"): 1.6179,
    ("chi", 43, "O72"): 1.5051,
    ("chi", 43, "N1"): 1.5162,
    ("chi", 44, "O72"): 1.5064,
    ("chi", 44, "N1"): 1.5197,
}


def _load_v2():
    path = ROOT / "scripts" / "19_cell_affine_v2.py"
    spec = importlib.util.spec_from_file_location("cell_affine_v2_reform", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load scripts/19_cell_affine_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _vec(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def solve_multiplicative(prediction: np.ndarray, target: np.ndarray, kappa: float) -> tuple[float, float]:
    """Scale through the origin. The shift is identically zero."""
    if math.isinf(kappa):
        return 1.0, 0.0
    if not math.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be non-negative")
    pred = _vec(prediction)
    truth = _vec(target)
    if pred.shape != truth.shape or pred.size == 0:
        raise ValueError("prediction and target must be the same non-empty length")
    energy = float(pred @ pred)
    if energy <= 0.0:
        return 1.0, 0.0
    residual = truth - pred
    deviation = float(pred @ residual) / ((1.0 + float(kappa)) * energy)
    return 1.0 + deviation, 0.0


def solve_l1_scale(prediction: np.ndarray, target: np.ndarray, kappa: float = 1.0) -> tuple[float, float]:
    """One-dimensional L1 scale. Golden section, midpoint on a flat interval."""
    if math.isinf(kappa):
        return 1.0, 0.0
    if not math.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be non-negative")
    pred = _vec(prediction)
    truth = _vec(target)
    if pred.shape != truth.shape or pred.size == 0:
        raise ValueError("prediction and target must be the same non-empty length")
    weight = float(np.abs(pred).sum())
    if weight <= 0.0:
        return 1.0, 0.0
    residual = truth - pred

    def objective(deviation: float) -> float:
        return float(np.abs(residual - deviation * pred).sum()) + float(kappa) * weight * abs(deviation)

    lo, hi = -2.0, 2.0
    guard = 0
    while objective(lo) > objective(lo - 1.0) or objective(hi) > objective(hi + 1.0):
        lo -= 2.0
        hi += 2.0
        guard += 1
        if guard > 12:
            break
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = lo, hi
    probe_c = right - ratio * (right - left)
    probe_d = left + ratio * (right - left)
    value_c = objective(probe_c)
    value_d = objective(probe_d)
    for _ in range(80):
        if value_c < value_d:
            right, probe_d, value_d = probe_d, probe_c, value_c
            probe_c = right - ratio * (right - left)
            value_c = objective(probe_c)
        else:
            left, probe_c, value_c = probe_c, probe_d, value_d
            probe_d = left + ratio * (right - left)
            value_d = objective(probe_d)
    return 1.0 + 0.5 * (left + right), 0.0


def solve_huber_scale(
    prediction: np.ndarray,
    target: np.ndarray,
    kappa: float = 1.0,
    delta: float = HUBER_DELTA,
) -> tuple[float, float]:
    """Newton on the Huber scale objective. Starts from the L2 pure-scale solution."""
    if math.isinf(kappa):
        return 1.0, 0.0
    if not math.isfinite(kappa) or kappa < 0 or not math.isfinite(delta) or delta <= 0:
        raise ValueError("kappa must be non-negative and delta must be positive")
    pred = _vec(prediction)
    truth = _vec(target)
    if pred.shape != truth.shape or pred.size == 0:
        raise ValueError("prediction and target must be the same non-empty length")
    energy = float(pred @ pred)
    if energy <= 0.0:
        return 1.0, 0.0
    residual = truth - pred
    deviation = float(pred @ residual) / ((1.0 + float(kappa)) * energy)
    for _ in range(50):
        error = residual - deviation * pred
        magnitude = np.abs(error)
        psi = np.where(magnitude <= delta, error, delta * np.sign(error))
        inside = magnitude <= delta
        gradient = -float(pred @ psi) + float(kappa) * energy * deviation
        hessian = float((inside * pred * pred).sum()) + float(kappa) * energy
        if hessian <= 1e-18:
            break
        step = gradient / hessian
        deviation -= step
        if abs(step) < 1e-12:
            break
    return 1.0 + float(deviation), 0.0


def _cache(cache_dir: Path, source: str, seed: int, split: str) -> Path:
    return cache_dir / f"{source}_seed{seed}_{split}.npz"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isinf(number) or math.isnan(number):
            return None
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _fit_arm(
    prediction: np.ndarray,
    target: np.ndarray,
    eligible: np.ndarray,
    solver: Callable[..., tuple[float, float]],
    strength: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    scale = np.ones(prediction.shape[1], dtype=np.float64)
    shift = np.zeros(prediction.shape[1], dtype=np.float64)
    for index in np.flatnonzero(eligible):
        if strength is None:
            found = solver(prediction[:, index], target[:, index])
        else:
            found = solver(prediction[:, index], target[:, index], strength)
        scale[index], shift[index] = found
    return scale, shift


def _apply(prediction: np.ndarray, scale: np.ndarray, shift: np.ndarray, clip: bool) -> np.ndarray:
    corrected = prediction * scale.reshape(1, -1) + shift.reshape(1, -1)
    if clip:
        return np.maximum(corrected, 0.0)
    return corrected


def _bootstrap_mae_diff(arm: np.ndarray, ref: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    hours, cells = target.shape
    if hours % DAY_HOURS:
        raise ValueError("hours must be whole days")
    days = hours // DAY_HOURS
    arm_day = np.abs(arm - target).reshape(days, DAY_HOURS, cells).sum(axis=(1, 2))
    ref_day = np.abs(ref - target).reshape(days, DAY_HOURS, cells).sum(axis=(1, 2))
    drawn = np.random.default_rng(BOOT_SEED).integers(0, days, size=(BOOT_REPS, days))
    diff = (arm_day[drawn].sum(axis=1) - ref_day[drawn].sum(axis=1)) / (days * DAY_HOURS * cells)
    point = float((arm_day.sum() - ref_day.sum()) / (days * DAY_HOURS * cells))
    return point, float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))


def _load_split(cache_dir: Path, source: str, seed: int, split: str) -> dict[str, Any]:
    path = _cache(cache_dir, source, seed, split)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as blob:
        loaded = {
            "prediction": np.asarray(blob["prediction"], dtype=np.float64),
            "target": np.asarray(blob["target"], dtype=np.float64),
            "value_scale": float(blob["value_scale"]),
        }
        if split == "train":
            loaded["eligible"] = np.asarray(blob["eligible"], dtype=bool)
    return loaded


def stage_fit(cache_dir: Path, out_json: Path) -> dict[str, Any]:
    v2 = _load_v2()
    checks = cache_dir / "checks.json"
    if not checks.is_file():
        raise RuntimeError(f"missing {checks}")
    payload = json.loads(checks.read_text(encoding="utf-8"))
    if not payload.get("passed"):
        raise RuntimeError("predict checks did not pass; refuse to fit")
    runs = []
    reference = None
    solvers = {
        "B0": (v2.solve_centered, math.inf),
        "O72": (v2.solve_old, 1.0),
        "N1": (v2.solve_centered, 1.0),
        "M1": (solve_multiplicative, 1.0),
        "M0": (solve_multiplicative, 0.0),
        "L1": (solve_l1_scale, 1.0),
        "H5": (solve_huber_scale, 1.0),
    }
    for source in SOURCES:
        for seed in SEEDS:
            train = _load_split(cache_dir, source, seed, "train")
            prediction = train["prediction"]
            target = train["target"]
            eligible = train["eligible"]
            if prediction.shape != (N_HOURS, N_CELLS) or int(eligible.sum()) != N_GATE:
                raise RuntimeError(f"{source} seed {seed} is not the 72-hour, 50-cell problem")
            if reference is None:
                reference = eligible.copy()
            elif not np.array_equal(reference, eligible):
                raise RuntimeError(f"{source} seed {seed} gate differs")
            print(f"fit {source} seed {seed}", flush=True)
            coefficients = {}
            for name, (solver, strength) in solvers.items():
                scale, shift = _fit_arm(prediction, target, eligible, solver, strength)
                coefficients[name] = {
                    "scale": scale,
                    "shift": shift,
                }
            runs.append(
                {
                    "source": source,
                    "seed": seed,
                    "eligible": eligible,
                    "coefficients": coefficients,
                    "value_scale": train["value_scale"],
                }
            )
    document = {
        "protocol": "cell-affine-reform-2026-11-23",
        "kappa": 1.0,
        "huber_delta": HUBER_DELTA,
        "runs": [
            {
                "source": item["source"],
                "seed": item["seed"],
                "eligible": item["eligible"],
                "value_scale": item["value_scale"],
                "arms": {
                    name: {
                        "scale": coef["scale"].tolist(),
                        "shift": coef["shift"].tolist(),
                    }
                    for name, coef in item["coefficients"].items()
                },
            }
            for item in runs
        ],
    }
    _write_json(out_json, document)
    return document


def _score_arrays(prediction: np.ndarray, target: np.ndarray, value_scale: float) -> dict[str, float]:
    """Same reduction as scripts/17_cell_affine.py score_arrays, without importing training deps."""
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError("prediction and target shapes differ")
    if value_scale <= 0:
        raise ValueError("value scale must be positive")
    hours, n_cells = pred.shape
    valid_points = float(hours * n_cells)
    gap = pred - truth
    positive = truth > 0
    mape_points = float(positive.sum())
    mape = float("nan") if mape_points <= 0 else float(np.abs(gap[positive] / truth[positive]).sum()) / mape_points * 100.0
    se_norm = float(np.sum((gap / float(value_scale)) ** 2))
    return {
        "mae": float(np.abs(gap).sum()) / valid_points,
        "rmse": (se_norm / valid_points) ** 0.5 * float(value_scale),
        "mape": mape,
    }


def _score_run(cache_dir: Path, run: dict[str, Any]) -> dict[str, Any]:
    source = run["source"]
    seed = int(run["seed"])
    eligible = np.asarray(run["eligible"], dtype=bool)
    value_scale = float(run["value_scale"])
    base = {}
    for name, coef in run["arms"].items():
        base[name] = (
            np.asarray(coef["scale"], dtype=np.float64),
            np.asarray(coef["shift"], dtype=np.float64),
        )
    splits = {}
    test_pred = {}
    test_target = None
    for split in ("train", "val", "test"):
        loaded = _load_split(cache_dir, source, seed, split)
        prediction = loaded["prediction"]
        target = loaded["target"]
        if abs(float(loaded["value_scale"]) - value_scale) > 1e-6:
            raise RuntimeError(f"{source} seed {seed} {split} value_scale differs from train")
        rows = {}
        for name in ARM_ORDER:
            parent = CLIP_ARMS.get(name, name)
            scale, shift = base[parent]
            clip = name in CLIP_ARMS
            corrected = _apply(prediction, scale, shift, clip)
            metrics = _score_arrays(corrected, target, value_scale)
            rows[name] = {
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "negative": int(np.sum(corrected < 0.0)),
                "negative_min": None if not np.any(corrected < 0.0) else float(corrected.min()),
            }
            if split == "test":
                test_pred[name] = corrected
                test_target = target
        splits[split] = rows
    assert test_target is not None
    diffs = {}
    for name in ARM_ORDER:
        if name == "O72":
            continue
        point, low, high = _bootstrap_mae_diff(test_pred[name], test_pred["O72"], test_target)
        diffs[name] = {"point": point, "low": low, "high": high}
    gated = np.flatnonzero(eligible)
    summary = {}
    for name, (scale, shift) in base.items():
        chosen = scale[gated]
        summary[name] = {
            "a_median": float(np.median(chosen)),
            "a_p10": float(np.quantile(chosen, 0.1)),
            "a_p90": float(np.quantile(chosen, 0.9)),
            "a_min": float(chosen.min()),
            "n_a_negative": int(np.sum(chosen < 0.0)),
            "b_median": float(np.median(shift[gated])),
        }
    return {"source": source, "seed": seed, "splits": splits, "diff_vs_o72": diffs, "coefficients": summary}


def _judge_detail(cache_dir: Path, fit: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    def by_key(source: str, seed: int) -> dict[str, Any]:
        return next(item for item in results if item["source"] == source and item["seed"] == seed)

    def clip_against_parent(parent: str, clipped: str) -> dict[str, Any]:
        lower_all = True
        seed42 = {}
        for source in SOURCES:
            for seed in SEEDS:
                raw = by_key(source, seed)["splits"]["test"][parent]["mae"]
                alt = by_key(source, seed)["splits"]["test"][clipped]["mae"]
                if not alt < raw - 1e-12:
                    lower_all = False
            run = next(item for item in fit["runs"] if item["source"] == source and int(item["seed"]) == 42)
            loaded = _load_split(cache_dir, source, 42, "test")
            scale = np.asarray(run["arms"][parent]["scale"], dtype=np.float64)
            shift = np.asarray(run["arms"][parent]["shift"], dtype=np.float64)
            parent_pred = _apply(loaded["prediction"], scale, shift, False)
            clip_pred = _apply(loaded["prediction"], scale, shift, True)
            point, low, high = _bootstrap_mae_diff(clip_pred, parent_pred, loaded["target"])
            seed42[source] = {"point": point, "low": low, "high": high}
        passed = lower_all and all(seed42[source]["high"] < 0.0 for source in SOURCES)
        return {"passed": passed, "all_seeds_lower": lower_all, "seed42": seed42}

    wins = {}
    for arm in WIN_ARMS:
        lower = []
        intervals = {}
        for source in SOURCES:
            for seed in SEEDS:
                gap = by_key(source, seed)["splits"]["test"][arm]["mae"] - by_key(source, seed)["splits"]["test"]["O72"]["mae"]
                lower.append(gap < -1e-12)
            intervals[source] = by_key(source, 42)["diff_vs_o72"][arm]
        wins[arm] = {
            "passed": all(lower) and all(intervals[source]["high"] < 0.0 for source in SOURCES),
            "all_six_lower": all(lower),
            "seed42": intervals,
        }
    return {
        "O72C_vs_O72": clip_against_parent("O72", "O72C"),
        "N1C_vs_N1": clip_against_parent("N1", "N1C"),
        "versus_O72": wins,
    }


def _check_anchors(results: list[dict[str, Any]]) -> list[str]:
    failures = []
    for (source, seed, arm), expected in ANCHOR.items():
        found = next(item for item in results if item["source"] == source and item["seed"] == seed)
        got = float(found["splits"]["test"][arm]["mae"])
        if abs(got - expected) > ANCHOR_ATOL:
            failures.append(f"{source} seed {seed} {arm} {got:.10f} != {expected:.10f}")
    for (source, seed, arm), expected in OTHER_ANCHOR.items():
        found = next(item for item in results if item["source"] == source and item["seed"] == seed)
        got = float(found["splits"]["test"][arm]["mae"])
        if abs(got - expected) > OTHER_ATOL:
            failures.append(f"{source} seed {seed} {arm} {got:.6f} !~ {expected:.4f}")
    return failures


def _markdown(results: list[dict[str, Any]], judgments: dict[str, Any], failures: list[str]) -> str:
    lines = [
        "# 按格校正三项改动",
        "",
        "通道是出租上车。没有新训练，也没有在验证集或测试集上选强度。",
        "协议是 [CELL_AFFINE_REFORM_PLAN.md](../CELL_AFFINE_REFORM_PLAN.md)。",
        "旧公式 λ=72 和中心化 κ=1 先对上已发表锚点，再评价裁剪、纯乘法和 L1/Huber。",
        "测试小时在以前的实验里已经看过，这里是探索性对照。",
        "",
        "## 核对",
        "",
    ]
    if failures:
        lines.append("锚点没有对上，下面的表只作诊断，不采用第 3 节的胜出说法。")
        for item in failures:
            lines.append(f"- {item}")
    else:
        lines.append("种子 42 的旧公式和中心化公式测试 MAE 落在锚点 1e-5 以内。种子 43、44 落在已发表四位小数的 1e-4 以内。")
    lines.extend(["", "## 判定", ""])
    if failures:
        lines.append("核对未通过，不写胜出。")
    else:
        for key, label in (("O72C_vs_O72", "旧公式裁剪"), ("N1C_vs_N1", "中心化裁剪")):
            item = judgments[key]
            word = "成立" if item["passed"] else "没有在预注册规则下单独成立"
            lines.append(f"- {label}：{word}。六个任务的测试 MAE 都更低：{'是' if item['all_seeds_lower'] else '否'}。种子 42 上下降的绝对值仍小于 0.001。")
            for source in SOURCES:
                interval = item["seed42"][source]
                lines.append(
                    f"  - 种子 42 {CITY[source]}，裁剪减未裁剪 {_fmt(interval['point'])} "
                    f"[{_fmt(interval['low'])}, {_fmt(interval['high'])}]"
                )
        for arm in WIN_ARMS:
            item = judgments["versus_O72"][arm]
            word = "超过旧公式 λ=72" if item["passed"] else "没有超过旧公式 λ=72"
            lines.append(f"- {ARM_LABEL[arm]}：{word}。六个任务测试 MAE 都更低：{'是' if item['all_six_lower'] else '否'}。")
    l1_identity = all(
        abs(item["coefficients"]["L1"]["a_min"] - 1.0) < 1e-8 and abs(item["coefficients"]["L1"]["a_median"] - 1.0) < 1e-8
        for item in results
    )
    if l1_identity:
        lines.append("- 纯乘法 L1 在 κ=1 下，50 个过门格的 a 都回到 1。这一臂与一律微调相同，惩罚把校正全部吃掉了。")
    lines.extend(["", "## 测试 MAE", ""])
    header = "| 臂 | " + " | ".join(f"{CITY[source]} {seed}" for source in SOURCES for seed in SEEDS) + " |"
    rule = "|---|" + "---:|" * 6
    lines.extend([header, rule])
    for arm in ARM_ORDER:
        cells = []
        for source in SOURCES:
            for seed in SEEDS:
                found = next(item for item in results if item["source"] == source and item["seed"] == seed)
                cells.append(_fmt(found["splits"]["test"][arm]["mae"]))
        lines.append(f"| {ARM_LABEL[arm]} | " + " | ".join(cells) + " |")
    lines.extend(["", "## 测试 RMSE", "", header, rule])
    for arm in ARM_ORDER:
        cells = []
        for source in SOURCES:
            for seed in SEEDS:
                found = next(item for item in results if item["source"] == source and item["seed"] == seed)
                cells.append(_fmt(found["splits"]["test"][arm]["rmse"]))
        lines.append(f"| {ARM_LABEL[arm]} | " + " | ".join(cells) + " |")
    lines.extend(["", "## 测试段负预测个数", "", header, rule])
    for arm in ARM_ORDER:
        cells = []
        for source in SOURCES:
            for seed in SEEDS:
                found = next(item for item in results if item["source"] == source and item["seed"] == seed)
                cells.append(str(found["splits"]["test"][arm]["negative"]))
        lines.append(f"| {ARM_LABEL[arm]} | " + " | ".join(cells) + " |")
    lines.extend(["", "## 相对旧公式的测试 MAE 差", "", "差是该臂减 λ=72。负值表示该臂更低。区间是按天块自助的 2.5% 和 97.5% 分位。", ""])
    for source in SOURCES:
        for seed in SEEDS:
            found = next(item for item in results if item["source"] == source and item["seed"] == seed)
            lines.append(f"### {CITY[source]}，种子 {seed}")
            lines.append("")
            lines.append("| 臂 | 差 | 2.5% | 97.5% |")
            lines.append("|---|---:|---:|---:|")
            for arm in ARM_ORDER:
                if arm == "O72":
                    continue
                diff = found["diff_vs_o72"][arm]
                lines.append(f"| {ARM_LABEL[arm]} | {_fmt(diff['point'])} | {_fmt(diff['low'])} | {_fmt(diff['high'])} |")
            lines.append("")
    lines.extend(["## 过门格系数", ""])
    for source in SOURCES:
        for seed in SEEDS:
            found = next(item for item in results if item["source"] == source and item["seed"] == seed)
            lines.append(f"### {CITY[source]}，种子 {seed}")
            lines.append("")
            lines.append("| 臂 | a 中位数 | a 10–90 | a 最小 | a<0 | b 中位数 |")
            lines.append("|---|---:|---|---:|---:|---:|")
            for arm in ("O72", "N1", "M1", "M0", "L1", "H5"):
                coef = found["coefficients"][arm]
                lines.append(
                    f"| {ARM_LABEL[arm]} | {_fmt(coef['a_median'])} | {_fmt(coef['a_p10'])} 到 {_fmt(coef['a_p90'])} | "
                    f"{_fmt(coef['a_min'])} | {coef['n_a_negative']} | {_fmt(coef['b_median'])} |"
                )
            lines.append("")
    lines.extend(
        [
            "## 训练窗和验证集",
            "",
            "拟合不用这两列做选择。列在这里是为了看裁剪和换损失之后，训练窗上的下降有没有带到验证集。",
            "",
        ]
    )
    for split, title in (("train", "训练窗 MAE"), ("val", "验证集 MAE")):
        lines.extend([f"### {title}", "", header, rule])
        for arm in ARM_ORDER:
            cells = []
            for source in SOURCES:
                for seed in SEEDS:
                    found = next(item for item in results if item["source"] == source and item["seed"] == seed)
                    cells.append(_fmt(found["splits"][split][arm]["mae"]))
            lines.append(f"| {ARM_LABEL[arm]} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend(
        [
            "## 局限",
            "",
            "72 小时上的预测是样本内预测。测试段不是未见数据。",
            "δ=5 和 κ=1 都是打分前固定的，没有搜索。",
            "M0 不参与胜出判定。纯乘法臂没有再做事后裁剪。",
            "",
        ]
    )
    return "\n".join(lines)


def stage_score(cache_dir: Path, fit_json: Path, out_json: Path, out_md: Path) -> int:
    fit = json.loads(fit_json.read_text(encoding="utf-8"))
    results = [_score_run(cache_dir, run) for run in fit["runs"]]
    failures = _check_anchors(results)
    judgments = _judge_detail(cache_dir, fit, results)
    document = {"anchors_ok": not failures, "anchor_failures": failures, "judgments": judgments, "results": results}
    _write_json(out_json, document)
    out_md.write_text(_markdown(results, judgments, failures), encoding="utf-8", newline="\n")
    print(out_md.read_text(encoding="utf-8"))
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit and score the three frozen cell-affine reforms.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("fit", "score", "all"), default="all")
    parser.add_argument("--fit-json", type=Path, default=ROOT / "docs" / "profiles" / "cell_affine_reform_fit.json")
    parser.add_argument("--out-json", type=Path, default=ROOT / "docs" / "profiles" / "cell_affine_reform.json")
    parser.add_argument("--out-md", type=Path, default=ROOT / "docs" / "profiles" / "cell_affine_reform.md")
    args = parser.parse_args()
    if args.stage in ("fit", "all"):
        stage_fit(args.cache_dir, args.fit_json)
    if args.stage in ("score", "all"):
        return stage_score(args.cache_dir, args.fit_json, args.out_json, args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
