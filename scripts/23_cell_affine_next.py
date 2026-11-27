#!/usr/bin/env python3
"""Hour bias, L1 affine, median intercept, and a residual gate.

Protocol: docs/CELL_AFFINE_NEXT_PLAN.md. No training and no tuning on val/test.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
DAY_HOURS = 24
HOUR_KAPPA = 3.0
GATE_BIAS = 0.5
LAD_ITERS = 40
LAD_EPS = 1e-3
BOOT_REPS = 2000
BOOT_SEED = 0
ANCHOR_ATOL = 1e-5
OTHER_ATOL = 1e-4
SOURCES = ("ny", "chi")
SEEDS = (42, 43, 44)
CITY = {"ny": "纽约", "chi": "芝加哥"}
ARM_ORDER = ("B0", "O72", "HOD", "O72H", "LAD", "MED", "GATE")
ARM_LABEL = {
    "B0": "一律微调",
    "O72": "旧公式 λ=72",
    "HOD": "钟点偏差，缩一半",
    "O72H": "旧公式再加钟点偏差",
    "LAD": "L1 仿射，λ=72",
    "MED": "旧斜率，中位数截距",
    "GATE": "只校正平均残差 > 0.5",
}
NEW_ARMS = ("HOD", "O72H", "LAD", "MED", "GATE")
ANCHOR = {("ny", 42): 1.7054192513504478, ("chi", 42): 1.4415295244856239}
OTHER_ANCHOR = {("ny", 43): 1.6272, ("ny", 44): 1.6175, ("chi", 43): 1.5051, ("chi", 44): 1.5064}


def _load_v2():
    path = ROOT / "scripts" / "19_cell_affine_v2.py"
    spec = importlib.util.spec_from_file_location("cell_affine_v2_next", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load scripts/19_cell_affine_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def solve_lad_ridge(prediction: np.ndarray, target: np.ndarray, lam: float, solve_old) -> tuple[float, float]:
    """IRLS absolute-error line with the same ridge center as the published formula."""
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred.shape != truth.shape or pred.size == 0:
        raise ValueError("prediction and target must be the same non-empty length")
    if lam < 0:
        raise ValueError("lambda must be non-negative")
    mult = float(lam) / float(pred.size)
    scale, shift = solve_old(pred, truth, mult)
    for _ in range(LAD_ITERS):
        error = truth - scale * pred - shift
        weight = 1.0 / np.maximum(np.abs(error), LAD_EPS)
        gram = np.array(
            [
                [float(weight @ (pred * pred)) + lam, float(weight @ pred)],
                [float(weight @ pred), float(weight.sum()) + lam],
            ],
            dtype=np.float64,
        )
        rhs = np.array([float(weight @ (pred * truth)) + lam, float(weight @ truth)], dtype=np.float64)
        scale, shift = (float(item) for item in np.linalg.solve(gram, rhs))
    return scale, shift


def hour_offsets(residual: np.ndarray, kappa: float = HOUR_KAPPA) -> np.ndarray:
    """Shrink each clock-hour mean toward zero. residual is one cell, length divisible by 24."""
    resid = np.asarray(residual, dtype=np.float64).reshape(-1)
    if resid.size % DAY_HOURS:
        raise ValueError("residual length must be a whole number of days")
    folded = resid.reshape(-1, DAY_HOURS)
    return folded.sum(axis=0) / (folded.shape[0] + float(kappa))


def median_intercept(prediction: np.ndarray, target: np.ndarray, scale: float) -> float:
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    return float(np.median(truth - scale * pred))


def _cache(cache_dir: Path, source: str, seed: int, split: str) -> Path:
    return cache_dir / f"{source}_seed{seed}_{split}.npz"


def _load_split(cache_dir: Path, source: str, seed: int, split: str) -> dict[str, Any]:
    path = _cache(cache_dir, source, seed, split)
    with np.load(path) as blob:
        loaded = {
            "prediction": np.asarray(blob["prediction"], dtype=np.float64),
            "target": np.asarray(blob["target"], dtype=np.float64),
            "value_scale": float(blob["value_scale"]),
        }
        if split == "train":
            loaded["eligible"] = np.asarray(blob["eligible"], dtype=bool)
    return loaded


def _score_arrays(prediction: np.ndarray, target: np.ndarray, value_scale: float) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
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


def _apply_affine(prediction: np.ndarray, scale: np.ndarray, shift: np.ndarray) -> np.ndarray:
    return prediction * scale.reshape(1, -1) + shift.reshape(1, -1)


def _apply_hour(prediction: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    clock = np.arange(prediction.shape[0]) % DAY_HOURS
    return prediction + offsets[clock, :]


def _bootstrap(arm: np.ndarray, ref: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    hours, cells = target.shape
    days = hours // DAY_HOURS
    arm_day = np.abs(arm - target).reshape(days, DAY_HOURS, cells).sum(axis=(1, 2))
    ref_day = np.abs(ref - target).reshape(days, DAY_HOURS, cells).sum(axis=(1, 2))
    drawn = np.random.default_rng(BOOT_SEED).integers(0, days, size=(BOOT_REPS, days))
    diff = (arm_day[drawn].sum(axis=1) - ref_day[drawn].sum(axis=1)) / (days * DAY_HOURS * cells)
    point = float((arm_day.sum() - ref_day.sum()) / (days * DAY_HOURS * cells))
    return point, float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))


def _fit_task(prediction: np.ndarray, target: np.ndarray, eligible: np.ndarray, solve_old) -> dict[str, Any]:
    n_cells = prediction.shape[1]
    scale = np.ones(n_cells, dtype=np.float64)
    shift = np.zeros(n_cells, dtype=np.float64)
    lad_scale = np.ones(n_cells, dtype=np.float64)
    lad_shift = np.zeros(n_cells, dtype=np.float64)
    med_scale = np.ones(n_cells, dtype=np.float64)
    med_shift = np.zeros(n_cells, dtype=np.float64)
    gate_scale = np.ones(n_cells, dtype=np.float64)
    gate_shift = np.zeros(n_cells, dtype=np.float64)
    hod = np.zeros((DAY_HOURS, n_cells), dtype=np.float64)
    o72h = np.zeros((DAY_HOURS, n_cells), dtype=np.float64)
    gated = []
    for index in np.flatnonzero(eligible):
        pred = prediction[:, index]
        truth = target[:, index]
        found_a, found_b = solve_old(pred, truth, 1.0)
        scale[index], shift[index] = found_a, found_b
        lad_scale[index], lad_shift[index] = solve_lad_ridge(pred, truth, float(N_HOURS), solve_old)
        med_scale[index] = found_a
        med_shift[index] = median_intercept(pred, truth, found_a)
        residual = truth - pred
        hod[:, index] = hour_offsets(residual)
        o72h[:, index] = hour_offsets(truth - found_a * pred - found_b)
        if float(residual.mean()) > GATE_BIAS:
            gate_scale[index], gate_shift[index] = found_a, found_b
            gated.append(int(index))
    return {
        "O72": (scale, shift, np.zeros_like(hod)),
        "HOD": (np.ones(n_cells), np.zeros(n_cells), hod),
        "O72H": (scale, shift, o72h),
        "LAD": (lad_scale, lad_shift, np.zeros_like(hod)),
        "MED": (med_scale, med_shift, np.zeros_like(hod)),
        "GATE": (gate_scale, gate_shift, np.zeros_like(hod)),
        "B0": (np.ones(n_cells), np.zeros(n_cells), np.zeros_like(hod)),
        "n_gate_bias": len(gated),
    }


def _predict(prediction: np.ndarray, pack: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, shift, offsets = pack
    return _apply_hour(_apply_affine(prediction, scale, shift), offsets)


def _fmt(value: float) -> str:
    return f"{float(value):.4f}"


def _markdown(results: list[dict[str, Any]], failures: list[str]) -> str:
    lines = [
        "# 钟点偏差、L1 仿射、中位数截距和残差门",
        "",
        "没有新训练，参数都在打分前固定。协议见 [CELL_AFFINE_NEXT_PLAN.md](../CELL_AFFINE_NEXT_PLAN.md)。",
        "测试小时以前已经看过，这里是探索性对照。",
        "",
        "## 核对",
        "",
    ]
    if failures:
        lines.append("锚点没有对上，下面不采用胜出说法。")
        lines.extend(f"- {item}" for item in failures)
    else:
        lines.append("旧公式 λ=72 的测试 MAE 对上了已发表锚点。")
    lines.extend(["", "## 判定", ""])
    if failures:
        lines.append("核对未通过。")
    else:
        for arm in NEW_ARMS:
            lowers = []
            intervals = {}
            for source in SOURCES:
                for seed in SEEDS:
                    found = next(item for item in results if item["source"] == source and item["seed"] == seed)
                    gap = found["test"][arm]["mae"] - found["test"]["O72"]["mae"]
                    lowers.append(gap < -1e-12)
                intervals[source] = next(item for item in results if item["source"] == source and item["seed"] == 42)["diff"][arm]
            passed = all(lowers) and all(intervals[source]["high"] < 0.0 for source in SOURCES)
            word = "超过旧公式 λ=72" if passed else "没有超过旧公式 λ=72"
            lines.append(f"- {ARM_LABEL[arm]}：{word}。六个任务都更低：{'是' if all(lowers) else '否'}。")
        lines.append("")
        lines.append("残差门实际校正的格子数（50 个过门格里平均残差大于 0.5 的）：")
        for item in results:
            lines.append(f"- {CITY[item['source']]} 种子 {item['seed']}：{item['n_gate_bias']} 格")
    lines.extend(["", "## 测试 MAE", ""])
    header = "| 臂 | " + " | ".join(f"{CITY[source]} {seed}" for source in SOURCES for seed in SEEDS) + " |"
    rule = "|---|" + "---:|" * 6
    lines.extend([header, rule])
    for arm in ARM_ORDER:
        cells = []
        for source in SOURCES:
            for seed in SEEDS:
                found = next(item for item in results if item["source"] == source and item["seed"] == seed)
                cells.append(_fmt(found["test"][arm]["mae"]))
        lines.append("| " + ARM_LABEL[arm] + " | " + " | ".join(cells) + " |")
    lines.extend(["", "## 测试 RMSE", "", header, rule])
    for arm in ARM_ORDER:
        cells = []
        for source in SOURCES:
            for seed in SEEDS:
                found = next(item for item in results if item["source"] == source and item["seed"] == seed)
                cells.append(_fmt(found["test"][arm]["rmse"]))
        lines.append("| " + ARM_LABEL[arm] + " | " + " | ".join(cells) + " |")
    lines.extend(["", "## 相对旧公式的测试 MAE 差", "", "差是该臂减 λ=72。负值表示该臂更低。", ""])
    for source in SOURCES:
        for seed in SEEDS:
            found = next(item for item in results if item["source"] == source and item["seed"] == seed)
            lines.extend([f"### {CITY[source]}，种子 {seed}", "", "| 臂 | 差 | 2.5% | 97.5% |", "|---|---:|---:|---:|"])
            for arm in NEW_ARMS:
                diff = found["diff"][arm]
                lines.append(f"| {ARM_LABEL[arm]} | {_fmt(diff['point'])} | {_fmt(diff['low'])} | {_fmt(diff['high'])} |")
            lines.append("")
    for split, title in (("train", "训练窗 MAE"), ("val", "验证集 MAE")):
        lines.extend([f"## {title}", "", header, rule])
        for arm in ARM_ORDER:
            cells = []
            for source in SOURCES:
                for seed in SEEDS:
                    found = next(item for item in results if item["source"] == source and item["seed"] == seed)
                    cells.append(_fmt(found[split][arm]["mae"]))
            lines.append("| " + ARM_LABEL[arm] + " | " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend(
        [
            "## 局限",
            "",
            "钟点偏差每个钟点只有 3 个训练残差，κ=3 是事先定的一半收缩，不是交叉验证选出来的。",
            "中位数截距不再惩罚 b。残差门的 0.5 也没有搜索。",
            "测试段不是未见数据。",
            "",
        ]
    )
    return "\n".join(lines)


def run(cache_dir: Path, out_json: Path, out_md: Path) -> int:
    v2 = _load_v2()
    checks = json.loads((cache_dir / "checks.json").read_text(encoding="utf-8"))
    if not checks.get("passed"):
        raise RuntimeError("prediction checks did not pass")
    reference = None
    results = []
    for source in SOURCES:
        for seed in SEEDS:
            train = _load_split(cache_dir, source, seed, "train")
            if train["prediction"].shape != (N_HOURS, N_CELLS) or int(train["eligible"].sum()) != N_GATE:
                raise RuntimeError(f"{source} seed {seed} cache shape is unexpected")
            if reference is None:
                reference = train["eligible"].copy()
            elif not np.array_equal(reference, train["eligible"]):
                raise RuntimeError("gate mask differs across tasks")
            print(f"fit {source} seed {seed}", flush=True)
            fitted = _fit_task(train["prediction"], train["target"], train["eligible"], v2.solve_old)
            record: dict[str, Any] = {"source": source, "seed": seed, "n_gate_bias": fitted["n_gate_bias"], "diff": {}}
            test_pred = {}
            test_target = None
            for split in ("train", "val", "test"):
                loaded = _load_split(cache_dir, source, seed, split)
                rows = {}
                for arm in ARM_ORDER:
                    corrected = _predict(loaded["prediction"], fitted[arm])
                    rows[arm] = _score_arrays(corrected, loaded["target"], loaded["value_scale"])
                    if split == "test":
                        test_pred[arm] = corrected
                        test_target = loaded["target"]
                record[split] = rows
            assert test_target is not None
            for arm in NEW_ARMS:
                point, low, high = _bootstrap(test_pred[arm], test_pred["O72"], test_target)
                record["diff"][arm] = {"point": point, "low": low, "high": high}
            results.append(record)
    failures = []
    for (source, seed), expected in ANCHOR.items():
        found = next(item for item in results if item["source"] == source and item["seed"] == seed)
        got = found["test"]["O72"]["mae"]
        if abs(got - expected) > ANCHOR_ATOL:
            failures.append(f"{source} seed {seed} {got:.10f} != {expected:.10f}")
    for (source, seed), expected in OTHER_ANCHOR.items():
        found = next(item for item in results if item["source"] == source and item["seed"] == seed)
        got = found["test"]["O72"]["mae"]
        if abs(got - expected) > OTHER_ATOL:
            failures.append(f"{source} seed {seed} {got:.6f} !~ {expected:.4f}")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"anchors_ok": not failures, "failures": failures, "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    text = _markdown(results, failures)
    out_md.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=ROOT / "docs" / "profiles" / "cell_affine_next.json")
    parser.add_argument("--out-md", type=Path, default=ROOT / "docs" / "profiles" / "cell_affine_next.md")
    args = parser.parse_args()
    return run(args.cache_dir, args.out_json, args.out_md)


if __name__ == "__main__":
    raise SystemExit(main())
