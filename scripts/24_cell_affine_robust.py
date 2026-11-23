#!/usr/bin/env python3
"""Robustness checks for the published line plus a shrunk hour-of-day bias.

Protocol: docs/CELL_AFFINE_ROBUST_PLAN.md. No training and no retuning.
"""

from __future__ import annotations

import argparse
import datetime as dt
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
DAY_HOURS = 24
TEST_HOURS = 1440
TEST_DAYS = 60
HALF_HOURS = 720
HOUR_KAPPA = 3.0
KAPPA_GRID = (0.0, 1.0, 3.0, 6.0, 12.0, 24.0)
NEIGHBOR_KAPPA = (1.0, 6.0)
MIN_CELLS_BETTER = 30
TOP_CELLS = 5
CONCENTRATION_FLAG = 0.8
N_PERM = 1000
PERM_SEED = 0
PERM_PASS_MAX = 50
BOOT_REPS = 2000
BOOT_SEED = 0
ANCHOR_ATOL = 1e-5
OTHER_ATOL = 1e-4
REPRO_ATOL = 1e-6
WEEK_BLOCK = 7
ORIGIN = dt.date(2016, 1, 1)
TEST_START_HOUR = 7344
SOURCES = ("ny", "chi")
SEEDS = (42, 43, 44)
CITY = {"ny": "纽约", "chi": "芝加哥"}
TASKS = tuple((source, seed) for source in SOURCES for seed in SEEDS)
ANCHOR = {("ny", 42): 1.7054192513504478, ("chi", 42): 1.4415295244856239}
OTHER_ANCHOR = {("ny", 43): 1.6272, ("ny", 44): 1.6175, ("chi", 43): 1.5051, ("chi", 44): 1.5064}


def _load_script(name: str, filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_day_dates(n_hours: int = TEST_HOURS) -> list[dt.date]:
    """Calendar date of each hour in the published test window."""
    if n_hours < 0:
        raise ValueError("hour count must be non-negative")
    start_day = TEST_START_HOUR // DAY_HOURS
    return [ORIGIN + dt.timedelta(days=start_day + (hour // DAY_HOURS)) for hour in range(n_hours)]


def weekday_mask(n_hours: int = TEST_HOURS) -> np.ndarray:
    """True on Monday–Friday. Saturday and Sunday are False."""
    return np.asarray([day.weekday() < 5 for day in test_day_dates(n_hours)], dtype=bool)


def block_lengths(n_days: int, width: int = WEEK_BLOCK) -> list[int]:
    if n_days <= 0 or width <= 0:
        raise ValueError("day count and block width must be positive")
    full, rest = divmod(n_days, width)
    lengths = [width] * full
    if rest:
        lengths.append(rest)
    return lengths


def leave_one_day_offsets(residual: np.ndarray, drop: int, kappa: float = HOUR_KAPPA) -> np.ndarray:
    """Hour offsets from every training day except ``drop``. Denominator is kept days + kappa."""
    resid = np.asarray(residual, dtype=np.float64).reshape(-1)
    if resid.size % DAY_HOURS:
        raise ValueError("residual length must be a whole number of days")
    folded = resid.reshape(-1, DAY_HOURS)
    if not 0 <= drop < folded.shape[0]:
        raise ValueError("dropped day is outside the training window")
    kept = np.delete(folded, drop, axis=0).reshape(-1)
    return hour_offsets_raw(kept, kappa)


def hour_offsets_raw(residual: np.ndarray, kappa: float) -> np.ndarray:
    resid = np.asarray(residual, dtype=np.float64).reshape(-1)
    if resid.size % DAY_HOURS:
        raise ValueError("residual length must be a whole number of days")
    folded = resid.reshape(-1, DAY_HOURS)
    return folded.sum(axis=0) / (folded.shape[0] + float(kappa))


def fit_line_and_residual(prediction: np.ndarray, target: np.ndarray, eligible: np.ndarray, solve_old):
    """Published (a, b) on all 72 hours, and the residual after that line."""
    n_cells = prediction.shape[1]
    scale = np.ones(n_cells, dtype=np.float64)
    shift = np.zeros(n_cells, dtype=np.float64)
    residual = np.zeros_like(prediction, dtype=np.float64)
    for index in np.flatnonzero(eligible):
        pred = prediction[:, index]
        truth = target[:, index]
        found_a, found_b = solve_old(pred, truth, 1.0)
        scale[index] = found_a
        shift[index] = found_b
        residual[:, index] = truth - found_a * pred - found_b
    return scale, shift, residual


def offsets_from_residual(residual: np.ndarray, eligible: np.ndarray, kappa: float) -> np.ndarray:
    offsets = np.zeros((DAY_HOURS, residual.shape[1]), dtype=np.float64)
    for index in np.flatnonzero(eligible):
        offsets[:, index] = hour_offsets_raw(residual[:, index], kappa)
    return offsets


def apply_correction(prediction: np.ndarray, scale: np.ndarray, shift: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    clock = np.arange(prediction.shape[0]) % DAY_HOURS
    base = prediction * scale.reshape(1, -1) + shift.reshape(1, -1)
    return base + offsets[clock, :]


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - target)))


def cell_deltas(candidate: np.ndarray, reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Positive means the candidate has a lower per-cell MAE than the reference."""
    return np.mean(np.abs(reference - target), axis=0) - np.mean(np.abs(candidate - target), axis=0)


def clock_permutation_count(
    base: np.ndarray,
    offsets: np.ndarray,
    target: np.ndarray,
    n_perm: int = N_PERM,
    seed: int = PERM_SEED,
) -> dict[str, float]:
    """How often a shared scramble of the 24 offsets matches or beats the true clock."""
    clock = np.arange(base.shape[0]) % DAY_HOURS
    true_pred = base + offsets[clock, :]
    true_mae = mae(true_pred, target)
    rng = np.random.default_rng(seed)
    count = 0
    null_maes = np.empty(n_perm, dtype=np.float64)
    batch = 25
    for start in range(0, n_perm, batch):
        width = min(batch, n_perm - start)
        perms = np.vstack([rng.permutation(DAY_HOURS) for _ in range(width)])
        scrambled = offsets[perms[:, clock], :]
        null = np.mean(np.abs(base + scrambled - target), axis=(1, 2))
        null_maes[start : start + width] = null
        count += int(np.sum(null <= true_mae))
    return {
        "true_mae": true_mae,
        "null_p05": float(np.quantile(null_maes, 0.05)),
        "n_null_le_true": count,
        "n_perm": n_perm,
    }


def week_block_interval(
    candidate: np.ndarray,
    reference: np.ndarray,
    target: np.ndarray,
    reps: int = BOOT_REPS,
    seed: int = BOOT_SEED,
) -> dict[str, float]:
    """Resample 7-day blocks. The last shorter block stays its own block."""
    hours, cells = target.shape
    if hours % DAY_HOURS:
        raise ValueError("test length must be a whole number of days")
    days = hours // DAY_HOURS
    day_diff = (
        np.abs(candidate - target).reshape(days, DAY_HOURS, cells).sum(axis=(1, 2))
        - np.abs(reference - target).reshape(days, DAY_HOURS, cells).sum(axis=(1, 2))
    )
    lengths = np.asarray(block_lengths(days), dtype=np.float64)
    starts = np.cumsum(np.concatenate(([0.0], lengths[:-1])))
    block_sum = np.asarray([day_diff[int(start) : int(start + length)].sum() for start, length in zip(starts, lengths)])
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, lengths.size, size=(reps, lengths.size))
    diffs = block_sum[drawn].sum(axis=1) / (lengths[drawn].sum(axis=1) * DAY_HOURS * cells)
    point = float(day_diff.sum() / (days * DAY_HOURS * cells))
    return {"point": point, "low": float(np.quantile(diffs, 0.025)), "high": float(np.quantile(diffs, 0.975))}


def mae_by_hour(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    gap = np.abs(prediction - target)
    clock = np.arange(prediction.shape[0]) % DAY_HOURS
    totals = np.zeros(DAY_HOURS, dtype=np.float64)
    counts = np.zeros(DAY_HOURS, dtype=np.float64)
    np.add.at(totals, clock, gap.sum(axis=1))
    np.add.at(counts, clock, prediction.shape[1])
    return totals / counts


def _task_key(source: str, seed: int) -> str:
    return f"{source}-seed{seed}"


def _fmt(value: float) -> str:
    return f"{float(value):.4f}"


def _load_split(cache_dir: Path, source: str, seed: int, split: str) -> dict[str, Any]:
    path = cache_dir / f"{source}_seed{seed}_{split}.npz"
    with np.load(path) as blob:
        loaded = {
            "prediction": np.asarray(blob["prediction"], dtype=np.float64),
            "target": np.asarray(blob["target"], dtype=np.float64),
        }
        if split == "train":
            loaded["eligible"] = np.asarray(blob["eligible"], dtype=bool)
    return loaded


def _evaluate_task(train: dict[str, Any], test: dict[str, Any], solve_old, hour_offsets) -> dict[str, Any]:
    scale, shift, residual = fit_line_and_residual(train["prediction"], train["target"], train["eligible"], solve_old)
    base = test["prediction"] * scale.reshape(1, -1) + shift.reshape(1, -1)
    reference = apply_correction(test["prediction"], scale, shift, np.zeros((DAY_HOURS, N_CELLS)))
    o72_mae = mae(reference, test["target"])
    kappa_mae = {}
    primary = None
    primary_offsets = None
    for kappa in KAPPA_GRID:
        offsets = offsets_from_residual(residual, train["eligible"], kappa)
        # Keep the published helper and this script on the same shrinkage.
        if kappa == HOUR_KAPPA:
            via_helper = np.zeros_like(offsets)
            for index in np.flatnonzero(train["eligible"]):
                via_helper[:, index] = hour_offsets(residual[:, index], kappa)
            if not np.allclose(via_helper, offsets):
                raise RuntimeError("hour offset helper disagrees with this script")
            primary_offsets = offsets
            primary = apply_correction(test["prediction"], scale, shift, offsets)
        kappa_mae[str(kappa)] = mae(apply_correction(test["prediction"], scale, shift, offsets), test["target"])
    if primary is None or primary_offsets is None:
        raise RuntimeError("kappa 3 was not in the grid")
    work = weekday_mask(test["prediction"].shape[0])
    if work.shape[0] != test["prediction"].shape[0]:
        raise RuntimeError("weekday mask length does not match the test window")
    delta = cell_deltas(primary, reference, test["target"])
    gated = train["eligible"]
    gated_delta = delta[gated]
    total = float(gated_delta.sum())
    top = float(np.sort(gated_delta)[-TOP_CELLS:].sum()) if gated_delta.size >= TOP_CELLS else float("nan")
    dropped = {}
    for day in range(N_HOURS // DAY_HOURS):
        offsets = np.zeros((DAY_HOURS, N_CELLS), dtype=np.float64)
        for index in np.flatnonzero(gated):
            offsets[:, index] = leave_one_day_offsets(residual[:, index], day, HOUR_KAPPA)
        dropped[str(day)] = mae(apply_correction(test["prediction"], scale, shift, offsets), test["target"])
    by_hour_ref = mae_by_hour(reference, test["target"])
    by_hour_new = mae_by_hour(primary, test["target"])
    return {
        "o72_mae": o72_mae,
        "kappa": kappa_mae,
        "half": {
            "early": mae(primary[:HALF_HOURS], test["target"][:HALF_HOURS]),
            "late": mae(primary[HALF_HOURS:], test["target"][HALF_HOURS:]),
            "early_o72": mae(reference[:HALF_HOURS], test["target"][:HALF_HOURS]),
            "late_o72": mae(reference[HALF_HOURS:], test["target"][HALF_HOURS:]),
        },
        "calendar": {
            "weekday": mae(primary[work], test["target"][work]),
            "weekend": mae(primary[~work], test["target"][~work]),
            "weekday_o72": mae(reference[work], test["target"][work]),
            "weekend_o72": mae(reference[~work], test["target"][~work]),
            "n_weekday_hours": int(work.sum()),
            "n_weekend_hours": int((~work).sum()),
        },
        "cells": {
            "n_better": int(np.sum(gated_delta > 0)),
            "n_worse": int(np.sum(gated_delta < 0)),
            "n_tie": int(np.sum(gated_delta == 0)),
            "top5_share": (top / total) if total > 0 else float("nan"),
        },
        "permutation": clock_permutation_count(base, primary_offsets, test["target"]),
        "leave_one_day": dropped,
        "week": week_block_interval(primary, reference, test["target"]),
        "hour_drop": (by_hour_ref - by_hour_new).tolist(),
    }


def _column(results: dict[str, dict[str, Any]], path: tuple[str, ...]) -> list[float]:
    values = []
    for source, seed in TASKS:
        found = results[_task_key(source, seed)]
        for key in path:
            found = found[key]
        values.append(float(found))
    return values


def _all_lower(candidate: list[float], reference: list[float]) -> bool:
    return all(left < right for left, right in zip(candidate, reference))


def _judge(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    o72 = _column(results, ("o72_mae",))
    neighbors = {str(kappa): _all_lower(_column(results, ("kappa", str(kappa))), o72) for kappa in NEIGHBOR_KAPPA}
    halves = {
        "early": _all_lower(_column(results, ("half", "early")), _column(results, ("half", "early_o72"))),
        "late": _all_lower(_column(results, ("half", "late")), _column(results, ("half", "late_o72"))),
    }
    calendar = {
        "weekday": _all_lower(_column(results, ("calendar", "weekday")), _column(results, ("calendar", "weekday_o72"))),
        "weekend": _all_lower(_column(results, ("calendar", "weekend")), _column(results, ("calendar", "weekend_o72"))),
    }
    cells_ok = all(item["cells"]["n_better"] >= MIN_CELLS_BETTER for item in results.values())
    concentrated = [key for key, item in results.items() if item["cells"]["top5_share"] > CONCENTRATION_FLAG]
    perm_ok = all(item["permutation"]["n_null_le_true"] <= PERM_PASS_MAX for item in results.values())
    lodo = {
        day: _all_lower(_column(results, ("leave_one_day", day)), o72) for day in ("0", "1", "2")
    }
    week_seed42 = all(results[_task_key(source, 42)]["week"]["high"] < 0.0 for source in SOURCES)
    checks = {
        "neighbor_kappa": all(neighbors.values()),
        "both_halves": all(halves.values()),
        "weekday_and_weekend": all(calendar.values()),
        "cells": cells_ok,
        "clock_alignment": perm_ok,
        "leave_one_day": all(lodo.values()),
    }
    return {
        "checks": checks,
        "neighbors": neighbors,
        "halves": halves,
        "calendar": calendar,
        "lodo": lodo,
        "concentrated_tasks": concentrated,
        "week_seed42_excludes_zero": week_seed42,
        "all_six_passed": all(checks.values()),
    }


def _markdown(payload: dict[str, Any]) -> str:
    results = payload["results"]
    judge = payload["judgment"]
    lines = [
        "# 钟点偏差稳健性",
        "",
        "没有新训练。检查在打分前写在 [CELL_AFFINE_ROBUST_PLAN.md](../CELL_AFFINE_ROBUST_PLAN.md)。",
        "测试小时以前已经看过。这里只检查已经报告的优势是不是靠局部碰上的。",
        "",
        "## 核对",
        "",
    ]
    if payload["failures"]:
        lines.append("锚点或上一份结果没有对上，下面的通过与否不采用。")
        lines.extend(f"- {item}" for item in payload["failures"])
    else:
        lines.append("旧公式 λ=72 对上了已发表锚点，κ=3 对上了上一份全市测试 MAE。")
    lines.extend(["", "## 判定", ""])
    if payload["failures"]:
        lines.append("未判定。")
    elif judge["all_six_passed"]:
        lines.append("六项事先写定的检查都通过。这不是靠单段日期、单侧日型、少数格子或 κ=3 的两个邻居碰出来的。")
    else:
        lines.append("六项没有全部通过。不能把这项结果写成已经排除了偶然。")
    if not payload["failures"]:
        labels = (
            ("neighbor_kappa", "相邻收缩 κ=1 和 κ=6"),
            ("both_halves", "测试段前 30 天和后 30 天"),
            ("weekday_and_weekend", "工作日和周末"),
            ("cells", "至少 30 个过门格更低"),
            ("clock_alignment", "真钟点优于 95% 的打乱"),
            ("leave_one_day", "三次留出训练日"),
        )
        for key, label in labels:
            lines.append(f"- {label}：{'通过' if judge['checks'][key] else '没有通过'}。")
        if judge["concentrated_tasks"]:
            names = "、".join(judge["concentrated_tasks"])
            lines.append(f"- 改进很集中（前 5 格占比超过 0.8）：{names}。这一条按计划不单独否决。")
        else:
            lines.append("- 前 5 格占比没有超过 0.8。")
        week = "种子 42 两个城市的按周区间都在 0 以下。" if judge["week_seed42_excludes_zero"] else "种子 42 的按周区间没有都在 0 以下。"
        lines.append(f"- 按周自助不参与上面六项。{week}")
    lines.extend(["", "## κ 网格的测试 MAE", "", "κ=3 是原方法。其余只报告，不替换原方法。", ""])
    header = "| κ | " + " | ".join(f"{CITY[source]} {seed}" for source, seed in TASKS) + " |"
    lines.extend([header, "|" + "---|" * 7])
    for kappa in KAPPA_GRID:
        cells = [_fmt(results[_task_key(source, seed)]["kappa"][str(kappa)]) for source, seed in TASKS]
        lines.append(f"| {kappa:g} | " + " | ".join(cells) + " |")
    o72_cells = [_fmt(results[_task_key(source, seed)]["o72_mae"]) for source, seed in TASKS]
    lines.append("| 旧公式 | " + " | ".join(o72_cells) + " |")
    lines.extend(["", "## 前 30 天和后 30 天", ""])
    lines.extend([header.replace("κ", "段"), "|" + "---|" * 7])
    for name, title in (("early", "前 30 天"), ("late", "后 30 天")):
        cells = [_fmt(results[_task_key(source, seed)]["half"][name]) for source, seed in TASKS]
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
        cells = [_fmt(results[_task_key(source, seed)]["half"][f"{name}_o72"]) for source, seed in TASKS]
        lines.append(f"| {title}，旧公式 | " + " | ".join(cells) + " |")
    lines.extend(["", "## 工作日和周末", ""])
    lines.extend([header.replace("κ", "段"), "|" + "---|" * 7])
    for name, title in (("weekday", "工作日"), ("weekend", "周末")):
        cells = [_fmt(results[_task_key(source, seed)]["calendar"][name]) for source, seed in TASKS]
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
        cells = [_fmt(results[_task_key(source, seed)]["calendar"][f"{name}_o72"]) for source, seed in TASKS]
        lines.append(f"| {title}，旧公式 | " + " | ".join(cells) + " |")
    sample = next(iter(results.values()))["calendar"]
    lines.append("")
    lines.append(
        f"工作日小时数 {sample['n_weekday_hours']}，周末小时数 {sample['n_weekend_hours']}。"
        "训练窗三天都是工作日。"
    )
    lines.extend(["", "## 过门格", "", "| 任务 | 更低 | 更高 | 相同 | 前 5 格占比 |", "|---|---:|---:|---:|---:|"])
    for source, seed in TASKS:
        cells = results[_task_key(source, seed)]["cells"]
        lines.append(
            f"| {CITY[source]} {seed} | {cells['n_better']} | {cells['n_worse']} | {cells['n_tie']} | {cells['top5_share']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 钟点打乱",
            "",
            "次数是 1000 次打乱里，测试 MAE 小于或等于真钟点的次数。不超过 50 才算通过。",
            "",
            "| 任务 | 次数 | 真钟点 MAE | 打乱的 5% 分位 |",
            "|---|---:|---:|---:|",
        ]
    )
    for source, seed in TASKS:
        perm = results[_task_key(source, seed)]["permutation"]
        lines.append(
            f"| {CITY[source]} {seed} | {perm['n_null_le_true']} | {_fmt(perm['true_mae'])} | {_fmt(perm['null_p05'])} |"
        )
    lines.extend(["", "## 留出一天后的测试 MAE", "", "直线仍用 72 小时。钟点偏差只用剩下的两天。", ""])
    lines.extend([header.replace("κ", "留出"), "|" + "---|" * 7])
    for day, title in (("0", "留出第 1 天"), ("1", "留出第 2 天"), ("2", "留出第 3 天")):
        cells = [_fmt(results[_task_key(source, seed)]["leave_one_day"][day]) for source, seed in TASKS]
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    lines.append("| 旧公式 | " + " | ".join(o72_cells) + " |")
    lines.extend(["", "## 按周自助", "", "差是本方法减旧公式。", "", "| 任务 | 差 | 2.5% | 97.5% |", "|---|---:|---:|---:|"])
    for source, seed in TASKS:
        week = results[_task_key(source, seed)]["week"]
        lines.append(f"| {CITY[source]} {seed} | {week['point']:.4f} | {week['low']:.4f} | {week['high']:.4f} |")
    lines.extend(["", "## 各钟点的测试 MAE 下降", "", "六个任务平均。正数表示这个钟点本方法更低。不参与判定。", ""])
    drops = np.vstack([results[_task_key(source, seed)]["hour_drop"] for source, seed in TASKS])
    mean_drop = drops.mean(axis=0)
    lines.append("| 钟点 | 平均下降 | 六个任务里更低的个数 |")
    lines.append("|---:|---:|---:|")
    for hour in range(DAY_HOURS):
        lines.append(f"| {hour} | {mean_drop[hour]:.4f} | {int(np.sum(drops[:, hour] > 0))} |")
    lines.extend(
        [
            "",
            "## 局限",
            "",
            "测试段不是未见数据。六项检查用的是同一段已经看过的测试小时。",
            "训练窗没有周末，周末这一项是在用没见过的日型检查工作日上学到的钟点偏差。",
            "按周只有 9 块，区间比按天宽。",
            "",
        ]
    )
    return "\n".join(lines)


def run(cache_dir: Path, prior_json: Path, out_json: Path, out_md: Path) -> int:
    v2 = _load_script("cell_affine_v2_robust", "19_cell_affine_v2.py")
    previous_script = _load_script("cell_affine_next_robust", "23_cell_affine_next.py")
    checks = json.loads((cache_dir / "checks.json").read_text(encoding="utf-8"))
    if not checks.get("passed"):
        raise RuntimeError("prediction checks did not pass")
    prior = json.loads(prior_json.read_text(encoding="utf-8"))
    reference_mask = None
    results: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for source, seed in TASKS:
        train = _load_split(cache_dir, source, seed, "train")
        test = _load_split(cache_dir, source, seed, "test")
        if train["prediction"].shape != (N_HOURS, N_CELLS) or int(train["eligible"].sum()) != N_GATE:
            raise RuntimeError(f"{source} seed {seed} train cache shape is unexpected")
        if test["prediction"].shape != (TEST_HOURS, N_CELLS):
            raise RuntimeError(f"{source} seed {seed} test cache shape is unexpected")
        if reference_mask is None:
            reference_mask = train["eligible"].copy()
        elif not np.array_equal(reference_mask, train["eligible"]):
            raise RuntimeError("gate mask differs across tasks")
        print(f"check {source} seed {seed}", flush=True)
        record = _evaluate_task(train, test, v2.solve_old, previous_script.hour_offsets)
        results[_task_key(source, seed)] = record
        if (source, seed) in ANCHOR and abs(record["o72_mae"] - ANCHOR[(source, seed)]) > ANCHOR_ATOL:
            failures.append(f"{source} seed {seed} old MAE {record['o72_mae']:.10f} != {ANCHOR[(source, seed)]:.10f}")
        if (source, seed) in OTHER_ANCHOR and abs(record["o72_mae"] - OTHER_ANCHOR[(source, seed)]) > OTHER_ATOL:
            failures.append(f"{source} seed {seed} old MAE {record['o72_mae']:.6f} !~ {OTHER_ANCHOR[(source, seed)]:.4f}")
        prior_row = next(item for item in prior["results"] if item["source"] == source and item["seed"] == seed)
        prior_mae = float(prior_row["test"]["O72H"]["mae"])
        if abs(record["kappa"]["3.0"] - prior_mae) > REPRO_ATOL:
            failures.append(f"{source} seed {seed} kappa 3 MAE {record['kappa']['3.0']:.10f} != prior {prior_mae:.10f}")
    first = test_day_dates(1)[0]
    if first != dt.date(2016, 11, 2) or first.weekday() != 2:
        raise RuntimeError(f"test start date {first} is not Wednesday 2016-11-02")
    judgment = None if failures else _judge(results)
    payload = {
        "anchors_ok": not failures,
        "failures": failures,
        "judgment": judgment,
        "results": results,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    text = _markdown(payload if judgment is not None else {**payload, "judgment": _judge(results), "failures": failures})
    if failures:
        text = _markdown({**payload, "judgment": _judge(results)})
    out_md.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--prior-json", type=Path, default=ROOT / "docs" / "profiles" / "cell_affine_next.json")
    parser.add_argument("--out-json", type=Path, default=ROOT / "docs" / "profiles" / "cell_affine_robust.json")
    parser.add_argument("--out-md", type=Path, default=ROOT / "docs" / "profiles" / "cell_affine_robust.md")
    args = parser.parse_args()
    return run(args.cache_dir, args.prior_json, args.out_json, args.out_md)


if __name__ == "__main__":
    raise SystemExit(main())
