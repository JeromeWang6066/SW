#!/usr/bin/env python3
"""Compare four error gates in front of the published hour-bias correction.

Protocol: docs/ERROR_GATE_PLAN.md. ``score-val`` never opens a test cache.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feedback.error_gate import DISTANCES, GRIDS, MODES, choose_threshold, gaps, release  # noqa: E402
from src.feedback.published_hour import (  # noqa: E402
    SERIES_DAY,
    TEST_DAY,
    TRAIN_HOUR,
    VAL_DAY,
    VAL_HOUR,
    correct_range,
    is_weekend,
    weekday_rows,
    window_for,
)

SELECT_SEED = 42
TEST_SEEDS = (42, 43, 44)
SOURCES = ("ny", "chi")
LABEL = {"level": "总量", "wape": "小时相对误差", "profile": "钟点轮廓", "utility": "留出日有用"}
MODE_LABEL = {"once": "一次判定", "daily": "按日判定"}


def _load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scripts/{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _loop():
    return _load("26_published_hour_loop.py", "published_hour_loop_gate")


def _hours(days: list[int]) -> np.ndarray:
    return np.concatenate([np.arange(day * 24, (day + 1) * 24) for day in days])


def _weekday_days(first: int, last: int) -> list[int]:
    return [day for day in range(first, last) if not is_weekend(day)]


def _published(path: Path, source: str, seed: int, arm: str) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for run in payload["runs"]:
        if run["source"] == source and int(run["seed"]) == seed:
            return float(run["arms"][arm]["weekday"]["mae"])
    raise KeyError(f"{source} seed {seed} {arm}")


def _pack(prediction: np.ndarray, truth: np.ndarray, scale: float, first: int, last: int) -> dict[str, float]:
    scored = _loop()._metrics(prediction, truth, scale, first, last)
    return {
        "weekday_mae": float(scored["weekday"]["mae"]),
        "weekday_rmse": float(scored["weekday"]["rmse"]),
        "weekend_mae": float(scored["weekend"]["mae"]),
    }


def _restore(corrected: np.ndarray, original: np.ndarray, keep_of: Callable[[int], np.ndarray], first: int, last: int) -> np.ndarray:
    out = np.array(corrected, copy=True)
    for day in range(first, last):
        drop = np.flatnonzero(~keep_of(day))
        if drop.size:
            start = day * 24
            out[start : start + 24, drop] = original[start : start + 24, drop]
    return out


def _prepare(pred: np.ndarray, truth: np.ndarray, eligible: np.ndarray, solve_old, first: int, last: int) -> dict[str, Any]:
    once_gap = gaps(pred[TRAIN_HOUR:VAL_HOUR], truth[TRAIN_HOUR:VAL_HOUR], solve_old, eligible)
    daily_gap: dict[int, dict[str, np.ndarray]] = {}
    for day in range(first, last):
        window = window_for(day, "CL")
        hours = _hours(window)
        daily_gap[day] = gaps(pred[hours], truth[hours], solve_old, eligible)
    return {
        "once_gap": once_gap,
        "daily_gap": daily_gap,
        "ol": correct_range(pred, truth, eligible, solve_old, "OL", first, last),
        "cl": correct_range(pred, truth, eligible, solve_old, "CL", first, last),
    }


def _keep_factory(prepared: dict[str, Any], eligible: np.ndarray, mode: str, distance: str, tau: float) -> Callable[[int], np.ndarray]:
    if mode == "once":
        keep = release(prepared["once_gap"][distance], tau, eligible)

        def _once(_day: int, locked: np.ndarray = keep) -> np.ndarray:
            return locked

        return _once

    def _daily(day: int) -> np.ndarray:
        return release(prepared["daily_gap"][day][distance], tau, eligible)

    return _daily


def _cells(keep_of: Callable[[int], np.ndarray], first: int, last: int) -> float:
    days = _weekday_days(first, last)
    return float(np.mean([int(keep_of(day).sum()) for day in days]))


def _score_grid(
    prepared: dict[str, Any],
    original: np.ndarray,
    truth: np.ndarray,
    eligible: np.ndarray,
    scale: float,
    first: int,
    last: int,
) -> dict[str, Any]:
    bases = {"once": prepared["ol"], "daily": prepared["cl"]}
    grid: dict[str, dict[str, list[dict[str, float]]]] = {mode: {name: [] for name in DISTANCES} for mode in MODES}
    baseline: dict[str, dict[str, float]] = {}
    for mode, corrected in bases.items():
        packed = _pack(corrected, truth, scale, first, last)
        baseline[mode] = packed
        for name in DISTANCES:
            for tau in GRIDS[name]:
                keep_of = _keep_factory(prepared, eligible, mode, name, tau)
                masked = _restore(corrected, original, keep_of, first, last)
                packed = _pack(masked, truth, scale, first, last)
                below = ~eligible
                if float(np.max(np.abs(masked[first * 24 : last * 24][:, below] - original[first * 24 : last * 24][:, below]))) != 0.0:
                    raise RuntimeError(f"{mode} {name} changed a cell below the 150 gate")
                grid[mode][name].append(
                    {
                        "tau": float(tau),
                        "weekday_mae": packed["weekday_mae"],
                        "weekday_rmse": packed["weekday_rmse"],
                        "weekend_mae": packed["weekend_mae"],
                        "cells": _cells(keep_of, first, last),
                    }
                )
    return {"baseline": baseline, "grid": grid}


def _choices(by_source: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {mode: {} for mode in MODES}
    for mode in MODES:
        for name in DISTANCES:
            rows = []
            for tau in GRIDS[name]:
                ny = next(row for row in by_source["ny"]["grid"][mode][name] if row["tau"] == tau)
                chi = next(row for row in by_source["chi"]["grid"][mode][name] if row["tau"] == tau)
                rows.append({"tau": tau, "ny": ny["weekday_mae"], "chi": chi["weekday_mae"], "ny_cells": ny["cells"], "chi_cells": chi["cells"]})
            pick = choose_threshold(rows, by_source["ny"]["baseline"][mode]["weekday_mae"], by_source["chi"]["baseline"][mode]["weekday_mae"])
            chosen[mode][name] = None if pick is None else {
                "tau": float(pick["tau"]),
                "ny": float(pick["ny"]),
                "chi": float(pick["chi"]),
                "ny_cells": float(pick["ny_cells"]),
                "chi_cells": float(pick["chi_cells"]),
            }
    return chosen


def _check_baseline(got: dict[str, Any], published_path: Path, seed: int) -> list[str]:
    failures = []
    expected = {"once": "OL", "daily": "CL"}
    for source in SOURCES:
        for mode, arm in expected.items():
            left = got[source]["baseline"][mode]["weekday_mae"]
            right = _published(published_path, source, seed, arm)
            if abs(left - right) > 1e-8:
                failures.append(f"{source} seed {seed} {mode} weekday MAE {left:.8f} != {arm} {right:.8f}")
    return failures


def stage_val(v2_cache: Path, new_cache: Path, out_dir: Path) -> int:
    loop = _loop()
    solve_old = loop._solve_old()
    by_source = {}
    for source in SOURCES:
        print(f"val {source} seed {SELECT_SEED}", flush=True)
        pred, truth, scale, eligible = loop._series(v2_cache, new_cache, source, SELECT_SEED, ("prefix", "train", "val"))
        prepared = _prepare(pred, truth, eligible, solve_old, VAL_DAY, TEST_DAY)
        by_source[source] = _score_grid(prepared, pred, truth, eligible, scale, VAL_DAY, TEST_DAY)
    val_path = ROOT / "docs" / "profiles" / "clock_hour_loop_val.json"
    failures = _check_baseline(by_source, val_path, SELECT_SEED)
    payload = {
        "passed": not failures,
        "failures": failures,
        "seed": SELECT_SEED,
        "baselines": {source: by_source[source]["baseline"] for source in SOURCES},
        "grid": {source: by_source[source]["grid"] for source in SOURCES},
        "choices": {} if failures else _choices(by_source),
    }
    loop._write_json(out_dir / "error_gate_val.json", payload)
    if failures:
        for item in failures:
            print(item, file=sys.stderr)
        return 1
    for mode in MODES:
        for name in DISTANCES:
            pick = payload["choices"][mode][name]
            text = "沿用全部 50 格" if pick is None else f"τ={pick['tau']}"
            print(f"  {MODE_LABEL[mode]} {LABEL[name]}: {text}", flush=True)
    print(f"wrote {out_dir / 'error_gate_val.json'}", flush=True)
    return 0


def _same_tau(row: dict[str, Any], tau: float) -> bool:
    return abs(float(row["tau"]) - float(tau)) <= 1e-12


def stage_test(v2_cache: Path, new_cache: Path, out_dir: Path) -> int:
    val_path = out_dir / "error_gate_val.json"
    if not val_path.is_file():
        print("score-test refused: validation file is missing", file=sys.stderr)
        return 1
    val = json.loads(val_path.read_text(encoding="utf-8"))
    if not val.get("passed"):
        print("score-test refused: validation checks failed", file=sys.stderr)
        return 1
    loop = _loop()
    solve_old = loop._solve_old()
    runs: dict[str, dict[int, dict[str, Any]]] = {source: {} for source in SOURCES}
    for source in SOURCES:
        for seed in TEST_SEEDS:
            print(f"test {source} seed {seed}", flush=True)
            pred, truth, scale, eligible = loop._series(
                v2_cache, new_cache, source, seed, ("prefix", "train", "val", "test")
            )
            prepared = _prepare(pred, truth, eligible, solve_old, VAL_DAY, SERIES_DAY)
            scored = _score_grid(prepared, pred, truth, eligible, scale, TEST_DAY, SERIES_DAY)
            runs[source][seed] = {"scored": scored, "pred": pred, "truth": truth, "eligible": eligible, "prepared": prepared, "scale": scale}
    failures = _check_baseline(
        {source: runs[source][SELECT_SEED]["scored"] for source in SOURCES},
        ROOT / "docs" / "profiles" / "clock_hour_loop.json",
        SELECT_SEED,
    )
    if failures:
        for item in failures:
            print(item, file=sys.stderr)
        return 1
    verdicts = []
    for mode in MODES:
        for name in DISTANCES:
            pick = val["choices"][mode][name]
            record: dict[str, Any] = {"mode": mode, "distance": name, "choice": pick}
            if pick is None:
                record["verdict"] = "未采用"
            else:
                record["tasks"] = []
                for source in SOURCES:
                    for seed in TEST_SEEDS:
                        base = runs[source][seed]["scored"]["baseline"][mode]
                        row = next(item for item in runs[source][seed]["scored"]["grid"][mode][name] if _same_tau(item, pick["tau"]))
                        record["tasks"].append(
                            {
                                "source": source,
                                "seed": seed,
                                "mae": row["weekday_mae"],
                                "rmse": row["weekday_rmse"],
                                "weekend_mae": row["weekend_mae"],
                                "base_mae": base["weekday_mae"],
                                "base_rmse": base["weekday_rmse"],
                                "cells": row["cells"],
                            }
                        )
                diff = {}
                rmse_ok = True
                for source in SOURCES:
                    bundle = runs[source][SELECT_SEED]
                    keep_of = _keep_factory(bundle["prepared"], bundle["eligible"], mode, name, float(pick["tau"]))
                    corrected = bundle["prepared"]["ol" if mode == "once" else "cl"]
                    masked = _restore(corrected, bundle["pred"], keep_of, TEST_DAY, SERIES_DAY)
                    diff[source] = loop._weekday_bootstrap(masked, corrected, bundle["truth"], TEST_DAY, SERIES_DAY)
                    task = next(item for item in record["tasks"] if item["source"] == source and item["seed"] == SELECT_SEED)
                    rmse_ok = rmse_ok and task["rmse"] <= task["base_rmse"] + 1e-12
                record["bootstrap"] = diff
                record["verdict"] = _judge_pair(diff, rmse_ok)
                signs = [np.sign(item["mae"] - item["base_mae"]) for item in record["tasks"]]
                record["stable"] = len(set(float(item) for item in signs)) == 1 and float(signs[0]) != 0.0
            verdicts.append(record)
            print(f"  {MODE_LABEL[mode]} {LABEL[name]}: {record['verdict']}", flush=True)
    payload = {"passed": True, "failures": [], "verdicts": verdicts}
    loop._write_json(out_dir / "error_gate.json", payload)
    md_path = out_dir / "error_gate.md"
    md_path.write_text(_markdown(val, payload), encoding="utf-8", newline="\n")
    print(f"wrote {md_path}", flush=True)
    return 0


def _judge_pair(diff: dict[str, dict[str, float]], rmse_ok: bool) -> str:
    if diff["ny"]["high"] < 0 and diff["chi"]["high"] < 0 and rmse_ok:
        return "更好"
    if diff["ny"]["low"] > 0 or diff["chi"]["low"] > 0:
        return "更差"
    return "无法区分"


def _fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _markdown(val: dict[str, Any], test: dict[str, Any]) -> str:
    lines = [
        "# 150 门前面的四道误差判断",
        "",
        "协议在打分前写在 [ERROR_GATE_PLAN.md](../ERROR_GATE_PLAN.md)。",
        "一次判定等于在 OL 上关门。按日判定等于在 CL 上关门。阈值只在种子 42 的验证集网格里选择。",
        "",
        "## 验证集工作日 MAE",
        "",
    ]
    for mode in MODES:
        lines.append(f"### {MODE_LABEL[mode]}")
        lines.append("")
        base_ny = val["baselines"]["ny"][mode]["weekday_mae"]
        base_chi = val["baselines"]["chi"][mode]["weekday_mae"]
        lines.append(f"50 格全放行：纽约 {_fmt(base_ny)}，芝加哥 {_fmt(base_chi)}。")
        lines.append("")
        lines.append("| 距离 | τ | 纽约 | 芝加哥 | 纽约格数 | 芝加哥格数 | 选择 |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for name in DISTANCES:
            chosen = val["choices"][mode][name]
            for tau in GRIDS[name]:
                ny = next(row for row in val["grid"]["ny"][mode][name] if row["tau"] == tau)
                chi = next(row for row in val["grid"]["chi"][mode][name] if row["tau"] == tau)
                mark = "留下" if chosen is not None and abs(float(chosen["tau"]) - float(tau)) <= 1e-12 else ""
                lines.append(
                    f"| {LABEL[name]} | {tau:.2f} | {_fmt(ny['weekday_mae'])} | {_fmt(chi['weekday_mae'])} | "
                    f"{ny['cells']:.1f} | {chi['cells']:.1f} | {mark} |"
                )
        lines.append("")
    lines.extend(["## 测试", ""])
    lines.append("| 模式 | 距离 | 判定 | 种子 42 纽约 | 种子 42 芝加哥 |")
    lines.append("|---|---|---|---|---|")
    for record in test["verdicts"]:
        if record["verdict"] == "未采用":
            lines.append(f"| {MODE_LABEL[record['mode']]} | {LABEL[record['distance']]} | 未采用 | — | — |")
            continue
        ny = next(item for item in record["tasks"] if item["source"] == "ny" and item["seed"] == SELECT_SEED)
        chi = next(item for item in record["tasks"] if item["source"] == "chi" and item["seed"] == SELECT_SEED)
        boot = record["bootstrap"]
        lines.append(
            f"| {MODE_LABEL[record['mode']]} | {LABEL[record['distance']]} | {record['verdict']} | "
            f"{_fmt(ny['mae'])}（差 {_fmt(boot['ny']['point'])}） | {_fmt(chi['mae'])}（差 {_fmt(boot['chi']['point'])}） |"
        )
    lines.extend(["", "## 判定", ""])
    for record in test["verdicts"]:
        if record["verdict"] == "未采用":
            lines.append(f"- {MODE_LABEL[record['mode']]}，{LABEL[record['distance']]}：验证集没有留下。至少有一个源城的工作日 MAE 高于 50 格全放行。")
            continue
        detail = "；".join(
            f"{source} {_fmt(record['bootstrap'][source]['point'])} "
            f"[{_fmt(record['bootstrap'][source]['low'])}, {_fmt(record['bootstrap'][source]['high'])}]"
            for source in SOURCES
        )
        stable = "方向稳定" if record["stable"] else "方向不稳定"
        lines.append(f"- {MODE_LABEL[record['mode']]}，{LABEL[record['distance']]}，τ={record['choice']['tau']}：{record['verdict']}。{detail}。{stable}。")
    lines.extend(
        [
            "",
            "## 局限",
            "",
            "阈值网格是事先写定的五个点，不是连续搜索。种子 43、44 只检验选定的 τ，不另选。",
            "测试段在钟点偏差的实验里已经看过。这次仍是探索性对照。",
            "一次判定的周末沿用三个工作日上算出的那道门。按日判定的周末用周末自己的窗口。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("score-val", "score-test"))
    parser.add_argument("--v2-cache", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "logs" / "clock_hour_cache")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "profiles")
    args = parser.parse_args()
    v2_cache = _loop()._default_v2(args.v2_cache)
    if args.stage == "score-val":
        return stage_val(v2_cache, args.cache_dir, args.out_dir)
    return stage_test(v2_cache, args.cache_dir, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
