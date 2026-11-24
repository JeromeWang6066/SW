#!/usr/bin/env python3
"""Add below-gate cells to the daily hour-bias correction after a holdout vote.

Protocol: docs/ADD_BELOW_GATE_PLAN.md. ``score-val`` never opens a test cache.
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

from src.feedback.add_below_gate import PRIMARY_VOTES, VOTE_COUNTS, holdout_wins  # noqa: E402
from src.feedback.published_hour import (  # noqa: E402
    SERIES_DAY,
    SERIES_HOUR,
    TEST_DAY,
    TEST_HOUR,
    VAL_DAY,
    apply_day,
    correct_range,
    fit_window,
    is_weekend,
    window_for,
)

SOURCES = ("ny", "chi")
SEEDS = (42, 43, 44, 45, 46, 47, 48)
CHECK_SEEDS = (47, 48)


def _load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scripts/{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _loop():
    return _load("26_published_hour_loop.py", "published_hour_loop_add")


def _hours(days: list[int]) -> np.ndarray:
    return np.concatenate([np.arange(day * 24, (day + 1) * 24) for day in days])


def _weekday_days(first: int, last: int) -> list[int]:
    return [day for day in range(first, last) if not is_weekend(day)]


def _published(path: Path, source: str, seed: int) -> float | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for run in payload["runs"]:
        if run["source"] == source and int(run["seed"]) == seed:
            return float(run["arms"]["CL"]["weekday"]["mae"])
    return None


def _wins_for_days(pred: np.ndarray, truth: np.ndarray, candidates: np.ndarray, solve_old, first: int, last: int) -> dict[int, np.ndarray]:
    found = {}
    for day in range(first, last):
        window = window_for(day, "CL")
        if len(window) != 3:
            raise RuntimeError(f"day {day} does not have three earlier days of the same class")
        found[day] = holdout_wins(pred[_hours(window)], truth[_hours(window)], solve_old, candidates)
    return found


def _apply_votes(
    pred: np.ndarray,
    truth: np.ndarray,
    eligible: np.ndarray,
    baseline: np.ndarray,
    wins: dict[int, np.ndarray],
    solve_old,
    first: int,
    last: int,
) -> dict[int, np.ndarray]:
    outputs = {votes: np.array(baseline, copy=True) for votes in VOTE_COUNTS}
    below = ~eligible
    for day in range(first, last):
        add_any = below & (wins[day] >= 1)
        if not np.any(add_any):
            continue
        window = _hours(window_for(day, "CL"))
        scale, shift, offsets = fit_window(pred[window], truth[window], add_any, solve_old, with_hour=True)
        start = day * 24
        corrected = apply_day(pred[start : start + 24], scale, shift, offsets)
        for votes in VOTE_COUNTS:
            add = below & (wins[day] >= votes)
            outputs[votes][start : start + 24, add] = corrected[:, add]
    return outputs


def _added_totals(
    pred: np.ndarray,
    corrected: np.ndarray,
    truth: np.ndarray,
    eligible: np.ndarray,
    wins: dict[int, np.ndarray],
    votes: int,
    first: int,
    last: int,
) -> dict[str, float]:
    days = _weekday_days(first, last)
    counts = []
    base_ae = 0.0
    corr_ae = 0.0
    points = 0
    for day in days:
        add = (~eligible) & (wins[day] >= votes)
        counts.append(int(add.sum()))
        if not np.any(add):
            continue
        start = day * 24
        held = truth[start : start + 24][:, add]
        base_ae += float(np.abs(pred[start : start + 24][:, add] - held).sum())
        corr_ae += float(np.abs(corrected[start : start + 24][:, add] - held).sum())
        points += int(add.sum()) * 24
    return {
        "mean_cells": float(np.mean(counts) if counts else 0.0),
        "base_ae": base_ae,
        "corr_ae": corr_ae,
        "points": float(points),
    }


def _check_identity(corrected: np.ndarray, baseline: np.ndarray, pred: np.ndarray, eligible: np.ndarray, wins: dict[int, np.ndarray], votes: int, first: int, last: int) -> None:
    sl = slice(first * 24, last * 24)
    if not np.array_equal(corrected[sl][:, eligible], baseline[sl][:, eligible]):
        raise RuntimeError("a volume-gated cell changed")
    for day in range(first, last):
        stay = (~eligible) & (wins[day] < votes)
        start = day * 24
        if not np.array_equal(corrected[start : start + 24][:, stay], pred[start : start + 24][:, stay]):
            raise RuntimeError(f"day {day} changed a cell that lost the vote")


def _score_one(pred, truth, eligible, scale, solve_old, loop, first: int, last: int) -> dict[str, Any]:
    baseline = correct_range(pred, truth, eligible, solve_old, "CL", first, last)
    wins = _wins_for_days(pred, truth, ~eligible, solve_old, first, last)
    outputs = _apply_votes(pred, truth, eligible, baseline, wins, solve_old, first, last)
    packed = {"baseline": loop._metrics(baseline, truth, scale, first, last)}
    arms = {}
    for votes, values in outputs.items():
        _check_identity(values, baseline, pred, eligible, wins, votes, first, last)
        arms[str(votes)] = {
            "metrics": loop._metrics(values, truth, scale, first, last),
            "added": _added_totals(pred, values, truth, eligible, wins, votes, first, last),
        }
    packed["arms"] = arms
    packed["outputs"] = outputs
    packed["baseline_values"] = baseline
    return packed


def _arm_record(scored: dict[str, Any], votes: int) -> dict[str, float]:
    metrics = scored["arms"][str(votes)]["metrics"]
    added = scored["arms"][str(votes)]["added"]
    return {
        "weekday_mae": float(metrics["weekday"]["mae"]),
        "weekday_rmse": float(metrics["weekday"]["rmse"]),
        "weekend_mae": float(metrics["weekend"]["mae"]),
        "mean_cells": float(added["mean_cells"]),
        "base_ae": float(added["base_ae"]),
        "corr_ae": float(added["corr_ae"]),
        "points": float(added["points"]),
    }


def stage_val(v2_cache: Path, new_cache: Path, out_dir: Path) -> int:
    loop = _loop()
    solve_old = loop._solve_old()
    runs = []
    failures = []
    published = ROOT / "docs" / "profiles" / "clock_hour_loop_val.json"
    for source in SOURCES:
        for seed in SEEDS:
            print(f"val {source} seed {seed}", flush=True)
            pred, truth, scale, eligible = loop._series(v2_cache, new_cache, source, seed, ("prefix", "train", "val"))
            if int(eligible.sum()) != 50:
                failures.append(f"{source} seed {seed} gate {int(eligible.sum())}")
            scored = _score_one(pred, truth, eligible, scale, solve_old, loop, VAL_DAY, TEST_DAY)
            got = float(scored["baseline"]["weekday"]["mae"])
            expected = _published(published, source, seed)
            if expected is not None and abs(got - expected) > 1e-8:
                failures.append(f"{source} seed {seed} CL weekday {got:.8f} != {expected:.8f}")
            runs.append(
                {
                    "source": source,
                    "seed": seed,
                    "baseline": got,
                    "arms": {str(votes): _arm_record(scored, votes) for votes in VOTE_COUNTS},
                }
            )
            primary = runs[-1]["arms"][str(PRIMARY_VOTES)]
            print(
                f"  2 votes weekday {primary['weekday_mae']:.4f}  CL {got:.4f}  added {primary['mean_cells']:.2f}",
                flush=True,
            )
    payload = {"passed": not failures, "failures": failures, "primary_votes": PRIMARY_VOTES, "runs": runs}
    loop._write_json(out_dir / "add_below_gate_val.json", payload)
    if failures:
        for item in failures:
            print(item, file=sys.stderr)
        return 1
    print(f"wrote {out_dir / 'add_below_gate_val.json'}", flush=True)
    return 0


def _judge(diff: dict[str, dict[str, float]], rmse_ok: bool) -> str:
    if diff["ny"]["high"] < 0 and diff["chi"]["high"] < 0 and rmse_ok:
        return "更好"
    if diff["ny"]["low"] > 0 or diff["chi"]["low"] > 0:
        return "更差"
    return "无法区分"


def stage_test(v2_cache: Path, new_cache: Path, out_dir: Path) -> int:
    val_path = out_dir / "add_below_gate_val.json"
    if not val_path.is_file():
        print("score-test refused: validation file is missing", file=sys.stderr)
        return 1
    val = json.loads(val_path.read_text(encoding="utf-8"))
    if not val.get("passed"):
        print("score-test refused: validation checks failed", file=sys.stderr)
        return 1
    loop = _loop()
    solve_old = loop._solve_old()
    published = ROOT / "docs" / "profiles" / "clock_hour_loop.json"
    runs = []
    failures = []
    seed42 = {}
    for source in SOURCES:
        for seed in SEEDS:
            print(f"test {source} seed {seed}", flush=True)
            pred, truth, scale, eligible = loop._series(
                v2_cache, new_cache, source, seed, ("prefix", "train", "val", "test")
            )
            scored = _score_one(pred, truth, eligible, scale, solve_old, loop, TEST_DAY, SERIES_DAY)
            got = float(scored["baseline"]["weekday"]["mae"])
            expected = _published(published, source, seed)
            hours = int(scored["baseline"]["weekday"]["hours"])
            if expected is not None and abs(got - expected) > 1e-8:
                failures.append(f"{source} seed {seed} CL weekday {got:.8f} != {expected:.8f}")
            if seed in CHECK_SEEDS:
                logged = json.loads((ROOT / "logs" / f"G1C-{source.upper()}2DC_seed{seed}.json").read_text(encoding="utf-8"))
                raw = loop._score_arrays()(pred[TEST_HOUR:SERIES_HOUR], truth[TEST_HOUR:SERIES_HOUR], scale)
                if abs(float(raw["mae"]) - float(logged["test"]["mae"])) > 1e-3:
                    failures.append(f"{source} seed {seed} fine-tune {raw['mae']:.6f} != log {logged['test']['mae']:.6f}")
            if hours != 1032:
                failures.append(f"{source} seed {seed} weekday hours {hours}")
            record = {
                "source": source,
                "seed": seed,
                "baseline": {
                    "weekday_mae": got,
                    "weekday_rmse": float(scored["baseline"]["weekday"]["rmse"]),
                    "weekend_mae": float(scored["baseline"]["weekend"]["mae"]),
                },
                "arms": {str(votes): _arm_record(scored, votes) for votes in VOTE_COUNTS},
            }
            runs.append(record)
            if seed == 42:
                seed42[source] = {
                    "arm": scored["outputs"][PRIMARY_VOTES],
                    "base": scored["baseline_values"],
                    "truth": truth,
                }
            primary = record["arms"][str(PRIMARY_VOTES)]
            print(
                f"  2 votes {primary['weekday_mae']:.4f}  CL {got:.4f}  added {primary['mean_cells']:.2f}",
                flush=True,
            )
    if failures:
        for item in failures:
            print(item, file=sys.stderr)
        return 1
    diff = {}
    rmse_ok = True
    for source in SOURCES:
        bundle = seed42[source]
        diff[source] = loop._weekday_bootstrap(bundle["arm"], bundle["base"], bundle["truth"], TEST_DAY, SERIES_DAY)
        row = next(item for item in runs if item["source"] == source and item["seed"] == 42)
        rmse_ok = rmse_ok and row["arms"][str(PRIMARY_VOTES)]["weekday_rmse"] <= row["baseline"]["weekday_rmse"] + 1e-12
    signs = []
    for source in SOURCES:
        for seed in SEEDS:
            row = next(item for item in runs if item["source"] == source and item["seed"] == seed)
            signs.append(float(np.sign(row["arms"][str(PRIMARY_VOTES)]["weekday_mae"] - row["baseline"]["weekday_mae"])))
    verdict = {
        "primary_votes": PRIMARY_VOTES,
        "word": _judge(diff, rmse_ok),
        "bootstrap": diff,
        "stable": len(set(signs)) == 1 and signs[0] != 0.0,
        "signs": signs,
    }
    payload = {"passed": True, "failures": [], "runs": runs, "verdict": verdict}
    loop._write_json(out_dir / "add_below_gate.json", payload)
    md_path = out_dir / "add_below_gate.md"
    md_path.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print(verdict["word"], flush=True)
    print(f"wrote {md_path}", flush=True)
    return 0


def _fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _find(runs: list[dict[str, Any]], source: str, seed: int) -> dict[str, Any]:
    return next(item for item in runs if item["source"] == source and int(item["seed"]) == seed)


def _markdown(payload: dict[str, Any]) -> str:
    runs = payload["runs"]
    verdict = payload["verdict"]
    lines = [
        "# 150 门以下的格子按留出日加入",
        "",
        "协议在打分前写在 [ADD_BELOW_GATE_PLAN.md](../ADD_BELOW_GATE_PLAN.md)。",
        "50 格每天都校正。另外的格子要在最近三个同类日的留出日里赢够票，当天才使用同一套系数。",
        "",
        f"主比较是 2 票减 CL，种子 42：{verdict['word']}。方向{'稳定' if verdict['stable'] else '不稳定'}。",
        "",
        "## 工作日测试 MAE",
        "",
        "| 任务 | CL | 1 票 | 2 票 | 3 票 | 2 票新加入 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source in SOURCES:
        for seed in SEEDS:
            row = _find(runs, source, seed)
            lines.append(
                f"| {source} {seed} | {_fmt(row['baseline']['weekday_mae'])} | "
                f"{_fmt(row['arms']['1']['weekday_mae'])} | {_fmt(row['arms']['2']['weekday_mae'])} | "
                f"{_fmt(row['arms']['3']['weekday_mae'])} | {_fmt(row['arms']['2']['mean_cells'], 2)} |"
            )
    lines.extend(["", "## 新加入格子自己的绝对误差", "", "合计是这些格子在被加入的工作日小时上的绝对误差。", ""])
    lines.append("| 任务 | 微调 | 2 票 | 点数 |")
    lines.append("|---|---:|---:|---:|")
    for source in SOURCES:
        for seed in SEEDS:
            added = _find(runs, source, seed)["arms"]["2"]
            lines.append(f"| {source} {seed} | {_fmt(added['base_ae'], 1)} | {_fmt(added['corr_ae'], 1)} | {added['points']:.0f} |")
    boot = verdict["bootstrap"]
    lines.extend(
        [
            "",
            "## 判定",
            "",
            (
                f"种子 42，2 票对 CL：{verdict['word']}。"
                f"纽约 {_fmt(boot['ny']['point'])} [{_fmt(boot['ny']['low'])}, {_fmt(boot['ny']['high'])}]；"
                f"芝加哥 {_fmt(boot['chi']['point'])} [{_fmt(boot['chi']['low'])}, {_fmt(boot['chi']['high'])}]。"
            ),
            f"{len(SOURCES) * len(SEEDS)} 个任务的工作日 MAE 差，符号{'全部相同' if verdict['stable'] else '不完全相同'}。",
            "",
            "## 局限",
            "",
            "1 票和 3 票不是主比较。票数没有在验证集上重选。",
            "测试段在先前的实验里已经看过。闭环使用目标日之前已经发生的实际上车量。",
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
