#!/usr/bin/env python3
"""Refit the published line and kappa-3 hour bias on recent days of the same class.

Protocol: docs/CLOCK_HOUR_LOOP_PLAN.md. ``score-val`` never opens a test cache.
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

from src.feedback.published_hour import (  # noqa: E402
    ARMS,
    DAY_HOURS,
    PREFIX_HOUR,
    SERIES_DAY,
    SERIES_HOUR,
    TEST_DAY,
    TEST_HOUR,
    TRAIN_HOUR,
    VAL_DAY,
    VAL_HOUR,
    correct_range,
    day_date,
    fit_window,
    weekday_rows,
    window_for,
)

N_CELLS = 207
N_GATE = 50
SOURCES = ("ny", "chi")
SEEDS = (42, 43, 44, 45, 46)
ANCHOR_SEEDS = (42, 43, 44)
NEW_SEEDS = (45, 46)
ATOL_ANCHOR = 1e-6
ATOL_LOG = 1e-3
ATOL_OVERLAP = 1e-3
WEEKDAY_TEST_HOURS = 1032
PUBLISHED_TEST = {
    ("ny", 42): {"O72": 1.705419251350448, "OL": 1.6681090485403915},
    ("ny", 43): {"O72": 1.627244378502283, "OL": 1.5977687562203848},
    ("ny", 44): {"O72": 1.6175237957134116, "OL": 1.5971484717075715},
    ("chi", 42): {"O72": 1.4415295244856239, "OL": 1.4181985999585542},
    ("chi", 43): {"O72": 1.5050672717702502, "OL": 1.4858972558150847},
    ("chi", 44): {"O72": 1.5064227038919895, "OL": 1.4911081562782498},
}
PUBLISHED_VAL = {
    ("ny", 42): {"O72": 2.03601459370462, "OL": 1.9807708066634078},
    ("ny", 43): {"O72": 1.9406103291816612, "OL": 1.8932748788416505},
    ("ny", 44): {"O72": 1.9437238747450438, "OL": 1.9057501620152275},
    ("chi", 42): {"O72": 1.6863351164205158, "OL": 1.6374454942277492},
    ("chi", 43): {"O72": 1.7495140836384444, "OL": 1.7135964546218025},
    ("chi", 44): {"O72": 1.7575236848823623, "OL": 1.714253403220479},
}
ARM_LABEL = {
    "B0": "一律微调",
    "O72": "旧公式 λ=72",
    "OL": "训练窗上的钟点偏差",
    "RL": "只滚动直线",
    "CL": "同类日滚动",
    "CL72": "连续 72 小时滚动",
}


def _load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scripts/{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _solve_old():
    return _load("19_cell_affine_v2.py", "cell_affine_v2_loop").solve_old


def _score_arrays():
    return _load("17_cell_affine.py", "cell_affine_score_loop").score_arrays


def _bootstrap():
    return _load("19_cell_affine_v2.py", "cell_affine_v2_boot_loop").day_block_bootstrap


def _checkpoint(directory: Path, source: str, seed: int) -> Path:
    stem = f"finetune_{source}2dc" if seed == 42 else f"finetune_{source}2dc_seed{seed}"
    path = directory / f"{stem}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _v2_cache(cache_dir: Path, source: str, seed: int, split: str) -> Path:
    return cache_dir / f"{source}_seed{seed}_{split}.npz"


def _new_cache(cache_dir: Path, source: str, seed: int, split: str) -> Path:
    return cache_dir / f"{source}_seed{seed}_{split}.npz"


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as blob:
        return {key: blob[key] for key in blob.files}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _default_v2(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates = [
        ROOT / "logs" / "cell_affine_v2_cache",
        ROOT.parent / "sw-affine-v2" / "logs" / "cell_affine_v2_cache",
    ]
    for path in candidates:
        if (path / "ny_seed42_train.npz").is_file():
            return path
    raise FileNotFoundError("cell_affine_v2 cache not found; pass --v2-cache")


def _collect(net, dataset, indices: np.ndarray, mask, device, value_scale: float, min_val: float) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader, Dataset

    class _Index(Dataset):
        def __init__(self, base, rows: np.ndarray) -> None:
            self.base = base
            self.rows = np.asarray(rows, dtype=np.int64)

        def __len__(self) -> int:
            return int(self.rows.size)

        def __getitem__(self, item: int):
            return self.base[int(self.rows[item])]

    loader = DataLoader(_Index(dataset, indices), batch_size=32, shuffle=False)
    valid = mask.reshape(-1).bool()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device)
            target = target.to(device)
            out = net(inputs, spatial_mask=mask)
            y_valid = target.reshape(target.shape[0], 1, -1)[:, :, valid]
            preds.append((out * value_scale + min_val).detach().double().cpu().numpy()[:, 0, :])
            trues.append((y_valid * value_scale + min_val).detach().double().cpu().numpy()[:, 0, :])
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def stage_prefix(v2_cache: Path, out_dir: Path, processed: Path, checkpoints: Path, device_name: str, seeds: tuple[int, ...] = SEEDS) -> int:
    import torch
    from src.data.dataset import prepare_city_windows
    from src.utils.config import load_config

    route = _load("13_bin_route.py", "bin_route_prefix")
    prepared = prepare_city_windows("dc", processed, few_shot_days=0)
    value_scale = float(prepared["max_val"] - prepared["min_val"])
    min_val = float(prepared["min_val"])
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    if device_name == "cpu":
        torch.set_num_threads(4)
    device = torch.device(device_name)
    mask = torch.as_tensor(prepared["mask"])
    model_cfg = load_config(ROOT / "configs" / "default.yaml").get("model", {})
    reference = _read_npz(_v2_cache(v2_cache, "ny", 42, "train"))
    eligible = np.asarray(reference["eligible"], dtype=bool)
    if int(eligible.sum()) != N_GATE or eligible.shape != (N_CELLS,):
        raise RuntimeError(f"seed 42 gate is {int(eligible.sum())}, expected {N_GATE}")
    if abs(float(reference["value_scale"]) - value_scale) > 1e-6:
        raise RuntimeError("value scale differs from the cell_affine_v2 cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    prefix_rows = np.arange(PREFIX_HOUR - 6, TRAIN_HOUR - 6)
    overlap_rows = np.arange(TRAIN_HOUR - 6, VAL_HOUR - 6)
    for source in SOURCES:
        for seed in seeds:
            path = _checkpoint(checkpoints, source, seed)
            print(f"prefix {source} seed {seed}", flush=True)
            net = route._load_net(path, mask, device, model_cfg)
            prefix_pred, prefix_true = _collect(net, prepared["train"], prefix_rows, mask, device, value_scale, min_val)
            overlap_pred, overlap_true = _collect(net, prepared["train"], overlap_rows, mask, device, value_scale, min_val)
            if prefix_pred.shape != (TRAIN_HOUR - PREFIX_HOUR, N_CELLS):
                raise RuntimeError(f"{source} seed {seed} prefix shape {prefix_pred.shape}")
            np.savez_compressed(
                _new_cache(out_dir, source, seed, "prefix"),
                prediction=prefix_pred,
                target=prefix_true,
                value_scale=np.float64(value_scale),
            )
            if seed in ANCHOR_SEEDS:
                cached = _read_npz(_v2_cache(v2_cache, source, seed, "train"))
                gap = float(np.max(np.abs(overlap_pred - cached["prediction"])))
                gap_y = float(np.max(np.abs(overlap_true - cached["target"])))
                if gap > ATOL_OVERLAP or gap_y > ATOL_OVERLAP:
                    failures.append(f"{source} seed {seed} overlap max abs pred {gap:.3e} target {gap_y:.3e}")
                if not np.array_equal(np.asarray(cached["eligible"], dtype=bool), eligible):
                    failures.append(f"{source} seed {seed} gate differs from seed 42")
            else:
                ranges = {
                    "train": (prepared["train"], overlap_rows),
                    "val": (prepared["val"], np.arange(TEST_HOUR - VAL_HOUR)),
                    "test": (prepared["test"], np.arange(SERIES_HOUR - TEST_HOUR)),
                }
                for split, (dataset, rows) in ranges.items():
                    pred, true = (overlap_pred, overlap_true) if split == "train" else _collect(
                        net, dataset, rows, mask, device, value_scale, min_val
                    )
                    np.savez_compressed(
                        _new_cache(out_dir, source, seed, split),
                        prediction=pred,
                        target=true,
                        value_scale=np.float64(value_scale),
                        eligible=eligible,
                    )
            del net
            if device.type == "cuda":
                torch.cuda.empty_cache()
    payload = {"passed": not failures, "failures": failures, "device": device_name, "n_gate": int(eligible.sum())}
    _write_json(out_dir / "checks.json", payload)
    if failures:
        for item in failures:
            print(item, file=sys.stderr)
        return 1
    print("prefix checks passed", flush=True)
    return 0


def _series(v2_cache: Path, new_cache: Path, source: str, seed: int, splits: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    pred = np.full((SERIES_HOUR, N_CELLS), np.nan, dtype=np.float64)
    truth = np.full((SERIES_HOUR, N_CELLS), np.nan, dtype=np.float64)
    value_scale = None
    eligible = None
    pieces = {
        "prefix": (PREFIX_HOUR, _new_cache(new_cache, source, seed, "prefix")),
        "train": (TRAIN_HOUR, _v2_cache(v2_cache, source, seed, "train") if seed in ANCHOR_SEEDS else _new_cache(new_cache, source, seed, "train")),
        "val": (VAL_HOUR, _v2_cache(v2_cache, source, seed, "val") if seed in ANCHOR_SEEDS else _new_cache(new_cache, source, seed, "val")),
        "test": (TEST_HOUR, _v2_cache(v2_cache, source, seed, "test") if seed in ANCHOR_SEEDS else _new_cache(new_cache, source, seed, "test")),
    }
    for split in splits:
        start, path = pieces[split]
        if split == "test" and "test" not in splits:
            raise PermissionError("test cache is not open in this stage")
        if not path.is_file():
            raise FileNotFoundError(path)
        if split == "test" and "test" in path.name and "test" not in splits:
            raise PermissionError(path)
        loaded = _read_npz(path)
        block_pred = np.asarray(loaded["prediction"], dtype=np.float64)
        block_true = np.asarray(loaded["target"], dtype=np.float64)
        pred[start : start + block_pred.shape[0]] = block_pred
        truth[start : start + block_true.shape[0]] = block_true
        value_scale = float(loaded["value_scale"])
        if "eligible" in loaded:
            eligible = np.asarray(loaded["eligible"], dtype=bool)
    if value_scale is None or eligible is None:
        reference = _read_npz(_v2_cache(v2_cache, "ny", 42, "train"))
        eligible = np.asarray(reference["eligible"], dtype=bool)
        if value_scale is None:
            value_scale = float(reference["value_scale"])
    return pred, truth, value_scale, eligible


def _metrics(prediction: np.ndarray, target: np.ndarray, value_scale: float, first: int, last: int) -> dict[str, dict[str, float]]:
    score = _score_arrays()
    block_pred = prediction[first * DAY_HOURS : last * DAY_HOURS]
    block_true = target[first * DAY_HOURS : last * DAY_HOURS]
    weekend = ~weekday_rows(first, last)
    weekday = ~weekend
    return {
        "all": score(block_pred, block_true, value_scale),
        "weekday": score(block_pred[weekday], block_true[weekday], value_scale),
        "weekend": score(block_pred[weekend], block_true[weekend], value_scale),
    }


def _weekday_bootstrap(arm: np.ndarray, ref: np.ndarray, target: np.ndarray, first: int, last: int) -> dict[str, float]:
    days = [day for day in range(first, last) if not (day_date(day).weekday() >= 5)]
    def _stack(values: np.ndarray) -> np.ndarray:
        return np.concatenate([values[day * DAY_HOURS : (day + 1) * DAY_HOURS] for day in days], axis=0)
    point, low, high = _bootstrap()(_stack(arm), _stack(ref), _stack(target))
    return {"point": point, "low": low, "high": high}


def _run_arms(pred: np.ndarray, truth: np.ndarray, eligible: np.ndarray, solve_old, first: int, last: int) -> dict[str, np.ndarray]:
    return {
        mode: correct_range(pred, truth, eligible, solve_old, mode, first, last)
        for mode in ARMS
    }


def _record(outputs: dict[str, np.ndarray], truth: np.ndarray, value_scale: float, first: int, last: int) -> dict[str, Any]:
    recorded = {}
    for mode, values in outputs.items():
        recorded[mode] = _metrics(values, truth, value_scale, first, last)
    diffs = {}
    for mode in ARMS:
        if mode == "OL":
            continue
        diffs[mode] = _weekday_bootstrap(outputs[mode], outputs["OL"], truth, first, last)
    below = np.nanmax(np.abs(outputs["CL"][first * DAY_HOURS : last * DAY_HOURS] - outputs["B0"][first * DAY_HOURS : last * DAY_HOURS]))
    return {"arms": recorded, "weekday_vs_ol": diffs, "cl_vs_b0_max_abs": float(below)}


def _gate_unchanged(outputs: dict[str, np.ndarray], eligible: np.ndarray, first: int, last: int) -> bool:
    sl = slice(first * DAY_HOURS, last * DAY_HOURS)
    below = ~eligible
    base = outputs["B0"][sl][:, below]
    return all(np.array_equal(values[sl][:, below], base) for values in outputs.values())


def _labor_day_gap(pred: np.ndarray, truth: np.ndarray, eligible: np.ndarray, solve_old) -> float:
    """09-05 uses the same three training days as the frozen arm."""
    day = VAL_DAY + 2
    gap = 0.0
    fitted = {}
    for mode in ("CL", "OL"):
        hours = np.concatenate([np.arange(item * DAY_HOURS, (item + 1) * DAY_HOURS) for item in window_for(day, mode)])
        fitted[mode] = fit_window(pred[hours], truth[hours], eligible, solve_old, with_hour=True)
    for left, right in zip(fitted["CL"], fitted["OL"]):
        gap = max(gap, float(np.max(np.abs(left - right))))
    return gap


def stage_score(v2_cache: Path, new_cache: Path, out_dir: Path, *, test: bool) -> int:
    checks = json.loads((new_cache / "checks.json").read_text(encoding="utf-8"))
    if not checks.get("passed"):
        print("score refused: prefix checks failed", file=sys.stderr)
        return 1
    solve_old = _solve_old()
    splits = ("prefix", "train", "val", "test") if test else ("prefix", "train", "val")
    first = VAL_DAY
    last = SERIES_DAY if test else TEST_DAY
    failures: list[str] = []
    runs = []
    windows = {
        "2016-09-03": [day_date(day).isoformat() for day in window_for(VAL_DAY, "CL")],
        "2016-09-05": [day_date(day).isoformat() for day in window_for(VAL_DAY + 2, "CL")],
    }
    if windows["2016-09-03"] != ["2016-08-21", "2016-08-27", "2016-08-28"]:
        failures.append(f"2016-09-03 window {windows['2016-09-03']}")
    if windows["2016-09-05"] != ["2016-08-31", "2016-09-01", "2016-09-02"]:
        failures.append(f"2016-09-05 window {windows['2016-09-05']}")
    reference_gate = None
    for source in SOURCES:
        for seed in SEEDS:
            print(f"score {'test' if test else 'val'} {source} seed {seed}", flush=True)
            pred, truth, value_scale, eligible = _series(v2_cache, new_cache, source, seed, splits)
            if reference_gate is None:
                reference_gate = eligible.copy()
            elif not np.array_equal(eligible, reference_gate):
                failures.append(f"{source} seed {seed} gate differs")
            if int(eligible.sum()) != N_GATE:
                failures.append(f"{source} seed {seed} gate {int(eligible.sum())}")
            outputs = _run_arms(pred, truth, eligible, solve_old, first, last)
            if not _gate_unchanged(outputs, eligible, first, last):
                failures.append(f"{source} seed {seed} changed a cell below the gate")
            labor = _labor_day_gap(pred, truth, eligible, solve_old)
            if labor > 1e-8:
                failures.append(f"{source} seed {seed} 2016-09-05 coefficients differ by {labor:.3e}")
            span_first, span_last = (TEST_DAY, SERIES_DAY) if test else (VAL_DAY, TEST_DAY)
            record = _record(outputs, truth, value_scale, span_first, span_last)
            record.update({"source": source, "seed": seed})
            if not test and seed in ANCHOR_SEEDS:
                for mode, key in (("O72", "O72"), ("OL", "OL")):
                    got = record["arms"][mode]["all"]["mae"]
                    expected = PUBLISHED_VAL[(source, seed)][key]
                    if abs(got - expected) > ATOL_ANCHOR:
                        failures.append(f"{source} seed {seed} {mode} val MAE {got:.8f} != {expected:.8f}")
            if test and seed in ANCHOR_SEEDS:
                for mode, key in (("O72", "O72"), ("OL", "OL")):
                    got = record["arms"][mode]["all"]["mae"]
                    expected = PUBLISHED_TEST[(source, seed)][key]
                    if abs(got - expected) > ATOL_ANCHOR:
                        failures.append(f"{source} seed {seed} {mode} test MAE {got:.8f} != {expected:.8f}")
            if test and seed in NEW_SEEDS:
                log_path = ROOT / "logs" / f"G1C-{source.upper()}2DC_seed{seed}.json"
                logged = json.loads(log_path.read_text(encoding="utf-8"))["test"]["mae"]
                got = record["arms"]["B0"]["all"]["mae"]
                if abs(got - float(logged)) > ATOL_LOG:
                    failures.append(f"{source} seed {seed} fine-tune {got:.6f} != log {logged:.6f}")
            if test and int(record["arms"]["B0"]["weekday"]["hours"]) != WEEKDAY_TEST_HOURS:
                failures.append(f"weekday hours {record['arms']['B0']['weekday']['hours']}")
            runs.append(record)
            print(
                f"  CL weekday {record['arms']['CL']['weekday']['mae']:.4f}  "
                f"OL {record['arms']['OL']['weekday']['mae']:.4f}",
                flush=True,
            )
    payload: dict[str, Any] = {
        "split": "test" if test else "val",
        "windows": windows,
        "failures": failures,
        "passed": not failures,
        "runs": runs,
    }
    if test:
        val_path = out_dir / "clock_hour_loop_val.json"
        if not val_path.is_file():
            print("score-test refused: validation file is missing", file=sys.stderr)
            return 1
        payload["verdict"] = _verdict(runs)
        if failures:
            print("score-test checks failed", file=sys.stderr)
            for item in failures:
                print(f"  {item}", file=sys.stderr)
            return 1
        _write_json(out_dir / "clock_hour_loop.json", payload)
        md_path = out_dir / "clock_hour_loop.md"
        md_path.write_text(_markdown(payload, json.loads(val_path.read_text(encoding="utf-8"))), encoding="utf-8", newline="\n")
        print(payload["verdict"]["primary"], flush=True)
        print(f"wrote {md_path}", flush=True)
        return 0
    _write_json(out_dir / "clock_hour_loop_val.json", payload)
    if failures:
        print("score-val checks failed", file=sys.stderr)
        for item in failures:
            print(f"  {item}", file=sys.stderr)
        return 1
    print(f"wrote {out_dir / 'clock_hour_loop_val.json'}", flush=True)
    return 0


def _city(runs: list[dict[str, Any]], source: str, seed: int) -> dict[str, Any]:
    return next(item for item in runs if item["source"] == source and int(item["seed"]) == seed)


def _verdict(runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for source in SOURCES:
        item = _city(runs, source, 42)
        diff = item["weekday_vs_ol"]["CL"]
        rmse_ok = item["arms"]["CL"]["weekday"]["rmse"] <= item["arms"]["OL"]["weekday"]["rmse"] + 1e-12
        rows.append({"source": source, **diff, "rmse_ok": rmse_ok})
    if all(row["high"] < 0 and row["rmse_ok"] for row in rows):
        primary = "更低"
    elif any(row["low"] > 0 for row in rows):
        primary = "更高"
    else:
        primary = "无法区分"
    signs = []
    for seed in SEEDS:
        for source in SOURCES:
            signs.append(float(np.sign(_city(runs, source, seed)["weekday_vs_ol"]["CL"]["point"])))
    stable = len(set(signs)) == 1 and signs[0] != 0.0
    return {"primary": primary, "rows": rows, "stable": stable, "signs": signs}


def _fmt(value: float) -> str:
    return f"{float(value):.4f}"


def _interval(row: dict[str, float]) -> str:
    return f"{_fmt(row['point'])} [{_fmt(row['low'])}, {_fmt(row['high'])}]"


def _markdown(test: dict[str, Any], val: dict[str, Any]) -> str:
    lines = [
        "# 已发表钟点偏差的按日闭环",
        "",
        "协议在打分前写在 [CLOCK_HOUR_LOOP_PLAN.md](../CLOCK_HOUR_LOOP_PLAN.md)。",
        "闭环使用了目标日之前已经发生的实际上车量。判定只用周一到周五。",
        "OL 是训练窗上只拟合一次的旧公式再加 κ=3 钟点偏差。CL 每个同类日开始时用最近三个同类日重估。",
        "",
        f"主比较，种子 42，工作日测试 MAE，CL 对 OL：{test['verdict']['primary']}。方向{'稳定' if test['verdict']['stable'] else '不稳定'}。",
        "",
        "## 核对",
        "",
        "- 种子 42 到 44 的全市验证集、测试集 MAE 对上了已发表的 λ=72 和「旧公式再加钟点偏差」，容差 1e-6。",
        "- 种子 45、46 的微调全市测试 MAE 对上了训练日志，容差 1e-3。",
        "- 十个任务的过门名单相同，50 格。157 格与微调的最大绝对差是 0。",
        f"- 2016-09-03 的 CL 窗口是 {', '.join(test['windows']['2016-09-03'])}。",
        f"- 2016-09-05 的 CL 窗口是 {', '.join(test['windows']['2016-09-05'])}。",
        "",
        "## 工作日测试 MAE",
        "",
        "| 臂 | " + " | ".join(f"{source} {seed}" for source in SOURCES for seed in SEEDS) + " |",
        "|---|" + "---:|" * (len(SOURCES) * len(SEEDS)),
    ]
    for mode in ARMS:
        cells = []
        for source in SOURCES:
            for seed in SEEDS:
                cells.append(_fmt(_city(test["runs"], source, seed)["arms"][mode]["weekday"]["mae"]))
        lines.append(f"| {ARM_LABEL[mode]} | " + " | ".join(cells) + " |")
    lines.extend(["", "## 相对训练窗钟点偏差的工作日差", "", "差是该臂减 OL。负值表示该臂更低。", ""])
    lines.append("| 任务 | CL | RL | CL72 |")
    lines.append("|---|---|---|---|")
    for source in SOURCES:
        for seed in SEEDS:
            item = _city(test["runs"], source, seed)
            lines.append(
                f"| {source} {seed} | {_interval(item['weekday_vs_ol']['CL'])} | "
                f"{_interval(item['weekday_vs_ol']['RL'])} | {_interval(item['weekday_vs_ol']['CL72'])} |"
            )
    lines.extend(["", "## 周末测试 MAE（不参与判定）", ""])
    lines.append("| 臂 | " + " | ".join(f"{source} {seed}" for source in SOURCES for seed in SEEDS) + " |")
    lines.append("|---|" + "---:|" * (len(SOURCES) * len(SEEDS)))
    for mode in ("O72", "OL", "CL"):
        cells = [_fmt(_city(test["runs"], source, seed)["arms"][mode]["weekend"]["mae"]) for source in SOURCES for seed in SEEDS]
        lines.append(f"| {ARM_LABEL[mode]} | " + " | ".join(cells) + " |")
    lines.extend(["", "## 验证集工作日 MAE", ""])
    lines.append("| 臂 | " + " | ".join(f"{source} {seed}" for source in SOURCES for seed in SEEDS) + " |")
    lines.append("|---|" + "---:|" * (len(SOURCES) * len(SEEDS)))
    for mode in ("OL", "CL", "RL", "CL72"):
        cells = [_fmt(_city(val["runs"], source, seed)["arms"][mode]["weekday"]["mae"]) for source in SOURCES for seed in SEEDS]
        lines.append(f"| {ARM_LABEL[mode]} | " + " | ".join(cells) + " |")
    detail = "；".join(f"{row['source']} {_interval(row)}" for row in test["verdict"]["rows"])
    lines.extend(
        [
            "",
            "## 判定",
            "",
            f"种子 42，CL 对 OL：{test['verdict']['primary']}。{detail}。",
            f"十个任务的工作日 MAE 差，符号{'全部相同' if test['verdict']['stable'] else '不完全相同'}。",
            "",
            "## 局限",
            "",
            "测试段在这条钟点偏差发表之前就已经看过。这次仍是探索性对照。",
            "周末切片的信息集和 OL 不同：CL 用更早的周六周日重估，OL 用三个工作日的系数。",
            "κ=3 没有因为这条闭环而更换。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prefix", "score-val", "score-test"))
    parser.add_argument("--v2-cache", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "logs" / "clock_hour_cache")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "profiles")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()
    v2_cache = _default_v2(args.v2_cache)
    if args.stage == "prefix":
        seeds = tuple(args.seeds) if args.seeds else SEEDS
        return stage_prefix(v2_cache, args.cache_dir, args.processed_dir, args.checkpoint_dir, args.device, seeds)
    return stage_score(v2_cache, args.cache_dir, args.out_dir, test=args.stage == "score-test")


if __name__ == "__main__":
    raise SystemExit(main())
