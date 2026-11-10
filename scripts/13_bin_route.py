#!/usr/bin/env python3
"""Route each Washington cell to an already trained 3-day model.

Taxi pickup only. The short-window bin is the Monday–Friday profile on
[5832, 5904), holidays kept, matching scripts/06_bin_agreement.py. A cell
above 0.8 uses that source city's fine-tune. A cell above 0.5 and at most
0.8 uses that city's random-init distillation. The rest use the Washington
scratch model. No training and no checkpoint writes.
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

from src.data.dataset import prepare_city_windows
from src.models.st_net import STNet
from src.utils.config import load_config

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover
    torch = None
    DataLoader = None  # type: ignore

PUBLISHED_ATOL = 1e-4
CELL_ATOL = 2e-4
BIN_COUNTS = {
    "ny": {"le_0.5": 3, "gt_0.5_le_0.8": 111, "gt_0.8": 75, "n_undefined": 18, "n_same": 88, "n_both": 189},
    "chi": {"le_0.5": 3, "gt_0.5_le_0.8": 100, "gt_0.8": 86, "n_undefined": 18, "n_same": 95, "n_both": 189},
}
PUBLISHED = {
    "scratch": {"val": (2.1333, 6.2564), "test": (1.8701, 5.5528)},
    "ny_finetune": {"val": (2.0548, 6.1937), "test": (1.7153, 5.2786)},
    "chi_finetune": {"val": (1.7130, 5.5809), "test": (1.4773, 5.0578)},
    "ny_kd": {"val": (2.0248, 6.3896), "test": (1.7519, 5.5064)},
    "chi_kd": {"val": (2.0314, 6.1986), "test": (1.7485, 5.4130)},
}
CHECKPOINTS = {
    "scratch": "baseline_dc_3day_stnet.pt",
    "ny_finetune": "finetune_ny2dc.pt",
    "chi_finetune": "finetune_chi2dc.pt",
    "ny_kd": "student_kd_ny2dc.pt",
    "chi_kd": "student_kd_chi2dc.pt",
}
SOURCES = ("ny", "chi")
CITY = {"ny": "纽约", "chi": "芝加哥"}
METHOD_LABEL = {
    "scratch": "从零",
    "finetune": "纯微调",
    "kd": "纯蒸馏",
    "route": "按档拼接",
}
BIN_LABEL = {
    "high": "高于 0.8",
    "mid": "0.5 到 0.8",
    "low": "不超过 0.5 或无定义",
}


def route_choice(bin_code: int) -> str:
    """Map one similarity bin to a whole-city model. Undefined uses scratch."""
    if int(bin_code) == 2:
        return "finetune"
    if int(bin_code) == 1:
        return "kd"
    return "scratch"


def route_choices(bins: np.ndarray) -> np.ndarray:
    choices = np.full(bins.shape, "scratch", dtype=object)
    choices[np.asarray(bins) == 1] = "kd"
    choices[np.asarray(bins) == 2] = "finetune"
    return choices


def metrics_from_cells(
    se: np.ndarray,
    ae: np.ndarray,
    ape: np.ndarray,
    mape_points: np.ndarray,
    hours: float,
    scale: float,
) -> dict[str, float]:
    """City MAE / RMSE / MAPE from per-cell sums. Same reduction as training."""
    n_cells = int(np.asarray(se).shape[0])
    valid_points = float(hours) * n_cells
    if n_cells <= 0 or valid_points <= 0:
        raise ValueError("no cells to score")
    se_sum = float(np.sum(se))
    mape_denom = float(np.sum(mape_points))
    mape = float("nan") if mape_denom <= 0 else float(np.sum(ape)) / mape_denom * 100.0
    rmse_norm = (se_sum / valid_points) ** 0.5
    return {
        "mae": float(np.sum(ae)) / valid_points,
        "rmse": rmse_norm * float(scale),
        "mape": mape,
    }


def stitch_cells(
    totals: dict[str, dict[str, Any]],
    choices: np.ndarray,
) -> dict[str, Any]:
    """Take each cell's sums from the model named in ``choices``."""
    choices = np.asarray(choices)
    names = ("scratch", "finetune", "kd")
    missing = [name for name in names if name not in totals]
    if missing:
        raise ValueError(f"missing model totals: {missing}")
    hours = float(totals["scratch"]["hours"])
    for name in names:
        if float(totals[name]["hours"]) != hours:
            raise ValueError(f"{name} hours {totals[name]['hours']} != {hours}")
        if totals[name]["se"].shape != choices.shape:
            raise ValueError(f"{name} cells {totals[name]['se'].shape} != {choices.shape}")
    se = np.empty(choices.shape, dtype=np.float64)
    ae = np.empty(choices.shape, dtype=np.float64)
    ape = np.empty(choices.shape, dtype=np.float64)
    mape_points = np.empty(choices.shape, dtype=np.float64)
    for name in names:
        picked = choices == name
        se[picked] = totals[name]["se"][picked]
        ae[picked] = totals[name]["ae"][picked]
        ape[picked] = totals[name]["ape"][picked]
        mape_points[picked] = totals[name]["mape_points"][picked]
    if not np.isin(choices, list(names)).all():
        raise ValueError("a cell was not assigned to scratch, finetune, or kd")
    return {"se": se, "ae": ae, "ape": ape, "mape_points": mape_points, "hours": hours}


def _load_bin_module():
    spec = importlib.util.spec_from_file_location(
        "bin_agreement",
        ROOT / "scripts" / "06_bin_agreement.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load scripts/06_bin_agreement.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_bins(processed_dir: Path) -> dict[str, np.ndarray]:
    report = _load_bin_module().build_agreement(processed_dir)
    bins = {}
    for item in report["pairs"]:
        source = item["source"]
        expected = BIN_COUNTS[source]
        counts = item["protocol_counts"]
        if item["n_same"] != expected["n_same"] or item["n_both_defined"] != expected["n_both"]:
            raise ValueError(
                f"{source} agreement {item['n_same']}/{item['n_both_defined']} "
                f"!= {expected['n_same']}/{expected['n_both']}"
            )
        for key in ("le_0.5", "gt_0.5_le_0.8", "gt_0.8", "n_undefined"):
            if counts[key] != expected[key]:
                raise ValueError(f"{source} short-window {key}={counts[key]} != {expected[key]}")
        protocol = np.asarray(item["protocol_bins"])
        if int((protocol == 2).sum()) != expected["gt_0.8"]:
            raise ValueError(f"{source} high-bin count drifted")
        bins[source] = protocol
    return bins


def _scalar_metrics(scalar: dict[str, float], scale: float) -> dict[str, float]:
    """Match ``metrics_from_totals``: one float32 reduction per batch, then Python floats."""
    valid_points = float(scalar["valid_points"])
    if valid_points <= 0:
        raise ValueError("evaluation saw no valid points")
    rmse_norm = (float(scalar["se"]) / valid_points) ** 0.5
    mape_points = float(scalar["mape_points"])
    mape = float("nan") if mape_points <= 0 else float(scalar["ape"]) / mape_points * 100.0
    return {
        "mae": float(scalar["ae_raw"]) / valid_points,
        "rmse": rmse_norm * float(scale),
        "mape": mape,
    }


def _check_published(name: str, split: str, metrics: dict[str, float]) -> None:
    mae, rmse = PUBLISHED[name][split]
    if abs(metrics["mae"] - mae) > PUBLISHED_ATOL or abs(metrics["rmse"] - rmse) > PUBLISHED_ATOL:
        raise ValueError(
            f"{name} {split} mae/rmse {metrics['mae']:.6f}/{metrics['rmse']:.6f} "
            f"!= {mae:.4f}/{rmse:.4f}"
        )


def _empty_cells(n_cells: int) -> dict[str, Any]:
    return {
        "se": np.zeros(n_cells, dtype=np.float64),
        "ae": np.zeros(n_cells, dtype=np.float64),
        "ape": np.zeros(n_cells, dtype=np.float64),
        "mape_points": np.zeros(n_cells, dtype=np.float64),
        "hours": 0.0,
    }


def _accumulate(net, loader, mask, device, scale: float, min_val: float) -> dict[str, Any]:
    if torch is None:
        raise ImportError("PyTorch is required")
    net.eval()
    valid = mask.reshape(-1).bool()
    n_cells = int(valid.sum().item())
    totals = _empty_cells(n_cells)
    scalar = {"se": 0.0, "ae_raw": 0.0, "ape": 0.0, "mape_points": 0.0, "hours": 0.0}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = net(x, spatial_mask=mask)
            y_valid = y.reshape(y.shape[0], 1, -1)[:, :, valid]
            if out.shape != y_valid.shape:
                raise ValueError(f"prediction {tuple(out.shape)} != target {tuple(y_valid.shape)}")
            diff = out - y_valid
            totals["se"] += diff.detach().double().pow(2).sum(dim=(0, 1)).cpu().numpy()
            scalar["se"] += float(diff.pow(2).sum().item())
            y_raw = y_valid * scale + min_val
            p_raw = out * scale + min_val
            gap = (p_raw - y_raw).abs()
            totals["ae"] += gap.detach().double().sum(dim=(0, 1)).cpu().numpy()
            scalar["ae_raw"] += float(gap.sum().item())
            positive = y_raw > 0
            ape = torch.zeros_like(gap)
            ape[positive] = gap[positive] / y_raw[positive]
            totals["ape"] += ape.detach().double().sum(dim=(0, 1)).cpu().numpy()
            totals["mape_points"] += positive.detach().double().sum(dim=(0, 1)).cpu().numpy()
            scalar["ape"] += float(ape.sum().item())
            scalar["mape_points"] += float(positive.sum().item())
            totals["hours"] += float(x.shape[0])
            scalar["hours"] += float(x.shape[0])
    if totals["hours"] <= 0:
        raise ValueError("evaluation saw no windows")
    scalar["valid_points"] = scalar["hours"] * n_cells
    totals["scalar"] = scalar
    return totals


def _load_net(path: Path, mask, device, model_cfg: dict[str, Any]):
    if torch is None:
        raise ImportError("PyTorch is required")
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint missing: {path}")
    blob = torch.load(path, map_location=device, weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    net = STNet(
        in_channels=1,
        channels=int(model_cfg.get("channels", 64)),
        residual_blocks=int(model_cfg.get("residual_blocks", 3)),
        lstm_hidden=int(model_cfg.get("lstm_hidden", 128)),
        pred_len=int(model_cfg.get("pred_len", 1)),
        spatial_mask=mask,
    ).to(device)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


def _score_model(net, loaders: dict[str, Any], mask, device, scale: float, min_val: float) -> dict[str, dict[str, Any]]:
    return {
        split: _accumulate(net, loader, mask, device, scale, min_val)
        for split, loader in loaders.items()
    }


def _bin_masks(bins: np.ndarray) -> dict[str, np.ndarray]:
    bins = np.asarray(bins)
    return {
        "high": bins == 2,
        "mid": bins == 1,
        "low": bins <= 0,
    }


def _route_group_mae(totals: dict[str, dict[str, Any]], choices: np.ndarray, mask: np.ndarray, scale: float) -> float:
    hours = float(totals["scratch"]["hours"])
    se = np.empty(int(mask.sum()), dtype=np.float64)
    ae = np.empty(int(mask.sum()), dtype=np.float64)
    ape = np.empty(int(mask.sum()), dtype=np.float64)
    mape_points = np.empty(int(mask.sum()), dtype=np.float64)
    offset = 0
    for index in np.flatnonzero(mask):
        name = str(choices[index])
        se[offset] = totals[name]["se"][index]
        ae[offset] = totals[name]["ae"][index]
        ape[offset] = totals[name]["ape"][index]
        mape_points[offset] = totals[name]["mape_points"][index]
        offset += 1
    return metrics_from_cells(se, ae, ape, mape_points, hours, scale)["mae"]


def _cache_dir() -> Path:
    return ROOT / "logs" / "bin_route_cache"


def _save_cached(name: str, split: str, totals: dict[str, Any]) -> None:
    folder = _cache_dir()
    folder.mkdir(parents=True, exist_ok=True)
    scalar = totals["scalar"]
    np.savez(
        folder / f"{name}_{split}.npz",
        se=totals["se"],
        ae=totals["ae"],
        ape=totals["ape"],
        mape_points=totals["mape_points"],
        hours=np.float64(totals["hours"]),
        scalar_se=np.float64(scalar["se"]),
        scalar_ae=np.float64(scalar["ae_raw"]),
        scalar_ape=np.float64(scalar["ape"]),
        scalar_mape=np.float64(scalar["mape_points"]),
        scalar_hours=np.float64(scalar["hours"]),
        scalar_valid=np.float64(scalar["valid_points"]),
    )


def _load_cached(name: str, split: str) -> dict[str, Any] | None:
    path = _cache_dir() / f"{name}_{split}.npz"
    if not path.is_file():
        return None
    blob = np.load(path)
    return {
        "se": blob["se"],
        "ae": blob["ae"],
        "ape": blob["ape"],
        "mape_points": blob["mape_points"],
        "hours": float(blob["hours"]),
        "scalar": {
            "se": float(blob["scalar_se"]),
            "ae_raw": float(blob["scalar_ae"]),
            "ape": float(blob["scalar_ape"]),
            "mape_points": float(blob["scalar_mape"]),
            "hours": float(blob["scalar_hours"]),
            "valid_points": float(blob["scalar_valid"]),
        },
    }


def build_report(processed_dir: Path, checkpoint_dir: Path, device_name: str) -> dict[str, Any]:
    if torch is None or DataLoader is None:
        raise ImportError("PyTorch is required")
    bins = _check_bins(processed_dir)
    cfg = load_config(ROOT / "configs" / "default.yaml")
    prepared = prepare_city_windows("dc", processed_dir, few_shot_days=3)
    mask = torch.as_tensor(prepared["mask"])
    n_valid = int(mask.sum().item())
    for source, protocol in bins.items():
        if protocol.shape != (n_valid,):
            raise ValueError(f"{source} bins {protocol.shape} != valid cells {n_valid}")
    scale = float(prepared["max_val"] - prepared["min_val"])
    min_val = float(prepared["min_val"])
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    if device_name == "cpu":
        torch.set_num_threads(4)
    device = torch.device(device_name)
    loaders = {
        "val": DataLoader(prepared["val"], batch_size=32, shuffle=False),
        "test": DataLoader(prepared["test"], batch_size=32, shuffle=False),
    }
    model_cfg = cfg.get("model", {})
    scored: dict[str, dict[str, dict[str, Any]]] = {}
    for name, filename in CHECKPOINTS.items():
        print(f"scoring {name}", flush=True)
        cached = {split: _load_cached(name, split) for split in ("val", "test")}
        if all(item is not None for item in cached.values()):
            scored[name] = cached
        else:
            net = _load_net(checkpoint_dir / filename, mask, device, model_cfg)
            scored[name] = _score_model(net, loaders, mask, device, scale, min_val)
            del net
            if device.type == "cuda":
                torch.cuda.empty_cache()
            for split, totals in scored[name].items():
                _save_cached(name, split, totals)
        val_metrics = _scalar_metrics(scored[name]["val"]["scalar"], scale)
        print(
            f"replayed {name} val {val_metrics['mae']:.4f}/{val_metrics['rmse']:.4f}",
            flush=True,
        )
    mismatches = []
    for name, splits in scored.items():
        for split, totals in splits.items():
            replay = _scalar_metrics(totals["scalar"], scale)
            cell_metrics = metrics_from_cells(
                totals["se"], totals["ae"], totals["ape"], totals["mape_points"], totals["hours"], scale
            )
            totals["metrics"] = cell_metrics
            totals["replay"] = replay
            try:
                _check_published(name, split, replay)
            except ValueError as exc:
                mismatches.append(str(exc))
            if (
                abs(cell_metrics["mae"] - replay["mae"]) > CELL_ATOL
                or abs(cell_metrics["rmse"] - replay["rmse"]) > CELL_ATOL
            ):
                mismatches.append(
                    f"{name} {split} per-cell {cell_metrics['mae']:.6f}/{cell_metrics['rmse']:.6f} "
                    f"vs replay {replay['mae']:.6f}/{replay['rmse']:.6f}"
                )
    if mismatches:
        raise ValueError("published replay failed:\n" + "\n".join(mismatches))

    cities = []
    for source in SOURCES:
        choices = route_choices(bins[source])
        city_totals = {
            "scratch": None,
            "finetune": scored[f"{source}_finetune"],
            "kd": scored[f"{source}_kd"],
        }
        splits = {}
        for split in ("val", "test"):
            bundle = {
                "scratch": scored["scratch"][split],
                "finetune": city_totals["finetune"][split],
                "kd": city_totals["kd"][split],
            }
            routed = stitch_cells(bundle, choices)
            route_metrics = metrics_from_cells(
                routed["se"], routed["ae"], routed["ape"], routed["mape_points"], routed["hours"], scale
            )
            methods = {
                "scratch": scored["scratch"][split]["metrics"],
                "finetune": bundle["finetune"]["metrics"],
                "kd": bundle["kd"]["metrics"],
                "route": route_metrics,
            }
            groups = {}
            for label, selected in _bin_masks(bins[source]).items():
                groups[label] = {
                    "n": int(selected.sum()),
                    "model": route_choice(2 if label == "high" else 1 if label == "mid" else 0),
                    "mae": _route_group_mae(bundle, choices, selected, scale),
                    "scratch": metrics_from_cells(
                        bundle["scratch"]["se"][selected],
                        bundle["scratch"]["ae"][selected],
                        bundle["scratch"]["ape"][selected],
                        bundle["scratch"]["mape_points"][selected],
                        bundle["scratch"]["hours"],
                        scale,
                    )["mae"],
                    "finetune": metrics_from_cells(
                        bundle["finetune"]["se"][selected],
                        bundle["finetune"]["ae"][selected],
                        bundle["finetune"]["ape"][selected],
                        bundle["finetune"]["mape_points"][selected],
                        bundle["finetune"]["hours"],
                        scale,
                    )["mae"],
                    "kd": metrics_from_cells(
                        bundle["kd"]["se"][selected],
                        bundle["kd"]["ae"][selected],
                        bundle["kd"]["ape"][selected],
                        bundle["kd"]["mape_points"][selected],
                        bundle["kd"]["hours"],
                        scale,
                    )["mae"],
                }
            splits[split] = {"methods": methods, "groups": groups}
        cities.append(
            {
                "source": source,
                "n_finetune": int((choices == "finetune").sum()),
                "n_kd": int((choices == "kd").sum()),
                "n_scratch": int((choices == "scratch").sum()),
                "splits": splits,
            }
        )
    return {
        "channel": "taxi_pickup",
        "seed": 42,
        "trained": False,
        "device": device_name,
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


def _method_table(methods: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| 方法 | MAE | RMSE | MAPE |",
        "|---|---:|---:|---:|",
    ]
    for key in ("scratch", "finetune", "kd", "route"):
        item = methods[key]
        lines.append(f"| {METHOD_LABEL[key]} | {_fmt(item['mae'])} | {_fmt(item['rmse'])} | {_fmt(item['mape'])} |")
    return lines


def _group_table(groups: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| 档 | 格子数 | 所用模型 | 该档 MAE | 从零 | 微调 | 蒸馏 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    model_name = {"scratch": "从零", "finetune": "微调", "kd": "蒸馏"}
    for key in ("high", "mid", "low"):
        item = groups[key]
        lines.append(
            f"| {BIN_LABEL[key]} | {item['n']} | {model_name[item['model']]} | {_fmt(item['mae'])} | "
            f"{_fmt(item['scratch'])} | {_fmt(item['finetune'])} | {_fmt(item['kd'])} |"
        )
    return lines


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 三天格子分流",
        "",
        "通道是出租上车。种子 42。这一步没有训练，也没有改权重。",
        "每个华盛顿有效格用三天窗口的最大相关选一个已经训好的整城模型。",
        "相关大于 0.8 用该源城的纯微调。大于 0.5 且不超过 0.8 用该源城的随机初始化蒸馏。",
        "不超过 0.5，以及短轮廓没有定义，用华盛顿从零训练的模型。",
        "切点不在验证集或测试集上改。纽约和芝加哥分开选，因为档位不是同一个。",
        "五个模型先按训练时的求和重放。MAE 和 RMSE 与已发表的四位小数相差不超过 0.0001，才写这张表。",
        "表里的数字来自按格累加，拼接和三个纯模型用的是同一种算法。",
        "",
    ]
    for city in report["cities"]:
        val = city["splits"]["val"]["methods"]
        direction = "更高" if val["route"]["mae"] > val["finetune"]["mae"] else "更低"
        if abs(val["route"]["mae"] - val["finetune"]["mae"]) <= PUBLISHED_ATOL:
            direction = "相同"
        lines.extend(
            [
                f"## {CITY[city['source']]} → 华盛顿",
                "",
                f"微调 {city['n_finetune']} 格，蒸馏 {city['n_kd']} 格，从零 {city['n_scratch']} 格。",
                "",
                "验证集：",
                "",
                *_method_table(city["splits"]["val"]["methods"]),
                "",
                (
                    f"验证集上，拼接 MAE {_fmt(val['route']['mae'])}，"
                    f"纯微调 {_fmt(val['finetune']['mae'])}，拼接{direction}。"
                ),
                "",
                "测试集：",
                "",
                *_method_table(city["splits"]["test"]["methods"]),
                "",
                "测试集照实报告，不改 0.5 和 0.8。",
                "",
                "验证集各档 MAE：",
                "",
                *_group_table(city["splits"]["val"]["groups"]),
                "",
                "测试集各档 MAE：",
                "",
                *_group_table(city["splits"]["test"]["groups"]),
                "",
            ]
        )
    lines.extend(
        [
            "重算：",
            "",
            "```bash",
            "python scripts/13_bin_route.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(processed_dir: Path, checkpoint_dir: Path, out_dir: Path, device_name: str) -> dict[str, Any]:
    report = build_report(processed_dir, checkpoint_dir, device_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "bin_route.md"
    json_path = out_dir / "bin_route.json"
    json_path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(f"wrote {md_path}")
    for city in report["cities"]:
        val = city["splits"]["val"]["methods"]["route"]
        test = city["splits"]["test"]["methods"]["route"]
        print(
            f"  {city['source']} route val {_fmt(val['mae'])}/{_fmt(val['rmse'])} "
            f"test {_fmt(test['mae'])}/{_fmt(test['rmse'])}"
        )
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
        print(f"bin route failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
