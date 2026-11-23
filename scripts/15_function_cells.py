#!/usr/bin/env python3
"""Whether POI function labels can be assigned per 1 km cell, and whether they
change the city error beyond a 3-day volume gate.

POI class order is CrossTReS Appendix C. Labels use only the static POI counts.
The volume gate uses the same three weekdays as the existing bin check.
Per-cell errors are the seed-42 cache from scripts/13_bin_route.py. No training.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import load_city_array, valid_region_mask
from src.utils.pearson_weekday_profile import hourly_calendar

_SPEC = importlib.util.spec_from_file_location("cell_volume", ROOT / "scripts" / "14_cell_volume.py")
assert _SPEC is not None and _SPEC.loader is not None
volume = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(volume)

# CrossTReS Appendix C. Do not rename these indexes.
POI_NAMES = (
    "景点",
    "医疗",
    "生活服务",
    "居住",
    "金融",
    "体育休闲",
    "文化教育",
    "购物",
    "住宿",
    "政府机构",
    "公司",
    "餐饮",
    "交通",
    "公共服务",
)
COMMERCIAL = (4, 7, 8, 10, 11)
RESIDENTIAL = 3
GATE = 150
CITY_CODE = {"ny": "NY", "chi": "CHI", "dc": "DC"}
CITY_NAME = {"ny": "纽约", "chi": "芝加哥", "dc": "华盛顿"}


def function_label(counts: np.ndarray) -> str:
    """One cell from its 14 POI counts. Empty and mixed cells stay unlabeled."""
    values = np.asarray(counts, dtype=np.float64).reshape(-1)
    if values.shape != (14,):
        raise ValueError(f"expected 14 POI classes, got {values.shape}")
    total = float(values.sum())
    if total <= 0:
        return "无兴趣点"
    share = values / total
    commercial = float(share[list(COMMERCIAL)].sum())
    residential = float(share[RESIDENTIAL])
    if commercial >= 0.5 and commercial >= residential + 0.15:
        return "商业就业"
    if residential >= 0.5 and residential >= commercial + 0.15:
        return "居住"
    return "混合"


def label_grid(poi: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if poi.ndim != 3 or poi.shape[2] != 14:
        raise ValueError(f"expected (H, W, 14), got {poi.shape}")
    if poi.shape[:2] != valid.shape:
        raise ValueError(f"POI {poi.shape[:2]} != mask {valid.shape}")
    labels = np.full(valid.shape, "", dtype=object)
    for row, col in zip(*np.nonzero(valid)):
        labels[row, col] = function_label(poi[row, col])
    return labels


def neighbor_same_rate(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Share of valid 8-neighbors with the same label. Cells with none are nan."""
    h, w = valid.shape
    out = np.full(int(valid.sum()), np.nan, dtype=np.float64)
    index = -np.ones(valid.shape, dtype=np.int64)
    index[valid] = np.arange(int(valid.sum()))
    flat_labels = labels[valid]
    for row, col in zip(*np.nonzero(valid)):
        same = 0
        seen = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = row + dr, col + dc
                if rr < 0 or cc < 0 or rr >= h or cc >= w or not valid[rr, cc]:
                    continue
                seen += 1
                same += int(labels[rr, cc] == labels[row, col])
        if seen:
            out[index[row, col]] = same / seen
        if flat_labels[index[row, col]] != labels[row, col]:
            raise ValueError("label order drifted from the valid-cell order")
    return out


def _load_poi(city: str, processed: Path) -> np.ndarray:
    path = processed / city / "upstream" / f"{CITY_CODE[city]}_poi.npy"
    poi = np.load(path)
    if poi.shape[-1] != 14:
        raise ValueError(f"{city} POI {poi.shape}")
    return np.asarray(poi, dtype=np.float64)


def _load_roads(city: str, processed: Path) -> np.ndarray:
    path = processed / city / "upstream" / f"{CITY_CODE[city]}_roads.npy"
    roads = np.load(path)
    return np.asarray(roads, dtype=np.float64)


def _road_degree(roads: np.ndarray) -> np.ndarray:
    h, w, h2, w2 = roads.shape
    if (h, w) != (h2, w2):
        raise ValueError(f"roads {roads.shape} are not a region-to-region matrix")
    flat = roads.reshape(h * w, h * w)
    eye = np.eye(h * w)
    diagonal = float((flat * eye).sum())
    degree = flat.sum(axis=1) - np.diag(flat)
    return degree.reshape(h, w), diagonal


def _cache_ae(name: str, split: str) -> tuple[np.ndarray, float]:
    path = ROOT / "logs" / "bin_route_cache" / f"{name}_{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    blob = np.load(path)
    return np.asarray(blob["ae"], dtype=np.float64), float(blob["hours"])


def _city_mae(ae: np.ndarray, hours: float) -> float:
    points = hours * ae.shape[0]
    if points <= 0:
        raise ValueError("no evaluation points")
    return float(ae.sum()) / points


def _group_rows(
    labels: np.ndarray,
    daily: np.ndarray,
    three: np.ndarray,
    ae: dict[str, np.ndarray],
    hours: float,
) -> list[dict[str, Any]]:
    rows = []
    total_trips = float(daily.sum())
    total_ae = float(ae["finetune"].sum())
    for name in ("商业就业", "居住", "混合", "无兴趣点"):
        mask = labels == name
        row: dict[str, Any] = {
            "label": name,
            "n": int(mask.sum()),
            "trip_share": float(daily[mask].sum() / total_trips) if total_trips else None,
            "ae_share": float(ae["finetune"][mask].sum() / total_ae) if total_ae else None,
            "three_median": float(np.median(three[mask])) if mask.any() else None,
            "n_gate": int((mask & (three >= GATE)).sum()),
        }
        for method, values in ae.items():
            chosen = values[mask]
            row[f"mae_{method}"] = (
                float(chosen.sum()) / (hours * int(mask.sum())) if mask.any() else None
            )
        rows.append(row)
    return rows


def build_report(processed: Path) -> dict[str, Any]:
    prepared = {}
    for city in ("ny", "chi", "dc"):
        flow = load_city_array(city, processed)
        valid = valid_region_mask(flow)
        prepared[city] = volume._prepare(flow, valid)
        prepared[city]["valid"] = valid
        prepared[city]["poi"] = _load_poi(city, processed)
        roads = _load_roads(city, processed)
        degree, diagonal = _road_degree(roads)
        prepared[city]["degree"] = degree
        prepared[city]["road_diagonal"] = diagonal

    dc = prepared["dc"]
    labels = label_grid(dc["poi"], dc["valid"])
    flat_labels = labels[dc["valid"]]
    poi_valid = dc["poi"][dc["valid"]]
    totals = poi_valid.sum(axis=1)
    dominant = np.divide(
        poi_valid.max(axis=1),
        totals,
        out=np.full(totals.shape, np.nan),
        where=totals > 0,
    )
    neighbor = neighbor_same_rate(labels, dc["valid"])
    three = dc["three_sum"].sum(axis=1)
    daily = dc["daily"]

    drop = load_city_array("dc", processed).copy()
    drop_path = processed / "dc" / "upstream" / "TaxiDC_dropoff.npy"
    dropoff = np.load(drop_path)
    if dropoff.shape != drop.shape:
        raise ValueError(f"dropoff {dropoff.shape} != pickup {drop.shape}")
    _, weekday = hourly_calendar(dropoff.shape[0])
    start, end = 5832, 5904
    morning = np.array([hour % 24 in (7, 8, 9, 10) for hour in range(dropoff.shape[0])])

    def _ratio(pickup: np.ndarray, drop_arr: np.ndarray, select: np.ndarray) -> np.ndarray:
        num = pickup[select][:, dc["valid"]].sum(axis=0)
        den = num + drop_arr[select][:, dc["valid"]].sum(axis=0)
        out = np.full(num.shape, np.nan)
        ok = den >= GATE
        out[ok] = num[ok] / den[ok]
        return out

    long_select = weekday.copy()
    long_select[TRAIN_END:] = False
    long_select &= morning
    short_select = np.zeros(dropoff.shape[0], dtype=bool)
    short_select[start:end] = True
    short_select &= morning
    long_ratio = _ratio(load_city_array("dc", processed), dropoff, long_select & (np.arange(dropoff.shape[0]) < TRAIN_END))
    short_ratio = _ratio(load_city_array("dc", processed), dropoff, short_select)

    both_ratio = np.isfinite(long_ratio) & np.isfinite(short_ratio)
    ratio_gap = np.full(long_ratio.shape, np.nan)
    ratio_gap[both_ratio] = np.abs(short_ratio[both_ratio] - long_ratio[both_ratio])

    ae = {}
    hours = None
    for name, key in (("scratch", "scratch"), ("finetune", "chi_finetune"), ("kd", "chi_kd")):
        values, this_hours = _cache_ae(key, "test")
        if values.shape != three.shape:
            raise ValueError(f"{key} cells {values.shape} != {three.shape}")
        if hours is None:
            hours = this_hours
        elif this_hours != hours:
            raise ValueError(f"{key} hours {this_hours} != {hours}")
        ae[name] = values
    published = _city_mae(ae["finetune"], hours)
    if abs(published - 1.4773) > 5e-4:
        raise ValueError(f"cached Chicago fine-tune test MAE {published} != 1.4773")

    ny_ae, _ = _cache_ae("ny_finetune", "test")
    if abs(_city_mae(ny_ae, hours) - 1.7153) > 5e-4:
        raise ValueError("cached New York fine-tune test MAE drifted")

    from src.utils.pearson_weekday_profile import assign_similarity_bin, max_source_match

    bins = {}
    for source in ("ny", "chi"):
        rho, _, _, _ = max_source_match(prepared[source]["long"], dc["short"])
        bins[source] = assign_similarity_bin(rho)

    def stitch(choice: np.ndarray) -> np.ndarray:
        out = np.empty_like(ae["finetune"])
        out[choice == "finetune"] = ae["finetune"][choice == "finetune"]
        out[choice == "kd"] = ae["kd"][choice == "kd"]
        out[choice == "scratch"] = ae["scratch"][choice == "scratch"]
        return out

    def bin_choice(source: str, mask: np.ndarray) -> np.ndarray:
        choice = np.full(mask.shape, "finetune", dtype=object)
        use = mask & (bins[source] == 1)
        choice[use] = "kd"
        use = mask & ((bins[source] == 0) | (bins[source] < 0))
        choice[use] = "scratch"
        return choice

    gate = three >= GATE
    commercial = flat_labels == "商业就业"
    rules = {
        "一律芝加哥微调": np.full(gate.shape, "finetune", dtype=object),
        "三天不少于 150 才按芝加哥分档": bin_choice("chi", gate),
        "只在商业就业格按芝加哥分档": bin_choice("chi", commercial),
        "商业就业且三天不少于 150 才按芝加哥分档": bin_choice("chi", commercial & gate),
        "三天不少于 150 才按纽约分档": bin_choice("ny", gate),
    }
    # New York models are not in ``ae``. Keep Chicago rules for the stitched MAE.
    rules.pop("三天不少于 150 才按纽约分档")

    methods = []
    base = _city_mae(ae["finetune"], hours)
    for name, choice in rules.items():
        mae = _city_mae(stitch(choice), hours)
        methods.append(
            {
                "name": name,
                "mae": mae,
                "delta_vs_finetune": mae - base,
                "n_not_finetune": int((choice != "finetune").sum()),
            }
        )
    oracle = np.minimum(np.minimum(ae["finetune"], ae["kd"]), ae["scratch"])
    oracle_mae = _city_mae(oracle, hours)
    outside = ~gate
    oracle_outside = ae["finetune"].copy()
    oracle_outside[outside] = oracle[outside]
    methods.append(
        {
            "name": "每个格子事后取三者最小，上界",
            "mae": oracle_mae,
            "delta_vs_finetune": oracle_mae - base,
            "n_not_finetune": int((oracle < ae["finetune"] - 1e-8).sum()),
        }
    )
    methods.append(
        {
            "name": "只在三天少于 150 的格子上取事后最小，上界",
            "mae": _city_mae(oracle_outside, hours),
            "delta_vs_finetune": _city_mae(oracle_outside, hours) - base,
            "n_not_finetune": int((outside & (oracle < ae["finetune"] - 1e-8)).sum()),
        }
    )

    degree = dc["degree"][dc["valid"]]
    summary = {
        "n_valid": int(dc["valid"].sum()),
        "n_zero_poi": int((flat_labels == "无兴趣点").sum()),
        "dominant_share_median": float(np.nanmedian(dominant)),
        "dominant_share_p25": float(np.nanquantile(dominant, 0.25)),
        "frac_dominant_below_0.4": float(np.nanmean(dominant < 0.4)),
        "residential_poi_total": float(dc["poi"][:, :, RESIDENTIAL].sum()),
        "residential_share_ge_0.3": int((poi_valid[:, RESIDENTIAL] / np.maximum(totals, 1) >= 0.3).sum()),
        "neighbor_same_median": float(np.nanmedian(neighbor)),
        "road_positive": int((degree > 0).sum()),
        "road_diagonal": dc["road_diagonal"],
        "ratio_n": int(both_ratio.sum()),
        "ratio_gap_median": float(np.nanmedian(ratio_gap)),
        "gate_n": int(gate.sum()),
        "gate_and_commercial": int((gate & commercial).sum()),
        "gate_not_commercial": int((gate & ~commercial).sum()),
        "commercial_not_gate": int((commercial & ~gate).sum()),
        "finetune_mae": base,
    }
    return {
        "trained": False,
        "seed": 42,
        "gate": GATE,
        "commercial_classes": [POI_NAMES[i] for i in COMMERCIAL],
        "summary": summary,
        "groups": _group_rows(flat_labels, daily, three, ae, hours),
        "methods": methods,
        "reading": _reading(summary, methods),
    }


TRAIN_END = 5904


def _reading(summary: dict[str, Any], methods: dict[str, Any] | list[dict[str, Any]]) -> list[str]:
    by_name = {item["name"]: item for item in methods}
    gate = by_name["三天不少于 150 才按芝加哥分档"]
    both = by_name["商业就业且三天不少于 150 才按芝加哥分档"]
    bound = by_name["只在三天少于 150 的格子上取事后最小，上界"]
    return [
        (
            f"华盛顿有效格 {summary['n_valid']} 个，其中 {summary['n_zero_poi']} 个没有兴趣点。"
            f"有兴趣点的格子里，最大一类占比的中位数是 {summary['dominant_share_median']:.2f}，"
            f"低于 0.4 的占 {summary['frac_dominant_below_0.4'] * 100:.1f}%。"
            f"和周围有效格标签相同的比例，中位数是 {summary['neighbor_same_median']:.2f}。"
        ),
        (
            f"三天上车不少于 150 的有 {summary['gate_n']} 个，其中商业就业 {summary['gate_and_commercial']} 个，"
            f"不是商业就业的 {summary['gate_not_commercial']} 个。"
            f"商业就业但没过这道门的有 {summary['commercial_not_gate']} 个。"
        ),
        (
            f"种子 42 的芝加哥微调测试 MAE 是 {summary['finetune_mae']:.4f}。"
            f"只在过门格子上按档拼接，MAE 变为 {gate['mae']:.4f}，差 {gate['delta_vs_finetune']:+.4f}。"
            f"再要求商业就业，MAE 变为 {both['mae']:.4f}，差 {both['delta_vs_finetune']:+.4f}。"
            f"门下格子即使事后每格挑误差最小的模型，全城 MAE 也只再低 {abs(bound['delta_vs_finetune']):.4f}。"
        ),
    ]


def _fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}{suffix}"


def render_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# 兴趣点能不能把每个格子分成功能区",
        "",
        "兴趣点类别和顺序来自 CrossTReS 附录 C：景点、医疗、生活服务、居住、金融、体育休闲、文化教育、购物、住宿、政府机构、公司、餐饮、交通、公共服务。",
        "商业就业是金融、购物、住宿、公司、餐饮这五类合计占比至少 0.5，并且比居住类至少高 0.15。居住类单独占比至少 0.5，并且比这五类至少高 0.15。其余有兴趣点的格子记为混合。",
        "这个切分在看误差之前定下。兴趣点是静态计数。三天门槛仍是 2016-08-31 至 09-02 的出租上车合计不少于 150。",
        "误差来自种子 42 已缓存的逐格绝对误差，芝加哥微调的全城测试 MAE 对不上 1.4773 就会停。没有新训练。",
        "",
        "## 标签盖不住每一个格子",
        "",
        f"- 有效格 {summary['n_valid']}，无兴趣点 {summary['n_zero_poi']}。",
        f"- 有兴趣点时，最大一类占比的中位数 {summary['dominant_share_median']:.2f}，25 分位 {summary['dominant_share_p25']:.2f}，低于 0.4 的比例 {summary['frac_dominant_below_0.4'] * 100:.1f}%。",
        f"- 居住类兴趣点全城合计 {summary['residential_poi_total']:.0f} 个。有效格里居住类占比达到 0.3 的有 {summary['residential_share_ge_0.3']} 个。",
        f"- 八邻域里有效邻居与自己同标签的比例，中位数 {summary['neighbor_same_median']:.2f}。",
        f"- 路网矩阵只记高速公路连接。对角元合计 {summary['road_diagonal']:.0f}。有效格里度大于 0 的有 {summary['road_positive']} 个。",
        f"- 早上 7–10 点上车占上下车之和的比例，长窗口和这三天都能算的格子有 {summary['ratio_n']} 个，绝对差的中位数 {summary['ratio_gap_median']:.2f}。",
        f"- 三天不少于 150 的格子 {summary['gate_n']} 个；其中商业就业 {summary['gate_and_commercial']}，其他 {summary['gate_not_commercial']}。商业就业但没过门的 {summary['commercial_not_gate']} 个。",
        "",
        "## 和上车量、测试误差的关系",
        "",
        "组内 MAE 是该组格子的绝对误差和，除以该组的小时数乘格子数。误差占比用芝加哥微调。",
        "",
        "| 标签 | 格子 | 占上车量 | 占微调绝对误差 | 三天总上车中位数 | 过 150 门 | 微调 MAE | 蒸馏 MAE | 从零 MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["groups"]:
        lines.append(
            f"| {row['label']} | {row['n']} | {_fmt(None if row['trip_share'] is None else row['trip_share'] * 100, 1, '%')} | "
            f"{_fmt(None if row['ae_share'] is None else row['ae_share'] * 100, 1, '%')} | "
            f"{_fmt(row['three_median'], 0)} | {row['n_gate']} | {_fmt(row['mae_finetune'], 3)} | "
            f"{_fmt(row['mae_kd'], 3)} | {_fmt(row['mae_scratch'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## 用功能区改分流之后的全城测试 MAE",
            "",
            "按档拼接沿用高于 0.8 微调、0.5 到 0.8 蒸馏、其余从零。没被规则点到的格子保持芝加哥微调。",
            "事后最小是每个格子在三个已有模型里挑绝对误差最小的那个，只能当作上界。",
            "",
            "| 规则 | 测试 MAE | 相对一律微调 | 改派的格子 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["methods"]:
        lines.append(
            f"| {row['name']} | {row['mae']:.4f} | {row['delta_vs_finetune']:+.4f} | {row['n_not_finetune']} |"
        )
    lines.extend(["", "## 读法", ""])
    for paragraph in report["reading"]:
        lines.append(paragraph)
        lines.append("")
    lines.extend(["重算：", "", "```bash", "python scripts/15_function_cells.py", "```", ""])
    return "\n".join(lines)


def main() -> int:
    report = build_report(ROOT / "data" / "processed")
    out_json = ROOT / "docs" / "profiles" / "function_cells.json"
    out_md = ROOT / "docs" / "profiles" / "function_cells.md"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(render_md(report), encoding="utf-8")
    print(
        f"zero poi {report['summary']['n_zero_poi']}; "
        f"gate commercial {report['summary']['gate_and_commercial']}/"
        f"{report['summary']['gate_n']}; "
        f"finetune {report['summary']['finetune_mae']:.4f}"
    )
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
