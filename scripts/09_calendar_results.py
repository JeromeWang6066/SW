"""Write docs/CALENDAR_RESULTS.md from logs/calendar/*.json.

Old seed-42 numbers are copied from docs/RESULTS.md. They are a different
protocol: three contiguous weekdays, not three weekdays plus two weekend days.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Seed 42 test metrics from docs/RESULTS.md. Protocol: last 72 training labels.
OLD_TEST = {
    "华盛顿从零，连续 3 个工作日": (1.8701, 5.5528, 75.9919),
    "纽约微调，连续 3 个工作日": (1.7153, 5.2786, 72.3947),
    "芝加哥微调，连续 3 个工作日": (1.4773, 5.0578, 73.3157),
    "纽约蒸馏，随机初始化，连续 3 个工作日": (1.7519, 5.5064, 59.2878),
    "芝加哥蒸馏，随机初始化，连续 3 个工作日": (1.7485, 5.4130, 65.3868),
}

RUNS = [
    ("cal-dc-scratch", "华盛顿从零，3 个工作日加 2 个周末日"),
    ("cal-finetune-ny", "纽约旧教师微调，3 个工作日加 2 个周末日"),
    ("cal-finetune-chi", "芝加哥旧教师微调，3 个工作日加 2 个周末日"),
    ("cal-kd-ny", "纽约旧教师蒸馏，随机初始化，3 个工作日加 2 个周末日"),
    ("cal-kd-chi", "芝加哥旧教师蒸馏，随机初始化，3 个工作日加 2 个周末日"),
    ("cal-stitch-ny", "纽约分开微调后按日历拼接"),
    ("cal-stitch-chi", "芝加哥分开微调后按日历拼接"),
]


def _load(run_id: str) -> dict:
    path = _ROOT / "logs" / "calendar" / f"{run_id}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"{run_id} is not complete")
    return payload


def _fmt(row: dict) -> str:
    return f"{row['mae']:.4f} | {row['rmse']:.4f} | {row['mape']:.4f}"


def _table(rows: list[tuple[str, str]]) -> str:
    lines = [
        "| 设定 | 范围 | MAE | RMSE | MAPE |",
        "|---|---|---:|---:|---:|",
    ]
    lines.extend(f"| {name} | {rest}" for name, rest in rows)
    return "\n".join(lines)


def _seed_run_id(base: str, seed: int) -> str:
    return base if seed == 42 else f"{base}_seed{seed}"


def _cell(payload: dict, kind: str) -> str:
    row = payload["test_slices"][kind]
    return f"{row['mae']:.4f} / {row['rmse']:.4f}"


def _append_extra_seeds(parts: list[str]) -> None:
    """Seeds 43 and 44 rerun Washington only. Skip the section until both exist."""
    bases = [
        ("cal-dc-scratch", "华盛顿从零"),
        ("cal-finetune-ny", "纽约一个模型"),
        ("cal-finetune-chi", "芝加哥一个模型"),
        ("cal-kd-ny", "纽约随机初始化蒸馏"),
        ("cal-kd-chi", "芝加哥随机初始化蒸馏"),
        ("cal-stitch-ny", "纽约分开拼接"),
        ("cal-stitch-chi", "芝加哥分开拼接"),
    ]
    seeds = (42, 43, 44)
    ids = [_seed_run_id(base, seed) for base, _ in bases for seed in seeds]
    if any(not (_ROOT / "logs" / "calendar" / f"{run_id}.json").is_file() for run_id in ids):
        return
    loaded = {run_id: _load(run_id) for run_id in ids}
    parts.extend(
        [
            "## 种子 43 和 44",
            "",
            "这两次只重跑华盛顿。旧教师和这一轮的工作日、周末教师都仍是种子 42 的权重。从零训练和蒸馏的学生换了随机初始化。微调从同一份教师权重出发，种子改变的是样本顺序。验证集和测试集没有换。",
            "",
            "下表是测试集全部小时，格子里是 MAE / RMSE。均值是三个种子的算术平均。",
            "",
            "| 设定 | 种子 42 | 种子 43 | 种子 44 | 均值 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for base, label in bases:
        rows = [loaded[_seed_run_id(base, seed)]["test_slices"]["all"] for seed in seeds]
        mean_mae = sum(row["mae"] for row in rows) / 3
        mean_rmse = sum(row["rmse"] for row in rows) / 3
        parts.append(
            "| "
            + " | ".join(
                [label, *(_cell(loaded[_seed_run_id(base, seed)], "all") for seed in seeds)]
            )
            + f" | {mean_mae:.4f} / {mean_rmse:.4f} |"
        )
    parts.extend(
        [
            "",
            "周末加节假日的测试 MAE。分开拼接低于同一个 120 小时的一个模型，才算分开训练有用。",
            "",
            "| 对照 | 种子 42 | 种子 43 | 种子 44 |",
            "|---|---:|---:|---:|",
        ]
    )
    notes: list[str] = []
    for source, mixed_base, stitch_base, label in (
        ("ny", "cal-finetune-ny", "cal-stitch-ny", "纽约"),
        ("chi", "cal-finetune-chi", "cal-stitch-chi", "芝加哥"),
    ):
        mixed_text = []
        stitch_text = []
        verdicts = []
        for seed in seeds:
            mixed = loaded[_seed_run_id(mixed_base, seed)]["test_slices"]["weekend"]["mae"]
            stitch = loaded[_seed_run_id(stitch_base, seed)]["test_slices"]["weekend"]["mae"]
            mixed_text.append(f"{mixed:.4f}")
            stitch_text.append(f"{stitch:.4f}")
            if stitch < mixed:
                verdicts.append("低于")
            elif stitch > mixed:
                verdicts.append("高于")
            else:
                verdicts.append("等于")
        parts.append(f"| {label}一个模型 | {' | '.join(mixed_text)} |")
        parts.append(f"| {label}分开拼接 | {' | '.join(stitch_text)} |")
        notes.append(
            f"{label}三个种子上，分开拼接的周末 MAE 依次{'、'.join(verdicts)}一个模型。"
            + ("这个方向三个种子相同。" if len(set(verdicts)) == 1 else "这个方向随种子改变。")
        )
        rmse_signs = []
        for seed in seeds:
            mixed_rmse = loaded[_seed_run_id(mixed_base, seed)]["test_slices"]["all"]["rmse"]
            stitch_rmse = loaded[_seed_run_id(stitch_base, seed)]["test_slices"]["all"]["rmse"]
            if stitch_rmse < mixed_rmse:
                rmse_signs.append("低于")
            elif stitch_rmse > mixed_rmse:
                rmse_signs.append("高于")
            else:
                rmse_signs.append("等于")
        notes.append(
            f"{label}全部小时的测试 RMSE，分开拼接依次{'、'.join(rmse_signs)}一个模型。"
        )
    parts.append("")
    parts.extend(notes)
    parts.append("")


def main() -> int:
    loaded = {run_id: _load(run_id) for run_id, _ in RUNS}
    teachers = [
        _load(run_id)
        for run_id in (
            "cal-teacher-ny-weekday",
            "cal-teacher-ny-weekend",
            "cal-teacher-chi-weekday",
            "cal-teacher-chi-weekend",
        )
    ]
    parts = [
        "# 工作日和周末分开的初步训练",
        "",
        "种子 42。权重在 `checkpoints/calendar/`，日志在 `logs/calendar/`。旧的 `checkpoints/` 只读。",
        "",
        "华盛顿这一轮的训练标签是 2016-08-27、2016-08-28，以及 2016-08-31 至 2016-09-02，共 120 小时。8 月 29 日和 8 月 30 日不作为标签。验证集和测试集仍是其后各 1440 小时。",
        "",
        "工作日是周一到周五，并且不是 2016 年这些美国联邦假日：1 月 1 日、1 月 18 日、2 月 15 日、5 月 30 日、7 月 4 日、9 月 5 日。周六、周日和这些假日是周末一类。劳动节在验证集，下标 5952，评价时用周末模型，不拿它训练。",
        "",
        "源城训练段标签：工作日 4104 小时，周末加节假日 1794 小时。",
        "",
        "一个模型的早停看验证集全部小时。分开的两个模型，早停只看同一类小时。拼接时按标签小时的类别选模型。全部小时这一行把两类的平方误差、绝对误差和正流量百分比误差加总，不是两个 RMSE 的平均。",
        "",
        "MAPE 只在反归一化后的真实流量大于 0 的格子上计算。",
        "",
        "## 全部小时，测试集",
        "",
        "旧实验是连续 3 个工作日。新实验是 3 个工作日加 2 个周末日。验证集和测试集的日历相同，训练标签不同，所以两行可以并列表，但不能当成同一次实验的消融。",
        "",
    ]
    test_rows = []
    for name, (mae, rmse, mape) in OLD_TEST.items():
        test_rows.append((name, f"全部小时 | {mae:.4f} | {rmse:.4f} | {mape:.4f} |"))
    for run_id, name in RUNS:
        row = loaded[run_id]["test_slices"]["all"]
        test_rows.append((name, f"全部小时 | {_fmt(row)} |"))
    parts.append(_table(test_rows))
    parts.extend(["", "## 验证集和测试集拆成三类", ""])
    for run_id, name in RUNS:
        payload = loaded[run_id]
        parts.append(f"### {name}")
        parts.append("")
        parts.append(f"最好的轮次：{payload.get('best_epoch', '拼接，无单独早停')}。")
        parts.append("")
        slice_rows = []
        for split in ("val", "test"):
            for kind, label in (("all", "全部小时"), ("weekday", "工作日"), ("weekend", "周末加节假日")):
                row = payload[f"{split}_slices"][kind]
                slice_rows.append((f"{split}", f"{label} | {_fmt(row)} |"))
        parts.append(_table(slice_rows))
        parts.append("")

    parts.extend(["## 分开训练是否有用", ""])
    for source, stitch_id, mixed_id, label in (
        ("ny", "cal-stitch-ny", "cal-finetune-ny", "纽约"),
        ("chi", "cal-stitch-chi", "cal-finetune-chi", "芝加哥"),
    ):
        stitch = loaded[stitch_id]
        mixed = loaded[mixed_id]
        parts.append(f"### {label}")
        parts.append("")
        parts.append(
            "对照是同一个 120 个标签、一个模型的微调。周末加节假日这一行更低，才说明分开训练有用。"
        )
        parts.append("")
        parts.append(
            "| 范围 | 一个模型 MAE | 一个模型 RMSE | 一个模型 MAPE | 分开拼接 MAE | 分开拼接 RMSE | 分开拼接 MAPE |"
        )
        parts.append("|---|---:|---:|---:|---:|---:|---:|")
        for split in ("val", "test"):
            for kind, kind_label in (("all", "全部小时"), ("weekend", "周末加节假日"), ("weekday", "工作日")):
                left = mixed[f"{split}_slices"][kind]
                right = stitch[f"{split}_slices"][kind]
                parts.append(
                    f"| {split} {kind_label} | {_fmt(left)} | {_fmt(right)} |"
                )
        parts.append("")
        for split in ("val", "test"):
            mixed_weekend = mixed[f"{split}_slices"]["weekend"]["mae"]
            stitch_weekend = stitch[f"{split}_slices"]["weekend"]["mae"]
            if stitch_weekend < mixed_weekend:
                verdict = "低于"
            elif stitch_weekend > mixed_weekend:
                verdict = "高于"
            else:
                verdict = "等于"
            parts.append(
                f"{split} 周末加节假日 MAE：分开拼接 {stitch_weekend:.4f}，一个模型 {mixed_weekend:.4f}，分开拼接{verdict}一个模型。"
            )
        parts.append("")

    _append_extra_seeds(parts)

    parts.extend(["## 源城教师", ""])
    parts.append("早停看验证集里同一类小时。下表仍是全部小时，以及拆开的两类。")
    parts.append("")
    for payload in teachers:
        parts.append(f"### {payload['run_id']}")
        parts.append("")
        parts.append(f"最好的轮次：{payload['best_epoch']}。训练窗口 {payload.get('train_windows')}。")
        parts.append("")
        rows = []
        for split in ("val", "test"):
            for kind, label in (("all", "全部小时"), ("weekday", "工作日"), ("weekend", "周末加节假日")):
                rows.append((split, f"{label} | {_fmt(payload[f'{split}_slices'][kind])} |"))
        parts.append(_table(rows))
        parts.append("")

    out = _ROOT / "docs" / "CALENDAR_RESULTS.md"
    out.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
