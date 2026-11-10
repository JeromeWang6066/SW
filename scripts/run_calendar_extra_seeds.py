"""Re-run the Washington calendar models at extra seeds.

Source teachers stay at seed 42, both the old checkpoints and the new weekday
and weekend teachers. ``--seed`` puts the new Washington weights and logs beside
the seed-42 files, with a ``_seedN`` suffix. Seeds 43 and 44 are the default.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

JOBS = [
    ("src.train.train_teacher", "configs/calendar/dc_scratch.yaml", "cal-dc-scratch"),
    ("src.train.train_finetune", "configs/calendar/finetune_ny_wd3we2.yaml", "cal-finetune-ny"),
    ("src.train.train_finetune", "configs/calendar/finetune_chi_wd3we2.yaml", "cal-finetune-chi"),
    ("src.train.train_student_kd", "configs/calendar/kd_ny_wd3we2.yaml", "cal-kd-ny"),
    ("src.train.train_student_kd", "configs/calendar/kd_chi_wd3we2.yaml", "cal-kd-chi"),
    ("src.train.train_finetune", "configs/calendar/finetune_ny_weekday.yaml", "cal-finetune-ny-weekday"),
    ("src.train.train_finetune", "configs/calendar/finetune_ny_weekend.yaml", "cal-finetune-ny-weekend"),
    ("src.train.train_finetune", "configs/calendar/finetune_chi_weekday.yaml", "cal-finetune-chi-weekday"),
    ("src.train.train_finetune", "configs/calendar/finetune_chi_weekend.yaml", "cal-finetune-chi-weekend"),
]


def _complete(run_id: str) -> bool:
    path = _ROOT / "logs" / "calendar" / f"{run_id}.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("status") == "complete" and payload.get("test_slices")


def main(argv: list[str] | None = None) -> int:
    seeds = [int(item) for item in (argv if argv is not None else sys.argv[1:])]
    if not seeds:
        seeds = [43, 44]
    for seed in seeds:
        for module, config, run_id in JOBS:
            tagged = f"{run_id}_seed{seed}"
            if _complete(tagged):
                print(f"skip {tagged}")
                continue
            print(f"start {tagged}")
            completed = subprocess.run(
                [sys.executable, "-u", "-m", module, "--config", config, "--seed", str(seed), "--resume"],
                cwd=_ROOT,
            )
            if completed.returncode != 0:
                print(f"{tagged} failed with code {completed.returncode}")
                return completed.returncode
        if _complete(f"cal-stitch-ny_seed{seed}") and _complete(f"cal-stitch-chi_seed{seed}"):
            print(f"skip stitch seed {seed}")
            continue
        print(f"start stitch seed {seed}")
        completed = subprocess.run(
            [sys.executable, "-u", "scripts/08_calendar_stitch.py", "--seed", str(seed)],
            cwd=_ROOT,
        )
        if completed.returncode != 0:
            print(f"stitch seed {seed} failed with code {completed.returncode}")
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
