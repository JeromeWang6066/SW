"""Run the seed-42 calendar trainings, then stitch the split models.

Skips a run whose log json already says complete. Does not touch checkpoints/
or logs/ outside checkpoints/calendar and logs/calendar.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

JOBS = [
    ("src.train.train_teacher", "configs/calendar/teacher_ny_weekday.yaml", "cal-teacher-ny-weekday"),
    ("src.train.train_teacher", "configs/calendar/teacher_ny_weekend.yaml", "cal-teacher-ny-weekend"),
    ("src.train.train_teacher", "configs/calendar/teacher_chi_weekday.yaml", "cal-teacher-chi-weekday"),
    ("src.train.train_teacher", "configs/calendar/teacher_chi_weekend.yaml", "cal-teacher-chi-weekend"),
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


def main() -> int:
    for module, config, run_id in JOBS:
        if _complete(run_id):
            print(f"skip {run_id}")
            continue
        print(f"start {run_id}")
        completed = subprocess.run(
            [sys.executable, "-u", "-m", module, "--config", config, "--resume"],
            cwd=_ROOT,
        )
        if completed.returncode != 0:
            print(f"{run_id} failed with code {completed.returncode}")
            return completed.returncode
    print("start stitch")
    completed = subprocess.run([sys.executable, "scripts/08_calendar_stitch.py"], cwd=_ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
