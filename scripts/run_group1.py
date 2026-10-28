"""Run Group 1 in order. Skip runs that already wrote logs/<id>.json.

Re-running this script resumes an unfinished experiment from its ``.latest.pt``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    "configs/teacher_ny.yaml",
    "configs/teacher_chi.yaml",
    "configs/baseline_target_fewshot.yaml",
)


def _done(config_path: Path) -> bool:
    text = config_path.read_text(encoding="utf-8")
    run_id = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("id:"):
            run_id = stripped.split(":", 1)[1].strip()
            break
    if not run_id:
        return False
    result = ROOT / "logs" / f"{run_id}.json"
    if not result.is_file():
        return False
    payload = json.loads(result.read_text(encoding="utf-8"))
    return payload.get("status") == "complete" and payload.get("test")


def main() -> int:
    py = sys.executable
    for name in RUNS:
        config = ROOT / name
        if _done(config):
            print(f"skip complete {name}", flush=True)
            continue
        print(f"start {name}", flush=True)
        proc = subprocess.run(
            [py, "-u", "-m", "src.train.train_teacher", "--config", name, "--resume"],
            cwd=ROOT,
        )
        if proc.returncode != 0:
            print(f"failed {name} exit={proc.returncode}", flush=True)
            return proc.returncode
        if not _done(config):
            print(f"finished process but {name} has no result json", flush=True)
            return 2
    print("Group 1 complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
