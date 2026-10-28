"""Run G1-C fine-tunes in order. Skip a run that already wrote logs/<id>.json."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    "configs/finetune_ny2dc.yaml",
    "configs/finetune_chi2dc.yaml",
)


def _done(config_path: Path) -> bool:
    run_id = None
    for line in config_path.read_text(encoding="utf-8").splitlines():
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
    for name in RUNS:
        config = ROOT / name
        if _done(config):
            print(f"skip complete {name}", flush=True)
            continue
        print(f"start {name}", flush=True)
        proc = subprocess.run(
            [sys.executable, "-u", "-m", "src.train.train_finetune", "--config", name, "--resume"],
            cwd=ROOT,
        )
        if proc.returncode != 0:
            print(f"failed {name} exit={proc.returncode}", flush=True)
            return proc.returncode
        if not _done(config):
            print(f"finished process but {name} has no result json", flush=True)
            return 2
    print("G1-C complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
