"""CLI stub: train source (or target-only) ST-Net teacher.

Honest behavior: load config, set seed, print next steps, exit non-zero.
Does not train and does not invent metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python -m src.train.train_teacher`` from repo root
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger("sw.train_teacher")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ST-Net teacher (NOT IMPLEMENTED)")
    p.add_argument(
        "--config",
        type=str,
        default="configs/teacher_ny.yaml",
        help="Path to YAML config",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load config + set seed, then exit 0",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    exp = cfg.get("experiment", {})
    cities = cfg.get("cities", {})
    logger.info("Loaded config: %s", args.config)
    logger.info("experiment=%s cities=%s seed=%s", exp, cities, seed)
    logger.info(
        "model placeholders: %s",
        cfg.get("model", {}),
    )

    if args.dry_run:
        logger.info("Dry-run OK — config load + seed only. No training.")
        return 0

    logger.error("Training is NOT IMPLEMENTED (scaffold only).")
    logger.error("Next steps:")
    logger.error("  1) Download zip per docs/DATA.md (do not commit); scripts/01_verify_data_layout.py")
    logger.error("  2) Port ST-Net from KL4805/CrossTReS src/model.py (docs/UPSTREAM_MAP.md)")
    logger.error("  3) Implement the train loop in src/train/train_teacher.py")
    logger.error("  4) Follow docs/REPRODUCTION.md Day 4+ Group 1")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
