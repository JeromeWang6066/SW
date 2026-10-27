"""CLI stub: few-shot student with frozen-teacher KD.

Group 2: standard KD. Group 3 Pearson multiplier remains a config hook.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger("sw.train_student_kd")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train KD student under STBaT 3-day protocol (NOT IMPLEMENTED)"
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/student_kd_ny2dc.yaml",
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

    kd = cfg.get("kd", {})
    splits = cfg.get("splits", {})
    exp = cfg.get("experiment", {})
    cities = cfg.get("cities", {})

    logger.info("Loaded config: %s", args.config)
    logger.info("experiment=%s cities=%s seed=%s", exp, cities, seed)
    logger.info(
        "KD: enabled=%s lambda_kd=%s freeze_teacher=%s pearson_multiplier=%s",
        kd.get("enabled"),
        kd.get("lambda_kd"),
        kd.get("freeze_teacher", True),
        kd.get("pearson_multiplier", False),
    )
    logger.info("few_shot_days=%s", splits.get("few_shot_days", 3))

    if kd.get("pearson_multiplier"):
        logger.warning(
            "pearson_multiplier=true but Group 3 formula is TBD — "
            "see src/utils/pearson_weekday_profile.py"
        )

    if args.dry_run:
        logger.info("Dry-run OK — config load + seed only. No training.")
        return 0

    logger.error("Student KD training is NOT IMPLEMENTED (scaffold only).")
    logger.error("Next steps:")
    logger.error("  1) Finish Group 1 teacher checkpoint")
    logger.error("  2) Implement frozen-teacher KD loop (Group 2)")
    logger.error("  3) Keep pearson_multiplier=false until Group 3 design is fixed")
    logger.error("  4) Follow docs/REPRODUCTION.md Group 2")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
