#!/usr/bin/env python3
"""CLI for the val-selection protocol.

inventory     stat/hash only; never unpacks test predictions
lock-selections   train/val only; writes a scorable lock iff complete
score-test    verifies lock + rematerializes train/val + plan/impl/config, then
              scores test into an exclusive run directory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.val_select.bundle_adapter import adapt_pr20_bundle
from src.val_select.io_cache import inventory_caches, write_json
from src.val_select.score_test import score_test
from src.val_select.select import lock_all_selections
from src.val_select.snapshot import binding_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validation-based strength selection protocol")
    sub = parser.add_subparsers(dest="command", required=True)

    inv = sub.add_parser("inventory", help="stat/hash caches without unpacking arrays")
    inv.add_argument("--cache-dir", required=True)
    inv.add_argument("--out", required=False)

    lock = sub.add_parser("lock-selections", help="complete all non-test selections and hash the lock")
    lock.add_argument("--cache-dir", required=True)
    lock.add_argument("--lock-path", required=True)
    lock.add_argument("--allow-synthetic", action="store_true")
    lock.add_argument("--allow-documented-from-bundle", action="store_true")
    lock.add_argument("--seed-freeze-id", default=None)

    score = sub.add_parser("score-test", help="score test after verifying the selection lock")
    score.add_argument("--cache-dir", required=True)
    score.add_argument("--lock-path", required=True)
    score.add_argument("--report-path", required=False)
    score.add_argument("--run-dir", required=False)
    score.add_argument("--allow-synthetic", action="store_true")
    score.add_argument("--allow-documented-from-bundle", action="store_true")
    score.add_argument(
        "--i-understand-this-is-exploratory",
        action="store_true",
        help="Required acknowledgement that this test interval has already been seen.",
    )
    adapt = sub.add_parser(
        "adapt-bundle",
        help="Stage documented_from_bundle sidecars from a PR20 hourly zip (train/val only; never opens test arrays)",
    )
    adapt.add_argument("--zip", required=True)
    adapt.add_argument("--extracted-root", required=True)
    adapt.add_argument("--output-dir", required=True)
    adapt.add_argument("--allow-documented-from-bundle", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inventory":
        report = inventory_caches(Path(args.cache_dir))
        report["binding"] = binding_payload()
        if args.out:
            write_json(Path(args.out), report)
        else:
            print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "lock-selections":
        lock = lock_all_selections(
            Path(args.cache_dir),
            Path(args.lock_path),
            allow_synthetic=args.allow_synthetic,
            allow_documented_from_bundle=args.allow_documented_from_bundle,
            seed_freeze_id=args.seed_freeze_id,
        )
        print(json.dumps({"scorable": lock["scorable"], "lock_hash": lock.get("lock_hash"), "missing": lock["missing"], "blocked": lock["blocked"]}, indent=2))
        return 0 if lock["scorable"] else 2
    if args.command == "score-test":
        if not args.i_understand_this_is_exploratory:
            raise SystemExit("refusing to score test without --i-understand-this-is-exploratory")
        command = " ".join(["scripts/20_val_select_kappa.py", *([str(a) for a in (argv or sys.argv[1:])])])
        report = score_test(
            Path(args.cache_dir),
            Path(args.lock_path),
            args.report_path,
            allow_synthetic=args.allow_synthetic,
            allow_documented_from_bundle=args.allow_documented_from_bundle,
            run_dir=args.run_dir,
            command=command,
            i_understand_this_is_exploratory=True,
        )
        print(json.dumps({"lock_hash": report["lock_hash"], "run_dir": report["run_dir"], "report_written": True}, indent=2))
        return 0
    if args.command == "adapt-bundle":
        if not args.allow_documented_from_bundle:
            raise SystemExit("refusing to adapt a bundle without --allow-documented-from-bundle")
        evidence = adapt_pr20_bundle(
            zip_path=Path(args.zip),
            extracted_root=Path(args.extracted_root),
            output_dir=Path(args.output_dir),
            allow_documented_from_bundle=True,
            open_test_arrays=False,
        )
        print(json.dumps({
            "input_protocol_id": evidence["input_protocol_id"],
            "zip_sha256": evidence["zip_sha256"],
            "evidence_hash": evidence["evidence_hash"],
            "cache_dir": evidence["cache_dir"],
            "test_arrays_opened": False,
        }, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
