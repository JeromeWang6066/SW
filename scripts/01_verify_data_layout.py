#!/usr/bin/env python3
"""Verify expected data directories exist. Does NOT download anything."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DIRS = [
    ROOT / "data",
    ROOT / "data" / "raw",
    ROOT / "data" / "processed",
    ROOT / "data" / "processed" / "ny",
    ROOT / "data" / "processed" / "chi",
    ROOT / "data" / "processed" / "dc",
]


def main() -> int:
    print("=== SW data layout check (no download) ===")
    missing = [p for p in REQUIRED_DIRS if not p.is_dir()]
    for p in REQUIRED_DIRS:
        status = "OK" if p.is_dir() else "MISSING"
        print(f"  [{status}] {p.relative_to(ROOT)}")

    if missing:
        print("Create missing directories (empty is fine). See docs/DATA.md.")
        return 1

    # Optional: warn if processed cities look empty (only .gitkeep)
    for city in ("ny", "chi", "dc"):
        d = ROOT / "data" / "processed" / city
        files = [f for f in d.iterdir() if f.name != ".gitkeep"]
        if not files:
            print(f"  [EMPTY] data/processed/{city}/ — expected after real preprocess")
        else:
            print(f"  [HAS FILES] data/processed/{city}/ → {len(files)} entries")

    print(
        "Layout dirs OK. Download steps + verified URLs: docs/DATA.md "
        "(do not commit zip/npy)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
