#!/usr/bin/env python3
"""Check Python / core package versions. Does not train or download data."""

from __future__ import annotations

import platform
import sys


def main() -> int:
    print("=== SW env check ===")
    print(f"python: {sys.version.split()[0]} ({platform.platform()})")

    ok = True
    for name in ("numpy", "pandas", "scipy", "yaml", "tqdm"):
        try:
            mod = __import__(name if name != "yaml" else "yaml")
            ver = getattr(mod, "__version__", "?")
            print(f"  {name}: {ver}")
        except ImportError as e:
            print(f"  {name}: MISSING ({e})")
            ok = False

    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"  torch: {torch.__version__}  cuda_available={cuda}")
        if cuda:
            print(f"  cuda_device0: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"  torch: MISSING ({e}) — OK for docs-only; needed before training")
        ok = False

    if ok:
        print("Env check passed (core imports OK).")
        return 0
    print("Env check incomplete — install requirements.txt before training.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
