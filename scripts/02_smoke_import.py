#!/usr/bin/env python3
"""Smoke-test imports, config load, seed, metrics, and Pearson helper on fake tensors."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    print("=== SW smoke import ===")

    from src import __version__
    from src.eval.metrics import mae, rmse, summarize
    from src.utils.config import load_config
    from src.utils.pearson_weekday_profile import (
        compute_weekday_pearson_mask,
        weekday_24h_profile,
    )
    from src.utils.seed import set_seed
    from src.data.splits import few_shot_split_indices

    print(f"  src.__version__ = {__version__}")
    set_seed(0)

    cfg = load_config(ROOT / "configs" / "student_kd_ny2dc.yaml")
    assert cfg.get("kd", {}).get("enabled") is True
    assert cfg.get("splits", {}).get("few_shot_days") == 3
    assert cfg.get("kd", {}).get("pearson_multiplier") is False
    print("  config merge OK:", cfg.get("experiment", {}).get("id"))

    # Fake equal-N cities for shape/profile smoke only — not real data / not metrics report
    rng = np.random.default_rng(0)
    fake = rng.normal(size=(24 * 10, 4, 4))
    profile = weekday_24h_profile(fake)
    assert profile.shape == (16, 24)
    rho = compute_weekday_pearson_mask(fake, fake)
    assert rho.shape == (16,)
    print("  pearson profile smoke OK (synthetic equal-grid only)")

    # Metrics on synthetic arrays — verifying formula wiring, not experiment results
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 4.0])
    s = summarize(a, b)
    assert abs(s["mae"] - mae(a, b)) < 1e-9
    assert s["rmse"] == rmse(a, b)
    print("  metrics smoke OK:", s)

    # Long enough for approximate 2+2 month + 3 day protocol under 30*24 month hours
    T = (2 + 2) * 30 * 24 + 3 * 24
    splits = few_shot_split_indices(T, few_shot_days=3)
    assert splits.train.stop - splits.train.start == 3 * 24
    print("  few-shot split helper OK")

    # Model import (torch required)
    try:
        from src.models.st_net import STNet
        from src.models.kd import freeze_module, KDWrapper

        m = STNet()
        print("  STNet stub constructed:", m.extra_repr())
        _ = freeze_module, KDWrapper
    except ImportError as e:
        print(f"  torch models skipped: {e}")

    print("Smoke import passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
