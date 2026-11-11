"""Matched-cell transfer rules. No city arrays and no cache."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "matched_cell",
    ROOT / "scripts" / "17_matched_cell.py",
)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def test_eligibility_uses_both_cuts() -> None:
    rho = np.array([0.8, 0.8 + 1e-6, 0.9, np.nan])
    total = np.array([150.0, 149.0, 150.0, 1000.0])
    assert mod.eligible_mask(rho, total).tolist() == [False, False, True, False]


def test_scale_matches_the_washington_level() -> None:
    scaled = mod.scale_to_level(np.array([2.0, 2.0, 2.0]), dc_level=4.0, src_level=2.0)
    assert np.allclose(scaled, [4.0, 4.0, 4.0])


def test_lag_window_is_the_six_preceding_hours() -> None:
    series = np.arange(10, dtype=np.float64)
    x, y = mod.lag_windows(series, 6, 8)
    assert np.allclose(x[0], [0, 1, 2, 3, 4, 5])
    assert y.tolist() == [6.0, 7.0]


def test_level_fit_recovers_an_affine_map() -> None:
    pred = np.array([0.0, 1.0, 2.0, 3.0])
    coef = mod.fit_level(pred, 3.0 + 2.0 * pred)
    assert np.allclose(coef, [3.0, 2.0])
    assert np.allclose(mod.apply_level(np.array([-2.0, 1.0]), np.array([0.0, 1.0])), [0.0, 1.0])


def test_test_split_stays_closed_unless_validation_is_lower() -> None:
    assert mod.opens_test(1.0, 1.0) is False
    assert mod.opens_test(1.1, 1.0) is False
    assert mod.opens_test(0.9, 1.0) is True
