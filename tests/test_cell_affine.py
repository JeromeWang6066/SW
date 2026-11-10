"""Per-cell scale and shift. No city arrays and no checkpoints."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "cell_affine",
    ROOT / "scripts" / "17_cell_affine.py",
)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def test_zero_lambda_recovers_the_line() -> None:
    prediction = np.array([1.0, 2.0, 3.0, 4.0])
    target = 2.0 * prediction + 3.0
    scale, shift = mod.solve_ab(prediction, target, lam=0.0)
    assert abs(scale - 2.0) < 1e-8
    assert abs(shift - 3.0) < 1e-8


def test_ridge_pulls_toward_identity() -> None:
    prediction = np.array([1.0, 2.0, 3.0, 4.0])
    target = 2.0 * prediction + 3.0
    scale, shift = mod.solve_ab(prediction, target, lam=1.0e6)
    assert abs(scale - 1.0) < 1e-3
    assert abs(shift - 0.0) < 1e-3


def test_cells_below_the_gate_keep_the_original_prediction() -> None:
    prediction = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    target = np.array([[5.0, 10.0], [7.0, 12.0], [9.0, 14.0]])
    scale, shift = mod.fit_affine(prediction, target, np.array([False, True]), lam=0.0)
    assert abs(scale[0] - 1.0) < 1e-12
    assert abs(shift[0] - 0.0) < 1e-12
    assert abs(scale[1] - 2.0) < 1e-8
    assert abs(shift[1] - 8.0) < 1e-8
    corrected = mod.apply_affine(prediction, scale, shift)
    assert np.allclose(corrected[:, 0], prediction[:, 0])
    assert np.allclose(corrected[:, 1], target[:, 1])
