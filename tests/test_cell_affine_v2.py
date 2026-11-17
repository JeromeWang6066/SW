"""Centered and rescaled affine solvers. No city arrays and no checkpoints."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load("19_cell_affine_v2.py", "cell_affine_v2")
published = _load("17_cell_affine.py", "cell_affine_v1_for_v2")

EXAMPLE_ONE = ((10.0, 10.2, 9.8, 10.1), (12.0, 12.4, 11.6, 12.2))
EXAMPLE_TWO = ((4.0, 6.0, 5.0), (8.0, 9.0, 7.0))
ATOL = 5e-4


def test_published_hand_examples() -> None:
    scale, shift = mod.solve_old(*EXAMPLE_ONE, 1.0)
    assert abs(scale - 1.1984) < ATOL
    assert abs(shift - 0.0180) < ATOL
    scale, shift = mod.solve_old(*EXAMPLE_TWO, 1.0)
    assert abs(scale - 1.5059) < ATOL
    assert abs(shift - 0.2353) < ATOL


def test_scale_only_hand_examples() -> None:
    scale, shift = mod.solve_scale_only(*EXAMPLE_ONE)
    assert abs(scale - 1.0675) < ATOL
    assert abs(shift - 0.6744) < ATOL
    scale, shift = mod.solve_scale_only(*EXAMPLE_TWO)
    assert abs(scale - 1.1845) < ATOL
    assert abs(shift - 1.0386) < ATOL


def test_centered_hand_examples() -> None:
    scale, shift = mod.solve_centered(*EXAMPLE_ONE, 1.0)
    assert abs(scale - 1.0002) < ATOL
    assert abs(shift - 1.0103) < ATOL
    scale, shift = mod.solve_centered(*EXAMPLE_TWO, 1.0)
    assert abs(scale - 0.9870) < ATOL
    assert abs(shift - 1.5649) < ATOL


def test_zero_kappa_is_least_squares_and_matches_zero_multiplier() -> None:
    scale, shift = mod.solve_centered(*EXAMPLE_TWO, 0.0)
    assert abs(scale - 0.5) < 1e-8
    assert abs(shift - 5.5) < 1e-8
    other_scale, other_shift = mod.solve_old(*EXAMPLE_TWO, 0.0)
    assert abs(scale - other_scale) < 1e-8
    assert abs(shift - other_shift) < 1e-8


def test_infinite_strength_is_identity() -> None:
    prediction = np.array([1.0, 2.0, 4.0])
    target = np.array([3.0, 1.0, 8.0])
    assert mod.solve_centered(prediction, target, math.inf) == (1.0, 0.0)
    assert mod.solve_old(prediction, target, math.inf) == (1.0, 0.0)


def test_constant_prediction_keeps_unit_scale() -> None:
    prediction = np.full(10, 5.0)
    target = np.arange(10.0)
    for kappa in (0.0, 1.0):
        scale, _shift = mod.solve_centered(prediction, target, kappa)
        assert abs(scale - 1.0) < 1e-12


def test_scale_is_unchanged_when_counts_are_rescaled() -> None:
    rng = np.random.default_rng(2)
    prediction = rng.gamma(2.0, 4.0, 20)
    target = 1.1 * prediction + rng.normal(0.0, 1.0, 20)
    for solver in (
        mod.solve_scale_only,
        lambda pred, truth: mod.solve_centered(pred, truth, 1.0),
    ):
        scale, shift = solver(prediction, target)
        scaled, shifted = solver(7.0 * prediction, 7.0 * target)
        assert abs(scale - scaled) < 1e-8
        assert abs(shifted - 7.0 * shift) < 1e-6


def test_unit_multiplier_matches_the_published_solver() -> None:
    rng = np.random.default_rng(3)
    prediction = rng.gamma(2.0, 4.0, 72)
    target = 0.9 * prediction + rng.normal(0.0, 1.0, 72)
    scale, shift = mod.solve_old(prediction, target, 1.0)
    published_scale, published_shift = published.solve_ab(prediction, target, 72.0)
    assert abs(scale - published_scale) < 1e-8
    assert abs(shift - published_shift) < 1e-8


def test_infinite_cross_validation_matches_the_uncorrected_mae() -> None:
    rng = np.random.default_rng(0)
    prediction = rng.normal(size=(72, 4))
    target = prediction + 1.0
    eligible = np.array([True, False, True, True])
    curve = mod.lodo_curve(prediction, target, eligible, mod.solve_centered)
    base = float(np.abs(prediction[:, eligible] - target[:, eligible]).mean())
    assert math.isinf(curve[-1]["strength"])
    assert abs(curve[-1]["mae"] - base) < 1e-12


def test_tie_keeps_the_larger_strength() -> None:
    curve = [{"strength": 1.0, "mae": 1.0}, {"strength": 2.0, "mae": 1.0 + 5e-10}]
    assert mod.pick_strength(curve) == 2.0


def test_identical_arms_have_a_zero_interval() -> None:
    values = np.arange(48.0).reshape(48, 1)
    point, low, high = mod.day_block_bootstrap(values, values.copy(), values)
    assert point == 0.0
    assert low == 0.0
    assert high == 0.0


def test_cells_below_the_gate_stay_at_identity() -> None:
    prediction = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    target = np.array([[5.0, 10.0], [7.0, 12.0], [9.0, 14.0]])
    scale, shift = mod.fit_cells(prediction, target, np.array([False, True]), mod.solve_centered, 0.0)
    assert scale[0] == 1.0 and shift[0] == 0.0
    assert abs(scale[1] - 2.0) < 1e-8
