"""Distances and the validation rule for the error gate. No city arrays."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feedback.error_gate import (  # noqa: E402
    choose_threshold,
    gaps,
    level_gap,
    profile_gap,
    release,
    utility_gap,
    wape_gap,
)
from src.feedback.published_hour import VAL_DAY, window_for  # noqa: E402


def _solve_old():
    spec = importlib.util.spec_from_file_location("cell_affine_v2_gate", ROOT / "scripts" / "19_cell_affine_v2.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solve_old


def test_constant_level_and_wape_are_one_half() -> None:
    pred = np.ones((72, 1))
    truth = np.full((72, 1), 2.0)
    assert abs(float(level_gap(pred, truth)[0]) - 0.5) < 1e-12
    assert abs(float(wape_gap(pred, truth)[0]) - 0.5) < 1e-12
    assert abs(float(profile_gap(pred, truth)[0]) - 0.5) < 1e-12


def test_release_is_strict_and_keeps_the_volume_gate() -> None:
    distance = np.array([0.5, 0.5, 0.2])
    eligible = np.array([True, False, True])
    assert np.array_equal(release(distance, 0.5, eligible), np.array([False, False, False]))
    assert np.array_equal(release(distance, 0.49, eligible), np.array([True, False, False]))


def test_utility_is_positive_for_a_constant_bias() -> None:
    pred = np.ones((72, 1))
    truth = np.full((72, 1), 4.0)
    assert float(utility_gap(pred, truth, _solve_old())[0]) > 0.5


def test_zero_actual_total_does_not_release() -> None:
    pred = np.ones((72, 1))
    truth = np.zeros((72, 1))
    assert float(level_gap(pred, truth)[0]) == 0.0
    assert not bool(release(level_gap(pred, truth), 0.0, np.array([True]))[0])


def test_editing_the_target_day_does_not_change_its_distance() -> None:
    solve_old = _solve_old()
    day = VAL_DAY + 2
    window = window_for(day, "CL")
    assert day not in window
    hours = np.concatenate([np.arange(item * 24, (item + 1) * 24) for item in window])
    rng = np.random.default_rng(0)
    pred = rng.random(((day + 1) * 24, 2)) + 1.0
    truth = pred + rng.normal(scale=0.1, size=pred.shape)
    before = gaps(pred[hours], truth[hours], solve_old)
    truth[day * 24 : (day + 1) * 24, 0] += 6.0
    after = gaps(pred[hours], truth[hours], solve_old)
    for name in before:
        assert np.allclose(before[name], after[name])


def test_choose_threshold_keeps_a_gate_only_when_both_cities_hold() -> None:
    rows = [
        {"tau": 0.05, "ny": 1.0, "chi": 0.9},
        {"tau": 0.10, "ny": 0.8, "chi": 0.8},
        {"tau": 0.20, "ny": 0.7, "chi": 1.2},
    ]
    chosen = choose_threshold(rows, 1.0, 1.0)
    assert chosen is not None
    assert chosen["tau"] == 0.10
    assert choose_threshold([{"tau": 0.1, "ny": 0.9, "chi": 1.1}], 1.0, 1.0) is None
    assert choose_threshold([{"tau": 0.0, "ny": 1.0, "chi": 1.0}], 1.0, 1.0) is None
