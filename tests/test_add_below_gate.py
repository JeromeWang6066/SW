"""Holdout votes for cells below the 150 gate. No city arrays."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feedback.add_below_gate import admit, holdout_wins  # noqa: E402
from src.feedback.published_hour import VAL_DAY, window_for  # noqa: E402


def _solve_old():
    spec = importlib.util.spec_from_file_location("cell_affine_v2_add", ROOT / "scripts" / "19_cell_affine_v2.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solve_old


def test_constant_bias_wins_every_holdout() -> None:
    pred = np.ones((72, 1))
    truth = np.full((72, 1), 4.0)
    wins = holdout_wins(pred, truth, _solve_old(), np.array([True]))
    assert int(wins[0]) == 3


def test_exact_forecast_wins_nothing() -> None:
    pred = np.ones((72, 1))
    wins = holdout_wins(pred, pred.copy(), _solve_old(), np.array([True]))
    assert int(wins[0]) == 0


def test_cells_outside_the_candidate_list_are_not_fit() -> None:
    pred = np.ones((72, 2))
    truth = np.full((72, 2), 4.0)
    wins = holdout_wins(pred, truth, _solve_old(), np.array([False, True]))
    assert int(wins[0]) == 0
    assert int(wins[1]) == 3


def test_two_votes_keep_the_volume_gate_and_a_two_win_cell() -> None:
    eligible = np.array([True, False, False])
    wins = np.array([0, 2, 1])
    assert np.array_equal(admit(wins, 2, eligible), np.array([True, True, False]))
    assert np.array_equal(admit(wins, 3, eligible), np.array([True, False, False]))


def test_editing_the_target_day_does_not_change_its_votes() -> None:
    solve_old = _solve_old()
    day = VAL_DAY + 2
    window = window_for(day, "CL")
    assert day not in window
    hours = np.concatenate([np.arange(item * 24, (item + 1) * 24) for item in window])
    rng = np.random.default_rng(0)
    pred = rng.random(((day + 1) * 24, 2)) + 1.0
    truth = pred + 0.4
    candidates = np.array([True, True])
    before = holdout_wins(pred[hours], truth[hours], solve_old, candidates)
    truth[day * 24 : (day + 1) * 24, 0] += 6.0
    after = holdout_wins(pred[hours], truth[hours], solve_old, candidates)
    assert np.array_equal(before, after)
