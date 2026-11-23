"""Window rules and the published hour-bias formula. No city arrays and no checkpoints."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feedback.published_hour import (  # noqa: E402
    HOUR_KAPPA,
    TRAIN_DAY,
    VAL_DAY,
    apply_day,
    correct_range,
    day_date,
    fit_window,
    hour_offsets,
    is_weekend,
    previous_days,
    weekday_rows,
    window_for,
)


def _solve_old():
    spec = importlib.util.spec_from_file_location("cell_affine_v2_loop", ROOT / "scripts" / "19_cell_affine_v2.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solve_old


def test_three_day_constant_residual_shrinks_by_half() -> None:
    residual = np.full(72, 4.0)
    offsets = hour_offsets(residual, HOUR_KAPPA)
    assert offsets.shape == (24,)
    assert np.allclose(offsets, 2.0)


def test_two_day_denominator_keeps_kappa_three() -> None:
    residual = np.full(48, 5.0)
    offsets = hour_offsets(residual, HOUR_KAPPA)
    assert np.allclose(offsets, 10.0 / 5.0)


def test_line_matches_solve_old() -> None:
    solve_old = _solve_old()
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(72, 4)) + 3.0
    truth = pred + rng.normal(scale=0.2, size=pred.shape)
    keep = np.array([True, False, True, False])
    scale, shift, offsets = fit_window(pred, truth, keep, solve_old, with_hour=True)
    for index in (0, 2):
        found_a, found_b = solve_old(pred[:, index], truth[:, index], 1.0)
        assert abs(scale[index] - found_a) < 1e-8
        assert abs(shift[index] - found_b) < 1e-8
        residual = truth[:, index] - (found_a * pred[:, index] + found_b)
        assert np.allclose(offsets[:, index], hour_offsets(residual), atol=1e-8)
    assert np.allclose(scale[[1, 3]], 1.0)
    assert np.allclose(shift[[1, 3]], 0.0)
    assert np.allclose(offsets[:, [1, 3]], 0.0)


def test_labor_day_is_a_weekday_and_uses_the_training_window() -> None:
    labor = TRAIN_DAY + 5  # 2016-09-05
    assert day_date(labor).isoformat() == "2016-09-05"
    assert is_weekend(labor) is False
    assert [day_date(day).isoformat() for day in window_for(labor, "CL")] == [
        "2016-08-31",
        "2016-09-01",
        "2016-09-02",
    ]


def test_first_saturday_uses_three_earlier_weekend_days() -> None:
    saturday = VAL_DAY
    assert day_date(saturday).isoformat() == "2016-09-03"
    dates = [day_date(day).isoformat() for day in window_for(saturday, "CL")]
    assert dates == ["2016-08-21", "2016-08-27", "2016-08-28"]
    assert "2016-09-02" not in dates
    assert all(is_weekend(day) for day in window_for(saturday, "CL"))


def test_consecutive_window_is_the_three_days_before() -> None:
    assert window_for(VAL_DAY, "CL72") == [VAL_DAY - 3, VAL_DAY - 2, VAL_DAY - 1]
    assert previous_days(10, 3, same_class=False) == [7, 8, 9]


def test_editing_a_day_does_not_change_that_day() -> None:
    solve_old = _solve_old()
    rng = np.random.default_rng(1)
    hours = 20 * 24
    pred = np.abs(rng.normal(size=(hours, 3))) + 1.0
    truth = pred + rng.normal(scale=0.3, size=pred.shape)
    keep = np.array([True, True, False])
    first = 10
    original = correct_range(pred, truth, keep, solve_old, "CL", first, 18)
    edited_truth = truth.copy()
    edited_truth[14 * 24 : 15 * 24, 0] += 5.0
    edited = correct_range(pred, edited_truth, keep, solve_old, "CL", first, 18)
    assert np.array_equal(edited[14 * 24 : 15 * 24], original[14 * 24 : 15 * 24])
    later = 15
    while later < 18 and is_weekend(later) != is_weekend(14):
        later += 1
    assert later < 18
    assert not np.array_equal(edited[later * 24 : (later + 1) * 24, 0], original[later * 24 : (later + 1) * 24, 0])


def test_cells_below_the_gate_stay_at_the_fine_tune() -> None:
    solve_old = _solve_old()
    rng = np.random.default_rng(2)
    pred = np.abs(rng.normal(size=(16 * 24, 3))) + 1.0
    truth = pred + 0.4
    keep = np.array([True, False, False])
    out = correct_range(pred, truth, keep, solve_old, "CL", 8, 16)
    assert np.array_equal(out[:, 1:], pred[:, 1:])


def test_weekday_mask_drops_saturday_and_sunday_only() -> None:
    mask = weekday_rows(VAL_DAY, VAL_DAY + 7)
    assert mask.shape == (7 * 24,)
    assert int((~mask).sum()) == 2 * 24
    assert int(mask.sum()) == 5 * 24


def test_bootstrap_of_identical_arms_is_zero() -> None:
    spec = importlib.util.spec_from_file_location("cell_affine_v2_boot", ROOT / "scripts" / "19_cell_affine_v2.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = np.ones((48, 2))
    point, low, high = module.day_block_bootstrap(values, values, values, n_rep=20, seed=0)
    assert point == 0.0 and low == 0.0 and high == 0.0


def test_apply_day_uses_the_clock_hour() -> None:
    pred = np.zeros((24, 1))
    offsets = np.arange(24, dtype=np.float64).reshape(24, 1)
    out = apply_day(pred, np.ones(1), np.zeros(1), offsets)
    assert np.allclose(out[:, 0], np.arange(24))
