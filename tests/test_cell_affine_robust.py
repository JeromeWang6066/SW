"""Toy checks for the hour-bias robustness rules. No city arrays."""

from __future__ import annotations

import datetime as dt
import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "24_cell_affine_robust.py"
    spec = importlib.util.spec_from_file_location("cell_affine_robust", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RobustRuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load()

    def test_test_window_starts_on_wednesday(self):
        first = self.mod.test_day_dates(24)[0]
        self.assertEqual(first, dt.date(2016, 11, 2))
        self.assertEqual(first.weekday(), 2)
        mask = self.mod.weekday_mask(24)
        self.assertTrue(bool(mask[0]))
        self.assertFalse(bool(self.mod.weekday_mask(24 * 4)[24 * 3]))

    def test_week_blocks_cover_sixty_days(self):
        lengths = self.mod.block_lengths(60, 7)
        self.assertEqual(lengths, [7, 7, 7, 7, 7, 7, 7, 7, 4])

    def test_leave_one_day_uses_the_other_two(self):
        residual = np.zeros(72, dtype=np.float64)
        residual[0] = 10.0
        residual[24] = 2.0
        residual[48] = 4.0
        self.assertAlmostEqual(self.mod.leave_one_day_offsets(residual, 0)[0], (2.0 + 4.0) / 5.0)
        self.assertAlmostEqual(self.mod.leave_one_day_offsets(residual, 1)[0], (10.0 + 4.0) / 5.0)
        self.assertAlmostEqual(self.mod.leave_one_day_offsets(residual, 0)[1], 0.0)

    def test_true_clock_beats_a_scramble(self):
        hours = 48
        base = np.zeros((hours, 1), dtype=np.float64)
        target = np.zeros((hours, 1), dtype=np.float64)
        target[0::24, 0] = 10.0
        offsets = np.zeros((24, 1), dtype=np.float64)
        offsets[0, 0] = 10.0
        found = self.mod.clock_permutation_count(base, offsets, target, n_perm=200, seed=0)
        self.assertAlmostEqual(found["true_mae"], 0.0)
        # A scramble still matches when it keeps hour 0 in place, about 1 in 24 draws.
        self.assertLess(found["n_null_le_true"], 20)
        self.assertGreater(found["null_p05"], 0.0)


if __name__ == "__main__":
    unittest.main()
