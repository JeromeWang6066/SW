"""Toy checks for the next cell-affine arms. No city arrays."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "23_cell_affine_next.py"
    spec = importlib.util.spec_from_file_location("cell_affine_next", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NextFormulaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load()

    def test_hour_offset_is_half_the_clock_mean(self):
        residual = np.array([2.0, 0.0] + [0.0] * 22 + [4.0, 0.0] + [0.0] * 22)
        offsets = self.mod.hour_offsets(residual, kappa=3.0)
        self.assertAlmostEqual(offsets[0], (2.0 + 4.0) / 5.0, places=9)
        self.assertAlmostEqual(offsets[1], 0.0, places=9)
        self.assertEqual(offsets.shape, (24,))

    def test_median_intercept_given_slope(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        truth = np.array([1.0, 2.0, 3.0, 100.0])
        # Slope fixed at 1. Residuals of y-p are 0,0,0,96. Median intercept is 0.
        self.assertAlmostEqual(self.mod.median_intercept(pred, truth, 1.0), 0.0, places=9)

    def test_lad_resists_one_outlier_more_than_least_squares(self):
        v2_path = ROOT / "scripts" / "19_cell_affine_v2.py"
        spec = importlib.util.spec_from_file_location("cell_affine_v2_next_test", v2_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        truth = np.array([1.0, 2.0, 3.0, 4.0, 40.0])
        lad, _ = self.mod.solve_lad_ridge(pred, truth, lam=0.0, solve_old=module.solve_old)
        least, _ = module.solve_old(pred, truth, 0.0)
        self.assertLess(abs(lad - 1.0), abs(least - 1.0))

    def test_perfect_line_stays_put(self):
        v2_path = ROOT / "scripts" / "19_cell_affine_v2.py"
        spec = importlib.util.spec_from_file_location("cell_affine_v2_next_test_line", v2_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pred = np.arange(1.0, 9.0)
        truth = 2.0 * pred + 1.0
        scale, shift = self.mod.solve_lad_ridge(pred, truth, lam=0.0, solve_old=module.solve_old)
        self.assertAlmostEqual(scale, 2.0, places=3)
        self.assertAlmostEqual(shift, 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
