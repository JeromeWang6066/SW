"""Hand checks for the frozen pure-scale reforms. No city data."""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "22_cell_affine_reform.py"
    spec = importlib.util.spec_from_file_location("cell_affine_reform", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReformFormulaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load()

    def test_multiplicative_closed_form(self):
        pred = np.array([4.0, 6.0, 5.0])
        truth = np.array([8.0, 9.0, 7.0])
        # p·p = 77, r = (4, 3, 2), p·r = 44, kappa = 1 -> ã = 44/154.
        scale, shift = self.mod.solve_multiplicative(pred, truth, 1.0)
        self.assertAlmostEqual(scale, 1.0 + 44.0 / 154.0, places=9)
        self.assertEqual(shift, 0.0)
        unpenalized, _ = self.mod.solve_multiplicative(pred, truth, 0.0)
        self.assertAlmostEqual(unpenalized, 121.0 / 77.0, places=9)

    def test_limits_and_scale_invariance(self):
        pred = np.array([10.0, 10.2, 9.8, 10.1])
        truth = np.array([12.0, 12.4, 11.6, 12.2])
        self.assertEqual(self.mod.solve_multiplicative(pred, truth, math.inf), (1.0, 0.0))
        self.assertEqual(self.mod.solve_l1_scale(pred, truth, math.inf), (1.0, 0.0))
        self.assertEqual(self.mod.solve_huber_scale(pred, truth, math.inf), (1.0, 0.0))
        scale, shift = self.mod.solve_multiplicative(pred, truth, 1.0)
        scaled, _ = self.mod.solve_multiplicative(pred * 7.0, truth * 7.0, 1.0)
        self.assertAlmostEqual(scale, scaled, places=9)
        self.assertEqual(shift, 0.0)
        flat, _ = self.mod.solve_multiplicative(np.zeros(4), truth, 1.0)
        self.assertEqual((flat, 0.0), (1.0, 0.0))

    def test_l1_exact_line_and_pull_to_identity(self):
        pred = np.array([1.0, 2.0, 3.0])
        truth = np.array([2.0, 4.0, 6.0])
        scale, shift = self.mod.solve_l1_scale(pred, truth, 1.0)
        self.assertAlmostEqual(scale, 2.0, places=5)
        self.assertEqual(shift, 0.0)
        # J(ã)=|2-ã|+|ã|+2|ã| on p=(1,1), y=(3,1). Minimum is ã=0.
        scale, shift = self.mod.solve_l1_scale(np.array([1.0, 1.0]), np.array([3.0, 1.0]), 1.0)
        self.assertAlmostEqual(scale, 1.0, places=4)
        self.assertEqual(shift, 0.0)

    def test_huber_matches_l2_inside_delta(self):
        pred = np.array([4.0, 6.0, 5.0])
        truth = np.array([8.0, 9.0, 7.0])
        l2, _ = self.mod.solve_multiplicative(pred, truth, 1.0)
        huber, shift = self.mod.solve_huber_scale(pred, truth, 1.0, delta=100.0)
        self.assertAlmostEqual(huber, l2, places=8)
        self.assertEqual(shift, 0.0)

    def test_clip_removes_only_negatives(self):
        pred = np.array([[1.0, -2.0], [3.0, 4.0]])
        scale = np.array([1.0, 1.0])
        shift = np.array([0.0, 0.0])
        clipped = self.mod._apply(pred, scale, shift, True)
        self.assertTrue(np.array_equal(clipped, np.array([[1.0, 0.0], [3.0, 4.0]])))
        raw = self.mod._apply(pred, scale, shift, False)
        self.assertAlmostEqual(float(raw.min()), -2.0)


if __name__ == "__main__":
    unittest.main()
