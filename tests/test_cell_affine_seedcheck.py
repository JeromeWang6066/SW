"""Toy checks for the weekday seed verification. No city arrays and no training."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "25_cell_affine_seedcheck.py"
    spec = importlib.util.spec_from_file_location("cell_affine_seedcheck", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SeedCheckRuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load()

    def test_weekday_mask_drops_the_first_weekend(self):
        keep = self.mod.weekday_hours(24 * 7)
        self.assertEqual(int(keep.sum()), 24 * 5)
        self.assertTrue(bool(keep[: 24 * 3].all()))
        self.assertFalse(bool(keep[24 * 3 : 24 * 5].any()))

    def test_slice_mae_ignores_dropped_hours(self):
        prediction = np.zeros((48, 1), dtype=np.float64)
        target = np.zeros((48, 1), dtype=np.float64)
        target[24:, 0] = 10.0
        keep = np.zeros(48, dtype=bool)
        keep[:24] = True
        self.assertAlmostEqual(self.mod.slice_mae(prediction, target, keep), 0.0)

    def test_bootstrap_uses_only_weekday_days(self):
        hours, cells = 48, 1
        candidate = np.zeros((hours, cells))
        reference = np.ones((hours, cells))
        target = np.zeros((hours, cells))
        keep = np.zeros(hours, dtype=bool)
        keep[:24] = True
        found = self.mod.weekday_bootstrap(candidate, reference, target, keep, reps=20, seed=0)
        self.assertEqual(found["n_days"], 1)
        self.assertAlmostEqual(found["point"], -1.0)


if __name__ == "__main__":
    unittest.main()
