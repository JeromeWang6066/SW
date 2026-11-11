"""Source-shape blend edges. No city arrays and no checkpoints."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "shape_blend",
    ROOT / "scripts" / "18_shape_blend.py",
)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def test_donors_stay_near_the_best_and_above_0_8() -> None:
    rho = np.array([0.97, 0.955, 0.90, np.nan])
    assert mod.donor_mask(rho).tolist() == [True, True, False, False]
    low = np.array([0.79, 0.78, 0.70])
    assert mod.donor_mask(low).tolist() == [False, False, False]


def test_average_is_scaled_to_the_three_day_level() -> None:
    profiles = np.array(
        [
            [2.0, 0.0] + [0.0] * 22,
            [0.0, 2.0] + [0.0] * 22,
        ]
    )
    shape = mod.scaled_shape(profiles, three_total=30.0)
    assert abs(float(shape.sum()) - 10.0) < 1e-8
    assert abs(float(shape[0] - shape[1])) < 1e-8


def test_missing_shape_keeps_the_finetune_forecast() -> None:
    prediction = np.array([[4.0, 8.0], [6.0, 10.0]])
    shape = np.array([[1.0, np.nan], [1.0, np.nan]])
    mixed = mod.blend(prediction, shape, weight=1.0)
    assert np.allclose(mixed[:, 0], 1.0)
    assert np.allclose(mixed[:, 1], prediction[:, 1])
