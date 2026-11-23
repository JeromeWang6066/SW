"""POI function labels. No city arrays."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location("function_cells", ROOT / "scripts" / "15_function_cells.py")
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def test_commercial_and_residential_need_a_clear_majority() -> None:
    commercial = np.zeros(14)
    commercial[10] = 8
    commercial[11] = 2
    assert mod.function_label(commercial) == "商业就业"
    residential = np.zeros(14)
    residential[3] = 9
    residential[11] = 1
    assert mod.function_label(residential) == "居住"
    assert mod.function_label(np.zeros(14)) == "无兴趣点"
    mixed = np.ones(14)
    assert mod.function_label(mixed) == "混合"


def test_neighbor_rate_ignores_invalid_cells() -> None:
    labels = np.array([["商业就业", "商业就业"], ["居住", ""]], dtype=object)
    valid = np.array([[True, True], [True, False]])
    rates = mod.neighbor_same_rate(labels, valid)
    assert rates.shape == (3,)
    assert abs(rates[0] - 0.5) < 1e-12
    assert abs(rates[1] - 0.5) < 1e-12
    assert abs(rates[2] - 0.0) < 1e-12
