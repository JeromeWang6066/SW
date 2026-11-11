"""Volume-gate and 3x3 routing edges. No city arrays and no cache."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "grid_route",
    ROOT / "scripts" / "16_grid_route.py",
)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def test_gate_includes_150_and_excludes_149() -> None:
    totals = np.array([149.0, 150.0, 151.0])
    assert mod.passes_gate(totals).tolist() == [False, True, True]


def test_cells_below_the_gate_stay_on_finetune() -> None:
    bins = np.array([0, -1, 0, 2])
    totals = np.array([149.0, 10.0, 150.0, 151.0])
    choices = mod.rule_choices(bins, bins, totals)
    assert choices["gate"].tolist() == ["finetune", "finetune", "scratch", "finetune"]
    assert choices["finetune"].tolist() == ["finetune", "finetune", "finetune", "finetune"]


def test_box_rule_uses_the_3x3_bins() -> None:
    bins_1km = np.array([2, 2, 2, 2])
    bins_box = np.array([0, 1, -1, 2])
    totals = np.array([0.0, 0.0, 1000.0, 1000.0])
    choices = mod.rule_choices(bins_1km, bins_box, totals)
    assert choices["box"].tolist() == ["scratch", "kd", "scratch", "finetune"]
    assert choices["gate"].tolist() == ["finetune", "finetune", "finetune", "finetune"]
