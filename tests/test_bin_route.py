"""Cell routing edges and stitched MAE. No city arrays and no checkpoints."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.pearson_weekday_profile import assign_similarity_bin

_SPEC = importlib.util.spec_from_file_location(
    "bin_route",
    ROOT / "scripts" / "13_bin_route.py",
)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def _totals(ae: list[float], hours: float = 2.0) -> dict:
    values = np.asarray(ae, dtype=np.float64)
    return {
        "se": values,
        "ae": values,
        "ape": np.zeros_like(values),
        "mape_points": np.ones_like(values),
        "hours": hours,
    }


def test_route_edges() -> None:
    rho = np.array([0.8 + 1e-6, 0.8, 0.5, np.nan])
    choices = [mod.route_choice(int(code)) for code in assign_similarity_bin(rho)]
    assert choices == ["finetune", "kd", "scratch", "scratch"]


def test_stitch_mae_is_the_selected_cells() -> None:
    hours = 2.0
    totals = {
        "scratch": _totals([1.0, 100.0, 100.0], hours),
        "finetune": _totals([100.0, 2.0, 100.0], hours),
        "kd": _totals([100.0, 100.0, 3.0], hours),
    }
    choices = np.array(["scratch", "finetune", "kd"], dtype=object)
    routed = mod.stitch_cells(totals, choices)
    metrics = mod.metrics_from_cells(
        routed["se"], routed["ae"], routed["ape"], routed["mape_points"], routed["hours"], scale=1.0
    )
    assert metrics["mae"] == np.float64(1.0) or abs(metrics["mae"] - 1.0) < 1e-12
    scratch = mod.metrics_from_cells(
        totals["scratch"]["se"],
        totals["scratch"]["ae"],
        totals["scratch"]["ape"],
        totals["scratch"]["mape_points"],
        hours,
        scale=1.0,
    )
    all_scratch = mod.stitch_cells(totals, np.array(["scratch", "scratch", "scratch"], dtype=object))
    replay = mod.metrics_from_cells(
        all_scratch["se"],
        all_scratch["ae"],
        all_scratch["ape"],
        all_scratch["mape_points"],
        all_scratch["hours"],
        scale=1.0,
    )
    assert abs(replay["mae"] - scratch["mae"]) < 1e-12
