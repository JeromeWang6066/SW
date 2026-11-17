"""Frozen protocol constants for the three-stage val-selection experiment."""

from __future__ import annotations

from typing import Any

# Explicit identity is a separate branch, not a large finite strength and not math.inf.
IDENTITY: str = "identity"

CANDIDATE_STRENGTHS: tuple[float, ...] = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
CANDIDATES: tuple[Any, ...] = CANDIDATE_STRENGTHS + (IDENTITY,)

SOURCES: tuple[str, ...] = ("ny", "chi")
PRELOCKED_SEEDS: tuple[int, ...] = (42, 43, 44)

# Arm ids. "S"/scale-only from PR20 is deliberately absent.
ARMS: tuple[str, ...] = (
    "FT",
    "OLD_FIXED",
    "NEW_FIXED",
    "OLD_VAL",
    "NEW_VAL",
    "SHIFT_VAL",
)
TUNABLE_ARMS: tuple[str, ...] = ("OLD_VAL", "NEW_VAL", "SHIFT_VAL")
FIXED_ARMS: tuple[str, ...] = ("FT", "OLD_FIXED", "NEW_FIXED")

ARM_METHOD: dict[str, str] = {
    "FT": "identity",
    "OLD_FIXED": "old",
    "NEW_FIXED": "centered",
    "OLD_VAL": "old",
    "NEW_VAL": "centered",
    "SHIFT_VAL": "shift",
}
ARM_FIXED_STRENGTH: dict[str, Any] = {
    "FT": IDENTITY,
    "OLD_FIXED": 1.0,  # m=1 => lambda = n
    "NEW_FIXED": 1.0,  # kappa=1
}

GATE = 150
N_TRAIN_HOURS = 72
N_VALID_CELLS_EXPECTED = 207
N_GATE_EXPECTED = 50
LABEL_START_TRAIN = 5832
EXPECTED_VAL_START = 5904
EXPECTED_VAL_HOURS = 1440
EXPECTED_TEST_START = 7344
EXPECTED_TEST_HOURS = 1440
N_VAL_BLOCKS = 4

# Near-constant detection only, frozen, not retuned from results.
# A series is near-constant when range(p) <= NEAR_CONSTANT_REL * max(1, max|p|).
# This does not add a ridge term or change the well-posed OLS objective.
NEAR_CONSTANT_REL = 1e-15

# Default val/test index ranges are the STBaT-style chain after a T=8784 crop.
# They are protocol defaults pending confirmation from the real generation chain.
# Hour *numbers* are unchanged. Status is now an input-protocol freeze:
# README + bundled script document the ranges; val/test starts are not
# independently proven by executing prepare_city_windows (that helper is a
# repo dependency, not shipped as a stand-alone proof in the bundle).
VAL_TEST_RANGE_STATUS = "documented_range_from_bundle_pending_independent_raw_recompute"
UNIT_DC_CLIPPED_COUNT = "dc_clipped_count"
NORM_PERCENTILE = 99.9
NORM_FITTED_ON = "full_raw_series"

# Documented bundle constants (input protocol only; not a new grid).
DOCUMENTED_VALUE_SCALE = 234.0
DOCUMENTED_MIN_VAL = 0.0
DOCUMENTED_MAX_VAL = 234.0
# Observed residual of train y vs clip(raw, 0, DOCUMENTED_MAX_VAL).
# This is NOT byte identity. Shared-task y *within one split* is byte-identical.
DOCUMENTED_TRAIN_Y_VS_CLIP_RAW_MAXABS = 7.62939453125e-06
DOCUMENTED_PREDICT_SCRIPT_REVISION = "7a04ca4"
DOCUMENTED_PREDICT_SCRIPT_FORM = "documented_short_revision"
FINE_TUNE_COMMIT_STATUS = "unknown"
CHECKPOINT_HASH_STATUS = "claimed"
TIME_INDEX_ORIGIN_DOCUMENTED = "documented_range_from_bundle"
PROVENANCE_DOCUMENTED = "documented_from_bundle"

# Numerical tie tolerance only, not a practical performance plateau.
TIE_EPS = 1e-10

# Algorithm/grid/tie-break protocol. Unchanged by the input-mode freeze below.
PROTOCOL_ID = "val-select-kappa-v3"
PROTOCOL_FROZEN = True
# Input-mode freeze only. Does not change candidates, arms, solvers, or tie-break.
INPUT_PROTOCOL_ID = "val-select-input-bundle-v1"

# Label-budget disclosure (must appear in every report).
LABEL_BUDGET = (
    "3-day target training (72h) is used only to fit per-cell parameters; "
    "this protocol also uses the official ~60-day DC validation labels to "
    "select strength. This is not a 72h-only pipeline. Existing test labels "
    "have already been seen in prior work; test scores are exploratory and "
    "must not change selections or protocol."
)


def is_identity(strength: Any) -> bool:
    return strength == IDENTITY or strength == "inf" or strength is IDENTITY


def strength_sort_key(strength: Any) -> tuple[int, float]:
    """Identity is treated as stronger shrinkage than any finite candidate."""
    if is_identity(strength):
        return (1, float("inf"))
    return (0, float(strength))


def tie_tolerance(min_loss: float) -> float:
    return TIE_EPS * max(1.0, abs(float(min_loss)))


def pick_strength(curve: list[dict[str, Any]]) -> Any:
    """Argmin of val citywide MAE with numerical-tie breaking.

    Tolerance is 1e-10 * max(1, |min_loss|). Among numerically tied
    candidates, prefer exact identity, then the larger finite strength.
    This is not a practical plateau and must not be retuned from results.
    """
    if not curve:
        raise ValueError("empty selection curve")
    losses = []
    for item in curve:
        loss = item["mae"]
        if loss is None or not _finite(loss):
            raise ValueError("selection curve contains a non-finite MAE; cannot silently drop a candidate")
        losses.append(float(loss))
    best = min(losses)
    tol = tie_tolerance(best)
    tied = [item["strength"] for item, loss in zip(curve, losses) if loss - best <= tol]
    if any(is_identity(strength) for strength in tied):
        return IDENTITY
    return max(tied, key=lambda s: float(s))


def _finite(value: Any) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return v == v and v not in (float("inf"), float("-inf"))


def canonical_strength(strength: Any) -> str:
    if is_identity(strength):
        return IDENTITY
    text = format(float(strength), ".15g")
    return text
