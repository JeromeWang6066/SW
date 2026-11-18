"""Frozen constants for dual-kappa pilot protocol v1.1-pilot."""

from __future__ import annotations

from typing import Any

# Protocol identity. Distinct from val-select-kappa-v3.
PROTOCOL_ID = "dual-kappa-pilot-v1.1"
PROTOCOL_VERSION = "v1.1-pilot"
INPUT_PROTOCOL_ID = "val-select-input-bundle-v1"  # inherited input mode, not a new input protocol
PROTOCOL_FROZEN = True

IDENTITY: str = "identity"

# Inherited source/seed/gate/unit/time conventions. Not retuned.
SOURCES: tuple[str, ...] = ("ny", "chi")
PRELOCKED_SEEDS: tuple[int, ...] = (42, 43, 44)

GATE = 150
N_TRAIN_HOURS = 72
N_VALID_CELLS_EXPECTED = 207
N_GATE_EXPECTED = 50
LABEL_START_TRAIN = 5832
EXPECTED_VAL_START = 5904
EXPECTED_VAL_HOURS = 1440
EXPECTED_TEST_START = 7344
EXPECTED_TEST_HOURS = 1440

NEAR_CONSTANT_REL = 1e-15
TIE_EPS = 1e-10
UNIT_DC_CLIPPED_COUNT = "dc_clipped_count"
DOCUMENTED_VALUE_SCALE = 234.0
TIME_INDEX_ORIGIN_DOCUMENTED = "documented_range_from_bundle"
PROVENANCE_DOCUMENTED = "documented_from_bundle"
VAL_TEST_RANGE_STATUS = "documented_range_from_bundle_pending_independent_raw_recompute"

# Grids. G is the inherited single-kappa finite set. H generates off-diagonal pairs.
G: tuple[float, ...] = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
H: tuple[float, ...] = (0.0, 0.25, 1.0, 4.0, 16.0)

# Five arms only. No SHIFT_VAL, no OLD_VAL, no rolling, no four-arm diagnosis.
ARMS: tuple[str, ...] = (
    "FT",
    "OLD_FIXED",
    "NEW_FIXED",
    "SINGLE_VAL10",
    "DOUBLE_VAL30",
)
FIXED_ARMS: tuple[str, ...] = ("FT", "OLD_FIXED", "NEW_FIXED")
TUNABLE_ARMS: tuple[str, ...] = ("SINGLE_VAL10", "DOUBLE_VAL30")

# Pre-locked STOP thresholds. Engineering screens, not significance.
# All four source×reference MAE comparisons and all four RMSE comparisons,
# plus the 4/6 task rule, must hold together. Do not retune after seeing results.
STOP_MAE_REL_MIN = 0.001
STOP_RMSE_RATIO_MAX = 1.005
STOP_N_TASKS_LE_OLD_MIN = 4
STOP_EQ_ABS = 1e-10
STOP_EQ_REL = 1e-12
STOP_REFERENCES: tuple[str, ...] = ("OLD_FIXED", "SINGLE_VAL10")

LABEL_BUDGET = (
    "3-day target training (72h) is used only to fit per-cell parameters; "
    "this protocol also uses the official ~60-day DC validation labels to "
    "select strengths. SINGLE_VAL10 searches 9 diagonal pairs + identity (10). "
    "DOUBLE_VAL30 searches 9 diagonal + 20 off-diagonal + identity (30). "
    "The 30-candidate search is a larger budget than the 10-candidate subset "
    "and is not an equal-complexity comparison. Existing test labels have "
    "already been seen in prior work; any test scores are exploratory and "
    "must not change selections, grids, or STOP thresholds."
)

BUDGET_NOTE = (
    "budget_nested10 is a subset of budget_expand30; 30 vs 10 are unequal "
    "search budgets and must be reported separately. Main paired claims after "
    "test still use the pre-locked STOP rule comparing DOUBLE_VAL30 to "
    "OLD_FIXED and to SINGLE_VAL10. Do not drop off-diagonal candidates "
    "after seeing scores to pretend equal complexity."
)


def is_identity(strength: Any) -> bool:
    return strength == IDENTITY or strength == "inf" or strength is IDENTITY


def _finite(value: Any) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return v == v and v not in (float("inf"), float("-inf"))


def canonical_strength_pair(candidate: Any) -> str:
    if is_identity(candidate) or candidate == IDENTITY:
        return IDENTITY
    if isinstance(candidate, str) and candidate == IDENTITY:
        return IDENTITY
    if isinstance(candidate, str) and "," in candidate:
        left, right = candidate.split(",", 1)
        return f"{format(float(left), '.15g')},{format(float(right), '.15g')}"
    if isinstance(candidate, dict):
        if is_identity(candidate.get("kind")) or candidate.get("kind") == IDENTITY:
            return IDENTITY
        return (
            f"{format(float(candidate['kappa_alpha']), '.15g')},"
            f"{format(float(candidate['kappa_c']), '.15g')}"
        )
    if isinstance(candidate, (tuple, list)) and len(candidate) == 2:
        return f"{format(float(candidate[0]), '.15g')},{format(float(candidate[1]), '.15g')}"
    raise ValueError(f"unrecognized dual-kappa candidate: {candidate!r}")


def parse_candidate(candidate: Any) -> Any:
    if is_identity(candidate):
        return IDENTITY
    if isinstance(candidate, dict):
        if is_identity(candidate.get("kind")) or candidate.get("kind") == IDENTITY:
            return IDENTITY
        return (float(candidate["kappa_alpha"]), float(candidate["kappa_c"]))
    if isinstance(candidate, str):
        if candidate == IDENTITY:
            return IDENTITY
        left, right = candidate.split(",", 1)
        return (float(left), float(right))
    if isinstance(candidate, (tuple, list)) and len(candidate) == 2:
        return (float(candidate[0]), float(candidate[1]))
    raise ValueError(f"unrecognized dual-kappa candidate: {candidate!r}")


def _pair_tuple(candidate: Any) -> tuple[float, float] | None:
    parsed = parse_candidate(candidate)
    if is_identity(parsed):
        return None
    return (float(parsed[0]), float(parsed[1]))


def diagonal_pairs() -> tuple[tuple[float, float], ...]:
    return tuple((float(k), float(k)) for k in G)


def off_diagonal_pairs() -> tuple[tuple[float, float], ...]:
    out: list[tuple[float, float]] = []
    for kappa_a in H:
        for kappa_c in H:
            if float(kappa_a) != float(kappa_c):
                out.append((float(kappa_a), float(kappa_c)))
    return tuple(out)


def budget_nested10() -> tuple[Any, ...]:
    return diagonal_pairs() + (IDENTITY,)


def budget_expand30() -> tuple[Any, ...]:
    return diagonal_pairs() + off_diagonal_pairs() + (IDENTITY,)


def candidate_records(budget: tuple[Any, ...]) -> list[dict[str, Any]]:
    rows = []
    for item in budget:
        if is_identity(item):
            rows.append(
                {
                    "key": IDENTITY,
                    "kind": IDENTITY,
                    "kappa_alpha": None,
                    "kappa_c": None,
                    "diagonal": False,
                }
            )
        else:
            kappa_a, kappa_c = item
            rows.append(
                {
                    "key": canonical_strength_pair(item),
                    "kind": "pair",
                    "kappa_alpha": float(kappa_a),
                    "kappa_c": float(kappa_c),
                    "diagonal": float(kappa_a) == float(kappa_c),
                }
            )
    return rows


def tie_tolerance(min_loss: float) -> float:
    return TIE_EPS * max(1.0, abs(float(min_loss)))


def tie_rank_key(candidate: Any) -> tuple[int, float, float, float]:
    """Descending preference after identity: (kappa_a+kappa_c, kappa_a, kappa_c).

    Returned as a sort key for max(...): identity ranks first, then larger sums.
    """
    if is_identity(candidate):
        return (1, float("inf"), float("inf"), float("inf"))
    kappa_a, kappa_c = _pair_tuple(candidate)  # type: ignore[misc]
    return (0, float(kappa_a + kappa_c), float(kappa_a), float(kappa_c))


def pick_strength(curve: list[dict[str, Any]]) -> Any:
    """Argmin of val citywide MAE with a global Lmin tolerance.

    Tolerance is 1e-10 * max(1, |Lmin|), computed once from the global minimum
    on this curve (the budget being selected). Among numerically tied
    candidates, prefer exact identity, then larger (kappa_a+kappa_c, kappa_a, kappa_c).
    This is not a practical plateau and must not be retuned from results.
    """
    if not curve:
        raise ValueError("empty selection curve")
    losses = []
    for item in curve:
        loss = item["mae"]
        if loss is None or not _finite(loss):
            raise ValueError(
                "selection curve contains a non-finite MAE; cannot silently drop a candidate"
            )
        losses.append(float(loss))
    best = min(losses)
    tol = tie_tolerance(best)
    tied = [item["candidate"] for item, loss in zip(curve, losses) if loss - best <= tol]
    if any(is_identity(candidate) for candidate in tied):
        return IDENTITY
    return max(tied, key=tie_rank_key)


def equality_tolerance(reference: float) -> float:
    return STOP_EQ_ABS + STOP_EQ_REL * abs(float(reference))


def mae_less_or_equal(new: float, reference: float) -> bool:
    return float(new) <= float(reference) + equality_tolerance(reference)


def relative_mae_improvement(new: float, reference: float) -> float:
    """(refmean - dualmean) / refmean. Rejects non-finite or non-positive references."""
    ref = float(reference)
    dual = float(new)
    if not _finite(ref) or not _finite(dual):
        raise ValueError("relative MAE improvement is undefined for a non-finite value")
    if ref <= 0.0:
        raise ValueError("relative MAE improvement is undefined for a non-positive reference mean")
    numer = ref - dual
    rel = numer / ref
    if not _finite(numer) or not _finite(rel):
        raise ValueError("relative MAE improvement overflowed or is non-finite")
    return rel


assert len(G) == 9
assert len(H) == 5
assert len(diagonal_pairs()) == 9
assert len(off_diagonal_pairs()) == 20
assert len(budget_nested10()) == 10
assert len(budget_expand30()) == 30
assert set(canonical_strength_pair(x) for x in budget_nested10()).issubset(
    set(canonical_strength_pair(x) for x in budget_expand30())
)
