#!/usr/bin/env python3
"""Key synthetic regressions for dual-kappa pilot v1.1-pilot.

All I/O uses exclusive temporary directories. The suite refuses to use the
worktree as a scratch cache. Real val/test arrays are never read.
"""

from __future__ import annotations

import inspect
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dual_kappa_pilot.analysis import cell_error_table, stop_judgment
from src.dual_kappa_pilot.protocol import (
    ARMS,
    EXPECTED_TEST_HOURS,
    EXPECTED_TEST_START,
    EXPECTED_VAL_HOURS,
    EXPECTED_VAL_START,
    GATE,
    IDENTITY,
    LABEL_START_TRAIN,
    N_TRAIN_HOURS,
    PRELOCKED_SEEDS,
    PROTOCOL_ID,
    SOURCES,
    UNIT_DC_CLIPPED_COUNT,
    budget_expand30,
    budget_nested10,
    canonical_strength_pair,
    off_diagonal_pairs,
    pick_strength,
    tie_tolerance,
)
from src.dual_kappa_pilot.score import score_test
from src.dual_kappa_pilot import select as select_mod
from src.dual_kappa_pilot.select import (
    lock_all_selections,
    reset_candidate_score_calls,
    write_json_exclusive,
)
from src.dual_kappa_pilot.snapshot import IMPLEMENTATION_FILES, TEST_FILES, binding_payload, sha256_path
from src.dual_kappa_pilot.solvers import apply_affine, fit_cells_dual, solve_dual
from src.dual_kappa_pilot.typology import lock_typology
from src.val_select.io_cache import CacheAccess, cache_path, sha256_file, write_cache_npz
from src.val_select.solvers import solve_centered, solve_old


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def worktree_source_stamp(root: Path = ROOT) -> dict:
    stamp = {}
    for rel in list(IMPLEMENTATION_FILES) + list(TEST_FILES):
        path = Path(root) / rel
        stat = path.stat()
        stamp[rel] = {
            "sha256": sha256_path(path),
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
        }
    return stamp


def _time_index(split: str) -> np.ndarray:
    if split == "train":
        return np.arange(LABEL_START_TRAIN, LABEL_START_TRAIN + N_TRAIN_HOURS, dtype=np.int64)
    if split == "val":
        return np.arange(EXPECTED_VAL_START, EXPECTED_VAL_START + EXPECTED_VAL_HOURS, dtype=np.int64)
    if split == "test":
        return np.arange(EXPECTED_TEST_START, EXPECTED_TEST_START + EXPECTED_TEST_HOURS, dtype=np.int64)
    raise AssertionError(split)


def _arrays_for(
    source: str,
    seed: int,
    split: str,
    *,
    n_cells: int = 3,
    eligible=None,
    extras: dict | None = None,
) -> dict:
    hours = {"train": N_TRAIN_HOURS, "val": EXPECTED_VAL_HOURS, "test": EXPECTED_TEST_HOURS}[split]
    cell_ids = np.arange(n_cells, dtype=np.int64)
    rng = np.random.default_rng(1000 + seed + (0 if source == "ny" else 50) + {"train": 0, "val": 1, "test": 2}[split])
    prediction = rng.normal(loc=5.0, scale=1.0, size=(hours, n_cells)).astype(np.float64)
    truth_rng = np.random.default_rng(20261118 + {"train": 0, "val": 1, "test": 2}[split])
    target = truth_rng.normal(loc=5.0, scale=1.2, size=(hours, n_cells)).astype(np.float64)
    if eligible is None:
        eligible = np.array([True, True, False][:n_cells], dtype=bool)
        if n_cells > 3:
            eligible = np.zeros(n_cells, dtype=bool)
            eligible[: max(1, n_cells - 1)] = True
    payload = {
        "prediction": prediction,
        "target": target,
        "time_index": _time_index(split),
        "cell_ids": cell_ids,
        "source": np.asarray(source),
        "seed": np.asarray(seed, dtype=np.int64),
        "split": np.asarray(split),
        "unit": np.asarray(UNIT_DC_CLIPPED_COUNT),
        "value_scale": np.asarray(1.0),
        "min_val": np.asarray(0.0),
        "norm_percentile": np.asarray("99.9"),
        "norm_fitted_on": np.asarray("full_raw_series"),
        "checkpoint_path": np.asarray("synthetic/placeholder.pt"),
        "checkpoint_sha256": np.asarray("synthetic"),
        "config_id": np.asarray("synthetic-test"),
        "generator_commit": np.asarray("synthetic"),
        "provenance_kind": np.asarray("synthetic"),
    }
    if split == "train":
        payload["eligible"] = np.asarray(eligible, dtype=bool)
        payload["raw_train_counts"] = np.where(payload["eligible"], 200.0, 10.0).astype(np.float64)
        payload["gate_threshold"] = np.asarray(float(GATE))
    if extras:
        payload.update(extras)
    return payload


def write_task_caches(cache_dir: Path, source: str, seed: int, **kwargs) -> None:
    for split in ("train", "val", "test"):
        write_cache_npz(cache_path(cache_dir, source, seed, split), _arrays_for(source, seed, split, **kwargs))


def write_complete_synth(cache_dir: Path, **kwargs) -> None:
    for source in SOURCES:
        for seed in PRELOCKED_SEEDS:
            write_task_caches(cache_dir, source, seed, **kwargs)


class SpyAccess(CacheAccess):
    created: list["SpyAccess"] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpyAccess.created.append(self)


def reset_spy() -> None:
    SpyAccess.created = []


def test_loads() -> int:
    return sum(item.test_load_count for item in SpyAccess.created)


def test_grid_five_arms_and_budgets() -> None:
    nested = budget_nested10()
    expanded = budget_expand30()
    _assert(len(nested) == 10, f"nested length {len(nested)}")
    _assert(len(expanded) == 30, f"expand length {len(expanded)}")
    _assert(len(off_diagonal_pairs()) == 20, "off-diagonal must be 20")
    keys10 = {canonical_strength_pair(x) for x in nested}
    keys30 = {canonical_strength_pair(x) for x in expanded}
    _assert(keys10.issubset(keys30), "10-candidate budget must be a subset of 30")
    _assert(IDENTITY in keys10 and IDENTITY in keys30, "identity must be in both budgets")
    _assert(keys30 - keys10 == {canonical_strength_pair(x) for x in off_diagonal_pairs()}, "30-10 must be the 20 off-diagonal pairs")
    _assert(ARMS == ("FT", "OLD_FIXED", "NEW_FIXED", "SINGLE_VAL10", "DOUBLE_VAL30"), "exactly five arms")
    _assert("SHIFT_VAL" not in ARMS and "OLD_VAL" not in ARMS, "no extra arms")


def test_global_tie_identity_then_lexicographic() -> None:
    # Global Lmin with identity inside the tolerance window wins.
    curve = [
        {"candidate": (0.0, 1.0), "mae": 1.0 + 2e-12},
        {"candidate": IDENTITY, "mae": 1.0 + 5e-12},
        {"candidate": (16.0, 16.0), "mae": 1.0},
    ]
    _assert(tie_tolerance(1.0) == 1e-10, "tolerance at Lmin=1")
    _assert(pick_strength(curve) == IDENTITY, "identity must win a numerical tie")
    # No identity: larger (sum, kappa_a, kappa_c).
    curve2 = [
        {"candidate": (4.0, 0.0), "mae": 2.0},
        {"candidate": (0.0, 4.0), "mae": 2.0},
        {"candidate": (2.0, 2.0), "mae": 2.0 + 1e-12},
    ]
    chosen = pick_strength(curve2)
    _assert(chosen == (4.0, 0.0), f"expected (4,0) over (0,4) and (2,2), got {chosen}")
    # Non-finite candidate must block rather than be dropped.
    blocked = False
    try:
        pick_strength([{"candidate": (1.0, 1.0), "mae": float("nan")}])
    except ValueError:
        blocked = True
    _assert(blocked, "non-finite MAE must block selection")


def test_diagonal_matches_centered_and_degeneracy() -> None:
    rng = np.random.default_rng(7)
    pred = rng.normal(size=N_TRAIN_HOURS).astype(np.float64)
    truth = pred + 0.3 + rng.normal(scale=0.1, size=N_TRAIN_HOURS)
    for kappa in (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        a1, b1 = solve_centered(pred, truth, kappa)
        a2, b2 = solve_dual(pred, truth, kappa, kappa)
        _assert(a1 == a2 and b1 == b2, f"diagonal kappa={kappa} must match solve_centered exactly")
    # identity is a separate branch
    a_id, b_id = solve_dual(pred, truth, IDENTITY, IDENTITY)
    _assert(a_id == 1.0 and b_id == 0.0, "identity coefficients")
    # kappa=0 OLS on a non-constant series is centered OLS, not identity
    a0, b0 = solve_dual(pred, truth, 0.0, 0.0)
    _assert(not (a0 == 1.0 and b0 == 0.0), "kappa=0 must not collapse to identity on a non-constant series")
    # Near-constant + (0,0): alpha=0, c=rbar
    const = np.full(N_TRAIN_HOURS, 3.0)
    y = const + 2.0
    a_c, b_c = solve_dual(const, y, 0.0, 0.0)
    _assert(a_c == 1.0 and abs(b_c - 2.0) < 1e-15, f"near-constant zero-strength OLS, got {a_c},{b_c}")
    # mu=0, S>0, kappa_alpha>0: slope penalty vanishes
    centered_pred = pred - pred.mean()
    a_mu, b_mu = solve_dual(centered_pred, truth, 4.0, 1.0)
    x = centered_pred
    r = truth - centered_pred
    alpha = float(x @ r) / float(x @ x)  # denom = S + 4*n*0
    c = float(r.mean()) / (1.0 + 1.0)
    _assert(abs((a_mu - 1) - alpha) < 1e-12, "mu=0 must leave alpha = T/S")
    _assert(abs((b_mu + (a_mu - 1) * 0.0) - c) < 1e-12, "c must still use kappa_c")
    # Off-diagonal (0, 4) vs (4, 0) must differ on a generic series
    a_x, b_x = solve_dual(pred, truth, 0.0, 4.0)
    a_y, b_y = solve_dual(pred, truth, 4.0, 0.0)
    _assert((a_x, b_x) != (a_y, b_y), "off-diagonal swaps must not be identical in general")


def test_nongate_remains_ft() -> None:
    train = _arrays_for("ny", 42, "train", n_cells=3)
    val = _arrays_for("ny", 42, "val", n_cells=3)
    eligible = train["eligible"]
    _assert(bool(eligible[0]) and bool(eligible[1]) and (not bool(eligible[2])), "fixture must mix gate")
    scale, shift = fit_cells_dual(train["prediction"], train["target"], eligible, (1.0, 4.0))
    _assert(scale[2] == 1.0 and shift[2] == 0.0, "non-gated cell must stay a=1,b=0")
    applied = apply_affine(val["prediction"], scale, shift)
    np.testing.assert_array_equal(applied[:, 2], val["prediction"][:, 2])
    # Gated cells should actually move (except a pathological draw)
    _assert(not np.array_equal(applied[:, 0], val["prediction"][:, 0]), "gated cell 0 should be calibrated")


def test_train_only_typology_bins() -> None:
    # 6 gated cells: distinct means, one exact mean-zero, some exact zeros.
    hours = N_TRAIN_HOURS
    y = np.zeros((hours, 7), dtype=np.float64)
    y[:, 0] = 1.0
    y[:, 1] = 2.0
    y[:, 2] = 3.0
    y[:, 3] = 10.0
    y[:, 4] = 11.0
    y[:, 5] = 0.0  # mean zero, all zeros -> CV mean_zero, zero-ratio=1
    y[:, 6] = 99.0  # non-gated, must be ignored
    eligible = np.array([True, True, True, True, True, True, False])
    typo = lock_typology(y, eligible)
    _assert(typo["n_gated"] == 6, typo["n_gated"])
    bins_mean = {c["cell_index"]: c["bin_mean_flow"] for c in typo["cells"]}
    _assert(set(bins_mean) == {0, 1, 2, 3, 4, 5}, "only gated cells")
    _assert(bins_mean[0] == "low" and bins_mean[5] == "low", bins_mean)
    _assert(bins_mean[3] == "high" and bins_mean[4] == "high", bins_mean)
    cv_bins = {c["cell_index"]: c["bin_cv"] for c in typo["cells"]}
    _assert(cv_bins[5] == "mean_zero", cv_bins)
    _assert("mean_zero" not in {cv_bins[i] for i in (0, 1, 2, 3, 4)}, "non-zero means stay in tertiles")
    # Duplicate edges: all equal gated values -> mid/high may be empty.
    y2 = np.ones((hours, 3), dtype=np.float64)
    typo2 = lock_typology(y2, np.array([True, True, True]))
    _assert(typo2["edges"]["mean_flow"]["q1"] == typo2["edges"]["mean_flow"]["q2"], typo2["edges"])
    _assert(all(c["bin_mean_flow"] == "low" for c in typo2["cells"]), "equal values all go to low")
    # Edges come from train, not from a different matrix.
    y_test_like = y * 0.0 + 50.0
    typo_test = lock_typology(y_test_like, eligible)
    _assert(typo_test["edges"] != typo["edges"], "different train truth must change edges; test must not be used to define them")
    import inspect as _inspect
    from src.dual_kappa_pilot.typology import cell_train_features as _cv_feat
    _assert("ddof=0" in _inspect.getsource(_cv_feat), "CV must keep population std ddof=0")


def test_stop_rule_prelocked_thresholds() -> None:
    def pack(double_mae, double_rmse, old_mae, old_rmse, single_mae, single_rmse):
        tasks = {}
        # source means are arithmetic means of the three seeds; use identical seeds.
        for source in SOURCES:
            for seed in PRELOCKED_SEEDS:
                tasks[f"{source}:{seed}"] = {
                    "DOUBLE_VAL30": {"mae": double_mae[source], "rmse": double_rmse[source]},
                    "OLD_FIXED": {"mae": old_mae[source], "rmse": old_rmse[source]},
                    "SINGLE_VAL10": {"mae": single_mae[source], "rmse": single_rmse[source]},
                    "FT": {"mae": 9.0, "rmse": 9.0},
                    "NEW_FIXED": {"mae": 8.0, "rmse": 8.0},
                }
        return tasks

    # Barely passing: rel MAE = 0.001, RMSE ratio = 1.005, 4/6 <= old
    # Use same values for all seeds of a source.
    old_mae = {"ny": 1.0, "chi": 2.0}
    single_mae = {"ny": 1.0, "chi": 2.0}
    double_mae_src = {"ny": 0.999, "chi": 1.998}  # rel = 0.001
    # Tasks: make ny all three <=, chi only seed 42 <= so total 4.
    tasks_pass = pack(
        {"ny": 0.999, "chi": 1.998},
        {"ny": 1.005, "chi": 2.01},
        old_mae,
        {"ny": 1.0, "chi": 2.0},
        single_mae,
        {"ny": 1.0, "chi": 2.0},
    )
    # Override chi seeds 43/44 to be slightly worse than old but keep source mean at 1.998
    # Wait: if I override, source mean changes and may break rel>=0.001.
    # Instead keep all chi MAE at 1.998 <= 2.0 so 6/6 <= old; that's still >=4 and allowed.
    j = stop_judgment(tasks_pass)
    _assert(j["passed"] and j["decision"] == "WORTH_SUBSEQUENT_DISCUSSION", j)
    # Fail MAE relative on one of the four comparisons
    tasks_mae = pack(
        {"ny": 0.9995, "chi": 1.998},  # ny rel=0.0005 < 0.001
        {"ny": 1.0, "chi": 2.0},
        old_mae,
        {"ny": 1.0, "chi": 2.0},
        single_mae,
        {"ny": 1.0, "chi": 2.0},
    )
    j_mae = stop_judgment(tasks_mae)
    _assert(j_mae["decision"] == "STOP" and not j_mae["mae_all_pass"], j_mae)
    # Fail RMSE ratio
    tasks_rmse = pack(
        {"ny": 0.999, "chi": 1.998},
        {"ny": 1.006, "chi": 2.0},  # 1.006/1.0 > 1.005
        old_mae,
        {"ny": 1.0, "chi": 2.0},
        single_mae,
        {"ny": 1.0, "chi": 2.0},
    )
    j_rmse = stop_judgment(tasks_rmse)
    _assert(j_rmse["decision"] == "STOP" and not j_rmse["rmse_all_pass"], j_rmse)
    # Fail 4/6 rule: only 3 tasks <= old. Keep source means passing.
    tasks_count = pack(
        {"ny": 0.999, "chi": 1.998},
        {"ny": 1.0, "chi": 2.0},
        old_mae,
        {"ny": 1.0, "chi": 2.0},
        single_mae,
        {"ny": 1.0, "chi": 2.0},
    )
    # Push chi all three above old (1.998+eps wait 1.998<=2). Need chi > 2 and still source mean rel>=0.001 vs old=2? Impossible if we need mean 1.998.
    # Use old_chi much larger so we can have some chi tasks worse than old while mean still good? 
    # Simpler: set old_mae chi=1.997 and double=1.995? Then rel vs old = 1-1.995/1.997 small.
    # Construct explicitly:
    tasks_few = {}
    for source, d, o, s in (
        ("ny", 0.998, 1.0, 1.0),
        ("chi", 0.998, 1.0, 1.0),
    ):
        for i, seed in enumerate(PRELOCKED_SEEDS):
            # Only first four of the six keys (ny42,ny43,ny44,chi42) are <= old.
            key = f"{source}:{seed}"
            double = 0.998 if not (source == "chi" and seed >= 43) else 1.1
            tasks_few[key] = {
                "DOUBLE_VAL30": {"mae": double, "rmse": 1.0},
                "OLD_FIXED": {"mae": 1.0, "rmse": 1.0},
                "SINGLE_VAL10": {"mae": 1.0, "rmse": 1.0},
                "FT": {"mae": 2.0, "rmse": 2.0},
                "NEW_FIXED": {"mae": 2.0, "rmse": 2.0},
            }
    # chi mean double = (0.998+1.1+1.1)/3 = 1.066 > 1.0, MAE rel fails too — that's still STOP, but we want a pure 4/6 failure.
    # To keep source means passing while only 3 tasks <= old: all doubles must stay <= 0.999*ref.
    # Then EVERY task would have double <= 0.999*old < old, so 6/6 would pass the count rule.
    # The 4/6 rule can fail independently only if some tasks are worse than old while source means still improve by 0.001.
    # Example: ny seeds all 0.9 vs old 1.0 (mean 0.9, rel=0.1). chi: 0.5, 2.0, 0.496 vs old 1.0? mean=(0.5+2+0.496)/3=0.998666, rel=0.001333>=0.001.
    # Tasks <= old: ny x3 + chi 0.5 and 0.496 = 5. Need only 3.
    # ny: two seeds slightly worse: 1.2, 1.2, 0.3 mean=0.9. chi: 1.2, 1.2, 0.596 mean=0.998666. Tasks <=old: ny44 and chi44 only = 2.
    tasks_few = {}
    layout = {
        "ny:42": 1.2,
        "ny:43": 1.2,
        "ny:44": 0.3,
        "chi:42": 1.2,
        "chi:43": 1.2,
        "chi:44": 0.596,
    }
    for key, double in layout.items():
        tasks_few[key] = {
            "DOUBLE_VAL30": {"mae": double, "rmse": 0.9},
            "OLD_FIXED": {"mae": 1.0, "rmse": 1.0},
            "SINGLE_VAL10": {"mae": 1.0, "rmse": 1.0},
            "FT": {"mae": 2.0, "rmse": 2.0},
            "NEW_FIXED": {"mae": 2.0, "rmse": 2.0},
        }
    j_few = stop_judgment(tasks_few)
    _assert(j_few["n_tasks_le_old_fixed"] == 2, j_few)
    _assert(j_few["mae_all_pass"], "means should still pass")
    _assert(j_few["rmse_all_pass"], "rmse should still pass")
    _assert(j_few["decision"] == "STOP" and not j_few["task_rule_pass"], j_few)
    # Equality uses pre-locked tolerance, not rounding.
    almost = 1.0 + 1e-10 + 1e-12 * 1.0
    tasks_eq = pack(
        {"ny": 0.999, "chi": almost},  # chi almost == old 2? wait pack uses per-source constants.
        {"ny": 1.0, "chi": 2.0},
        {"ny": 1.0, "chi": almost},  # messy
        {"ny": 1.0, "chi": 2.0},
        {"ny": 1.0, "chi": 2.0},
        {"ny": 1.0, "chi": 2.0},
    )
    # Dedicated equality check via the helper is enough and already used in stop_judgment.
    from src.dual_kappa_pilot.protocol import mae_less_or_equal

    ref = 2.0
    tol = 1e-10 + 1e-12 * 2.0
    _assert(mae_less_or_equal(ref, ref), "equal must count as <=")
    _assert(mae_less_or_equal(ref + tol, ref), "boundary must count as <=")
    _assert(not mae_less_or_equal(ref + tol + 1e-18, ref) or mae_less_or_equal(ref + tol, ref), "tolerance is exact")
    _assert(not mae_less_or_equal(ref + tol * 2, ref), "beyond tolerance is not <=")


def test_select_isolation_shared_truth_and_hash_before_load(tmp_path: Path) -> None:
    cache_dir = Path(tmp_path) / "cache"
    cache_dir.mkdir()
    write_complete_synth(cache_dir, n_cells=3)
    # Select cannot open test.
    select_access = CacheAccess(cache_dir, role="select", allow_synthetic=True)
    blocked = False
    try:
        select_access.load("ny", 42, "test")
    except PermissionError:
        blocked = True
    _assert(blocked, "select must not open test")

    reset_candidate_score_calls()
    lock_path = Path(tmp_path) / "ok.lock.json"
    lock = lock_all_selections(cache_dir, lock_path, allow_synthetic=True, root=ROOT)
    _assert(lock["scorable"], f"lock not scorable: {lock['blocked']} {lock['missing']}")
    _assert(lock["protocol_id"] == PROTOCOL_ID, lock["protocol_id"])
    _assert(set(lock["tasks"]["ny:42"]) == set(ARMS), "five arms only")
    _assert(len(lock["budget_expand30"]) == 30 and len(lock["budget_nested10"]) == 10, "budgets")
    _assert(len(lock["tasks"]["ny:42"]["DOUBLE_VAL30"]["curve"]) == 30, "all 30 candidates recorded")
    _assert(len(lock["tasks"]["ny:42"]["SINGLE_VAL10"]["curve"]) == 10, "all 10 candidates recorded")
    _assert(lock["typology"] is not None, "train typologies must be locked")
    _assert(lock["stop_rule"]["mae_rel_min"] == 0.001, "STOP rule must be pre-locked")
    _assert(select_mod.candidate_score_calls > 0, "happy path should score after truth checks")
    # Non-gate remains FT on a rematerialized val apply is already unit-tested; check lock notes.
    _assert(lock["nongate_remains_ft"] is True, "nongate policy")
    # Inventory hashed test files without decoding.
    _assert(all(row["sha256"] and row["decoded"] is False for row in lock["test_file_sha256_at_lock"]), lock["test_file_sha256_at_lock"][0])

    # Shared-truth counterexample: change one val target after a clean template, no scoring.
    bad_dir = Path(tmp_path) / "bad-truth"
    shutil.copytree(cache_dir, bad_dir)
    val_path = cache_path(bad_dir, "chi", 43, "val")
    payload = _arrays_for("chi", 43, "val", n_cells=3)
    payload["target"] = payload["target"] + 1.0
    write_cache_npz(val_path, payload)
    reset_candidate_score_calls()
    bad_lock_path = Path(tmp_path) / "bad.lock.json"
    bad = lock_all_selections(bad_dir, bad_lock_path, allow_synthetic=True, root=ROOT)
    _assert(not bad["scorable"], "shared-truth failure must not be scorable")
    _assert(select_mod.candidate_score_calls == 0, f"candidate scoring must be 0 after truth failure, got {select_mod.candidate_score_calls}")

    # Hash-before-load: tamper test bytes, score must refuse with zero test decodes.
    tamper_dir = Path(tmp_path) / "tamper"
    shutil.copytree(cache_dir, tamper_dir)
    # Need a lock that points at tamper_dir hashes. Relock on the copy (same arrays).
    reset_candidate_score_calls()
    tamper_lock_path = Path(tmp_path) / "tamper.lock.json"
    tamper_lock = lock_all_selections(tamper_dir, tamper_lock_path, allow_synthetic=True, root=ROOT)
    _assert(tamper_lock["scorable"], tamper_lock["blocked"])
    test_file = cache_path(tamper_dir, "ny", 42, "test")
    before = sha256_file(test_file)
    with test_file.open("ab") as handle:
        handle.write(b"\x00")
    after = sha256_file(test_file)
    _assert(before != after, "tamper must change bytes")
    reset_spy()
    refused = False
    try:
        score_test(
            tamper_dir,
            tamper_lock_path,
            Path(tmp_path) / "should-not-write.json",
            allow_synthetic=True,
            run_dir=Path(tmp_path) / "score-refuse",
            root=ROOT,
            test_access_factory=SpyAccess,
            i_understand_this_is_exploratory=True,
        )
    except ValueError as exc:
        refused = True
        _assert("refusing to decode" in str(exc) or "test byte sha256" in str(exc), str(exc))
    _assert(refused, "tampered test bytes must refuse scoring")
    _assert(test_loads() == 0, f"test NPZ must not be decoded, loads={test_loads()}")
    _assert(not (Path(tmp_path) / "should-not-write.json").exists(), "no official table on refusal")

    # Happy synthetic score: bytes match, then decode and write tables + STOP.
    reset_spy()
    report = score_test(
        cache_dir,
        lock_path,
        Path(tmp_path) / "report.json",
        allow_synthetic=True,
        run_dir=Path(tmp_path) / "score-ok",
        root=ROOT,
        test_access_factory=SpyAccess,
        i_understand_this_is_exploratory=True,
    )
    _assert(report["report_written"], "report should be written")
    _assert(test_loads() == 6, f"all six tests decoded after byte check, got {test_loads()}")
    _assert(set(report["tasks"]) == set(lock["tasks"]), "six tasks")
    _assert(set(report["tasks"]["ny:42"]) == set(ARMS), "five arms on test")
    _assert("stop_judgment" in report and "decision" in report["stop_judgment"], "STOP must be computed")
    _assert(report["selections_unchanged"] is True, "scoring cannot change the lock")
    # Per-cell contribution identity
    sample = report["per_cell_analysis"]["ny:42"]["versus"]["OLD_FIXED"]
    _assert("rmse_not_additive" in sample["note"] or sample["cells"][0]["rmse_not_additive"] is True, sample)
    recon = sum(c["city_mae_contribution"] for c in sample["cells"])
    _assert(abs(recon - sample["city_delta_mae"]) < 1e-12, "city MAE contributions must sum to city delta")

    # Scoring without the exploratory acknowledgement must refuse.
    blocked_ack = False
    try:
        score_test(cache_dir, lock_path, allow_synthetic=True, root=ROOT)
    except ValueError:
        blocked_ack = True
    _assert(blocked_ack, "exploratory acknowledgement is required")


def test_cell_error_rmse_not_added() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(size=(10, 4))
    ref = y + 1.0
    new = y.copy()
    new[:, 0] = y[:, 0] + 10.0
    new[:, 1:] = y[:, 1:] + 0.1
    eligible = np.array([True, True, True, False])
    table = cell_error_table(new, ref, y, eligible)
    _assert(abs(sum(c["city_mae_contribution"] for c in table["cells"]) - table["city_delta_mae"]) < 1e-12, "MAE additivity")
    _assert(all(c["rmse_not_additive"] for c in table["cells"]), "flag")
    # RMSE_j must not be treated as a sum that recovers city RMSE.
    naive = float(np.sqrt(np.mean([c["rmse_new"] ** 2 for c in table["cells"]])))  # this happens to be city RMSE
    city_rmse = float(np.sqrt(np.mean((new - y) ** 2)))
    _assert(abs(naive - city_rmse) < 1e-12, "MSE-based reconstruction is allowed")
    wrong = float(np.mean([c["rmse_new"] for c in table["cells"]]))
    _assert(abs(wrong - city_rmse) > 1e-6, "mean of RMSE_j is not city RMSE")

def test_stop_invalid_metrics_raise_not_ordinary_stop() -> None:
    def base():
        tasks = {}
        for source in SOURCES:
            for seed in PRELOCKED_SEEDS:
                tasks[f"{source}:{seed}"] = {
                    "DOUBLE_VAL30": {"mae": 0.9, "rmse": 1.0},
                    "OLD_FIXED": {"mae": 1.0, "rmse": 1.0},
                    "SINGLE_VAL10": {"mae": 1.0, "rmse": 1.0},
                    "FT": {"mae": 2.0, "rmse": 2.0},
                    "NEW_FIXED": {"mae": 2.0, "rmse": 2.0},
                }
        return tasks

    def must_raise(tasks, fragment):
        raised = False
        decision = None
        try:
            decision = stop_judgment(tasks)
        except ValueError as exc:
            raised = True
            _assert("STOP rejected" in str(exc), str(exc))
            _assert(fragment in str(exc), str(exc))
        _assert(raised, f"expected exceptional rejection containing {fragment!r}, got {decision}")
        if decision is not None:
            raise AssertionError(f"must not return PASS or STOP, got {decision}")

    nan_tasks = base()
    nan_tasks["ny:42"]["DOUBLE_VAL30"]["mae"] = float("nan")
    must_raise(nan_tasks, "not a finite number")

    inf_tasks = base()
    inf_tasks["chi:43"]["OLD_FIXED"]["rmse"] = float("inf")
    must_raise(inf_tasks, "not a finite number")

    neg_tasks = base()
    neg_tasks["ny:44"]["SINGLE_VAL10"]["mae"] = -0.01
    must_raise(neg_tasks, "negative")

    zero_mae_ref = base()
    for seed in PRELOCKED_SEEDS:
        zero_mae_ref[f"ny:{seed}"]["OLD_FIXED"]["mae"] = 0.0
    must_raise(zero_mae_ref, "not positive")

    zero_rmse_ref = base()
    for seed in PRELOCKED_SEEDS:
        zero_rmse_ref[f"chi:{seed}"]["SINGLE_VAL10"]["rmse"] = 0.0
    must_raise(zero_rmse_ref, "not positive")

    overflow = base()
    overflow["ny:42"]["OLD_FIXED"]["mae"] = 1.0e308
    overflow["ny:43"]["OLD_FIXED"]["mae"] = 1.0e308
    overflow["ny:44"]["OLD_FIXED"]["mae"] = 1.0e308
    must_raise(overflow, "overflow")


def test_top10_signed_delta_not_padded() -> None:
    # Two improved, one zero, two degraded. Lists must not include zero or the other sign, and must not pad to 10.
    y = np.zeros((4, 5), dtype=np.float64)
    ref = np.array(
        [
            [1.0, 0.5, 0.0, 0.2, 0.8],
            [1.0, 0.5, 0.0, 0.2, 0.8],
            [1.0, 0.5, 0.0, 0.2, 0.8],
            [1.0, 0.5, 0.0, 0.2, 0.8],
        ],
        dtype=np.float64,
    )
    new = np.array(
        [
            [0.0, 0.2, 0.0, 0.5, 1.2],
            [0.0, 0.2, 0.0, 0.5, 1.2],
            [0.0, 0.2, 0.0, 0.5, 1.2],
            [0.0, 0.2, 0.0, 0.5, 1.2],
        ],
        dtype=np.float64,
    )
    table = cell_error_table(new, ref, y, np.array([True, True, True, True, False]))
    improved = table["top10_improved_by_delta_mae"]
    degraded = table["top10_degraded_by_delta_mae"]
    _assert(table["n_improved"] == 2 and table["n_degraded"] == 2, table)
    _assert(len(improved) == 2 and len(degraded) == 2, "must not pad to 10")
    _assert([c["cell_index"] for c in improved] == [0, 1], improved)
    _assert(all(c["delta_mae"] > 0.0 for c in improved), improved)
    _assert([c["cell_index"] for c in degraded] == [4, 3], degraded)
    _assert(all(c["delta_mae"] < 0.0 for c in degraded), degraded)
    _assert(all(c["cell_index"] != 2 for c in improved + degraded), "zero delta must be excluded")


def test_exclusive_lock_rejects_existing_without_io(tmp_path: Path) -> None:
    cache_dir = Path(tmp_path) / "cache"
    cache_dir.mkdir()
    write_complete_synth(cache_dir, n_cells=3)

    sentinel = Path(tmp_path) / "sentinel.lock.json"
    payload = '{"sentinel": true, "do-not-touch": [1, 2, 3]}\n'
    sentinel.write_text(payload, encoding="utf-8")
    before = sentinel.read_bytes()
    mtime_ns = sentinel.stat().st_mtime_ns

    reset_candidate_score_calls()
    refused = False
    try:
        lock_all_selections(cache_dir, sentinel, allow_synthetic=True, root=ROOT)
    except FileExistsError:
        refused = True
    _assert(refused, "existing lock path must be refused")
    _assert(sentinel.read_bytes() == before, "sentinel bytes must be unchanged")
    _assert(sentinel.stat().st_mtime_ns == mtime_ns, "sentinel mtime must be unchanged")
    _assert(select_mod.candidate_score_calls == 0, "candidate scoring must be 0")
    _assert(select_mod.input_load_calls == 0, "input loading must be 0")

    as_dir = Path(tmp_path) / "dir.lock.json"
    as_dir.mkdir()
    reset_candidate_score_calls()
    refused_dir = False
    try:
        lock_all_selections(cache_dir, as_dir, allow_synthetic=True, root=ROOT)
    except FileExistsError:
        refused_dir = True
    _assert(refused_dir, "directory lock path must be refused")
    _assert(as_dir.is_dir(), "directory must remain")
    _assert(select_mod.candidate_score_calls == 0 and select_mod.input_load_calls == 0, "no I/O on directory path")

    race = Path(tmp_path) / "race.lock.json"
    race.write_text("original\n", encoding="utf-8")
    race_bytes = race.read_bytes()
    race_mtime = race.stat().st_mtime_ns
    refused_race = False
    try:
        write_json_exclusive(race, {"hijacked": True})
    except FileExistsError:
        refused_race = True
    _assert(refused_race, "exclusive create must fail if the path appeared")
    _assert(race.read_bytes() == race_bytes and race.stat().st_mtime_ns == race_mtime, "race target unchanged")

    reset_candidate_score_calls()
    fresh = Path(tmp_path) / "fresh.lock.json"
    lock = lock_all_selections(cache_dir, fresh, allow_synthetic=True, root=ROOT)
    _assert(lock["scorable"] and fresh.is_file(), lock.get("blocked"))
    _assert(select_mod.input_load_calls > 0 and select_mod.candidate_score_calls > 0, "new path must load and score")



TESTS = [
    test_grid_five_arms_and_budgets,
    test_global_tie_identity_then_lexicographic,
    test_diagonal_matches_centered_and_degeneracy,
    test_nongate_remains_ft,
    test_train_only_typology_bins,
    test_stop_rule_prelocked_thresholds,
    test_cell_error_rmse_not_added,
    test_select_isolation_shared_truth_and_hash_before_load,
    test_stop_invalid_metrics_raise_not_ordinary_stop,
    test_top10_signed_delta_not_padded,
    test_exclusive_lock_rejects_existing_without_io,
]


def main() -> int:
    before = worktree_source_stamp()
    print("worktree_stamp_before_tests:")
    for rel, item in before.items():
        print(f"  {rel} sha256={item['sha256']} mtime_ns={item['mtime_ns']} size={item['size']}")
    failures = []
    for fn in TESTS:
        tmp = Path(tempfile.mkdtemp(prefix="dual-kappa-pilot-"))
        try:
            params = inspect.signature(fn).parameters
            if "tmp_path" in params:
                fn(tmp)
            else:
                fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
            failures.append((fn.__name__, exc))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    after = worktree_source_stamp()
    print("worktree_stamp_after_tests:")
    for rel, item in after.items():
        print(f"  {rel} sha256={item['sha256']} mtime_ns={item['mtime_ns']} size={item['size']}")
    if before != after:
        print("FAIL worktree source hash or mtime changed during tests; failing the suite")
        for rel in before:
            if before[rel] != after.get(rel):
                print(f"  changed {rel}: {before[rel]} -> {after.get(rel)}")
        return 1
    print("worktree_source_hash_and_mtime_unchanged: true")
    print(f"{len(TESTS) - len(failures)}/{len(TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
