"""Synthetic unit tests for the val-selection protocol.

These tests use temporary files only. They do not read real caches, do not
score a real test set, and do not constitute a completed experiment.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.val_select.blocks import IncompleteValError, four_contiguous_blocks, rolling_rounds, weighted_block_mae
from src.val_select.bundle_adapter import BundleAdapterError, adapt_pr20_bundle, decode_utf8_sig_text, parse_claimed_checkpoint_hashes
from src.val_select.io_cache import CacheAccess, cache_path, canonical_hash, inventory_caches, sha256_file, write_cache_npz, write_json
from src.val_select.protocol import (
    CANDIDATES,
    DOCUMENTED_PREDICT_SCRIPT_FORM,
    DOCUMENTED_PREDICT_SCRIPT_REVISION,
    DOCUMENTED_TRAIN_Y_VS_CLIP_RAW_MAXABS,
    DOCUMENTED_VALUE_SCALE,
    EXPECTED_TEST_HOURS,
    EXPECTED_TEST_START,
    EXPECTED_VAL_HOURS,
    EXPECTED_VAL_START,
    GATE,
    IDENTITY,
    INPUT_PROTOCOL_ID,
    LABEL_START_TRAIN,
    NEAR_CONSTANT_REL,
    N_TRAIN_HOURS,
    PRELOCKED_SEEDS,
    PROVENANCE_DOCUMENTED,
    SOURCES,
    TIME_INDEX_ORIGIN_DOCUMENTED,
    UNIT_DC_CLIPPED_COUNT,
    pick_strength,
)
from src.val_select.schema import SchemaError
from src.val_select.score_test import score_test
from src.val_select.select import lock_all_selections, reset_candidate_score_calls, scorable_errors
import src.val_select.select as select_mod
from src.val_select.snapshot import IMPLEMENTATION_FILES, binding_payload, sha256_path
from src.val_select.solvers import fit_cells, is_near_constant, near_constant_report, solve_centered, solve_old, solve_shift


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def _pr20_solve_old(pred, truth, multiplier):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    lam = float(multiplier) * pred.size
    gram = np.array([[pred @ pred + lam, pred.sum()], [pred.sum(), pred.size + lam]], dtype=np.float64)
    rhs = np.array([pred @ truth + lam, truth.sum()], dtype=np.float64)
    scale, shift = np.linalg.solve(gram, rhs)
    return float(scale), float(shift)


def _pr20_solve_centered(pred, truth, kappa):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    n = pred.size
    center = float(pred.mean())
    residual = truth - pred
    centered = pred - center
    denom = float(centered @ centered) + float(kappa) * n * center * center
    deviation = 0.0 if denom <= 0 else float(centered @ residual) / denom
    level = float(residual.sum()) / (n * (1.0 + float(kappa)))
    return 1.0 + deviation, level - deviation * center


def test_solvers_zero_and_identity() -> None:
    rng = np.random.default_rng(0)
    pred = rng.normal(loc=8.0, scale=1.5, size=72)
    truth = 1.2 * pred + 0.7 + rng.normal(scale=0.05, size=72)
    old_a, old_b = solve_old(pred, truth, 0.0)
    new_a, new_b = solve_centered(pred, truth, 0.0)
    _assert(np.isclose(old_a, new_a) and np.isclose(old_b, new_b), "m=0 and kappa=0 must share centered OLS")
    _assert(np.allclose(old_a * pred + old_b, new_a * pred + new_b), "zero-strength predictions must match")

    const = np.full(72, 4.0)
    truth_c = const + 1.5
    a, b = solve_old(const, truth_c, 0.0)
    a2, b2 = solve_centered(const, truth_c, 0.0)
    _assert(a == 1.0 and a2 == 1.0, "constant zero-strength must set alpha=0")
    _assert(np.isclose(b, 1.5) and np.isclose(b2, 1.5), "constant zero-strength must set c=rbar")

    ident_a, ident_b = solve_old(pred, truth, IDENTITY)
    _assert(ident_a == 1.0 and ident_b == 0.0, "identity must be exact")
    _assert(solve_centered(pred, truth, IDENTITY) == (1.0, 0.0), "centered identity must be exact")
    _assert(solve_shift(pred, truth, IDENTITY) == (1.0, 0.0), "shift identity must be exact")
    _assert(solve_shift(pred, truth, 0.0)[1] == float((truth - pred).mean()), "kappa=0 shift is rbar")


def test_pr20_agreement_finite_strength() -> None:
    rng = np.random.default_rng(1)
    pred = rng.normal(loc=5.0, scale=2.0, size=72)
    truth = 0.8 * pred + 1.1 + rng.normal(scale=0.2, size=72)
    for strength in (0.125, 0.5, 1.0, 4.0, 16.0):
        got = solve_old(pred, truth, strength)
        ref = _pr20_solve_old(pred, truth, strength)
        _assert(np.allclose(got, ref, rtol=0, atol=1e-12), f"old solver must match PR20 at m={strength}")
        got_c = solve_centered(pred, truth, strength)
        ref_c = _pr20_solve_centered(pred, truth, strength)
        _assert(np.allclose(got_c, ref_c, rtol=0, atol=1e-12), f"centered solver must match PR20 at kappa={strength}")


def test_zero_strength_high_mean_low_variance_and_boundaries() -> None:
    # High mean, low but non-zero fluctuation, above the frozen near-constant threshold.
    base = 1.0e6
    pred = base + np.linspace(0.0, 1.0e-6, 72)
    span = float(np.ptp(pred))
    thresh = NEAR_CONSTANT_REL * max(1.0, float(np.max(np.abs(pred))))
    _assert(span > thresh, "fixture must remain above the frozen near-constant threshold")
    truth = pred + 3.5 + 0.25 * (pred - pred.mean())
    old_ab = solve_old(pred, truth, 0.0)
    new_ab = solve_centered(pred, truth, 0.0)
    _assert(old_ab == new_ab, "m=0 must use the identical centered OLS path as kappa=0")
    _assert(np.allclose(old_ab[0] * pred + old_ab[1], new_ab[0] * pred + new_ab[1], rtol=0, atol=0), "predictions must be identical")

    zeros = np.zeros(72)
    truth_z = zeros + 2.25
    a, b = solve_old(zeros, truth_z, 0.0)
    a2, b2 = solve_centered(zeros, truth_z, 0.0)
    _assert((a, b) == (a2, b2) == (1.0, 2.25), "zero-valued constant boundary must be alpha=0, c=rbar")

    huge = np.full(72, 1.0e6)
    truth_h = huge - 8.0
    a, b = solve_old(huge, truth_h, 0.0)
    a2, b2 = solve_centered(huge, truth_h, 0.0)
    _assert((a, b) == (a2, b2) == (1.0, -8.0), "high-mean exact constant must be alpha=0, c=rbar")
    _assert(is_near_constant(zeros) and is_near_constant(huge), "exact constants must trigger the shared near-constant test")
    _assert(not is_near_constant(pred), "high-mean low fluctuation above the frozen threshold must not trigger")
    report = near_constant_report(np.stack([zeros, pred], axis=1), np.array([True, True]))
    _assert(report["triggered_cells"] == [0], "near-constant triggers must be recorded per cell")
    _assert(report["near_constant_rel"] == NEAR_CONSTANT_REL, "both solvers share the frozen threshold")


def test_pick_strength_ties() -> None:
    curve = [
        {"strength": 0.5, "mae": 1.0},
        {"strength": IDENTITY, "mae": 1.0 + 1e-16},
        {"strength": 16.0, "mae": 1.0 + 2e-16},
    ]
    _assert(pick_strength(curve) == IDENTITY, "numerical ties must prefer identity")
    curve2 = [{"strength": 1.0, "mae": 2.0}, {"strength": 8.0, "mae": 2.0 + 1e-16}]
    _assert(pick_strength(curve2) == 8.0, "finite ties must prefer the larger strength")
    min_loss = 3.0
    worse = min_loss + 10 * (1e-10 * max(1.0, abs(min_loss)))
    curve3 = [{"strength": 1.0, "mae": min_loss}, {"strength": 16.0, "mae": worse}]
    _assert(pick_strength(curve3) == 1.0, "tolerance must not act as a performance plateau")


def test_nonfinite_or_short_series_blocks() -> None:
    pred = np.linspace(1.0, 2.0, 72)
    truth = pred.copy()
    truth[3] = np.nan
    failed = False
    try:
        solve_centered(pred, truth, 1.0)
    except ValueError:
        failed = True
    _assert(failed, "NaN must block the solver")

    # Legal (60, 2) arrays and matching eligible mask; only the hour count is wrong.
    short = np.linspace(1.0, 2.0, 120, dtype=np.float64).reshape(60, 2)
    truth_s = short + 0.1
    eligible = np.array([True, True])
    _assert(short.shape == (60, 2), "fixture shape must be a legal (60, 2)")
    failed = False
    try:
        fit_cells(short, truth_s, eligible, "old", 1.0)
    except ValueError:
        failed = True
    _assert(failed, "missing hours must block fitting even when other fields are legal")


def test_blocks_use_timestamps_and_refuse_gaps() -> None:
    idx = np.arange(EXPECTED_VAL_START, EXPECTED_VAL_START + EXPECTED_VAL_HOURS)
    blocks = four_contiguous_blocks(idx)
    _assert(len(blocks) == 4, "need 4 blocks")
    _assert(all(b["n_effective"] == 360 for b in blocks), "complete 1440h yields 360 per block as a consequence, not padding")
    _assert(blocks[0]["t_start"] == EXPECTED_VAL_START, "first block must start at val start")
    _assert(blocks[-1]["t_end"] == EXPECTED_VAL_START + EXPECTED_VAL_HOURS, "last block must end at val end")
    failed = False
    try:
        four_contiguous_blocks(idx[:-10])
    except IncompleteValError:
        failed = True
    _assert(failed, "incomplete val must not be silently split")
    gapped = np.concatenate([idx[:100], idx[110:]])
    failed = False
    try:
        four_contiguous_blocks(gapped)
    except IncompleteValError:
        failed = True
    _assert(failed, "gapped timestamps must be refused")
    rounds = rolling_rounds(blocks)
    _assert([r["prefix_blocks"] for r in rounds] == [[0], [0, 1], [0, 1, 2]], "expanding prefixes")
    parts = [(1.0, 10), (2.0, 30)]
    _assert(weighted_block_mae(parts) == (1.0 * 10 + 2.0 * 30) / 40, "weighted by observation count")


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
    n_cells: int = 2,
    cell_ids=None,
    unit: str = UNIT_DC_CLIPPED_COUNT,
    time_index=None,
    eligible=None,
    raw_counts=None,
    provenance_kind: str = "synthetic",
    omit: tuple[str, ...] = (),
    extras: dict | None = None,
) -> dict:
    hours = {"train": N_TRAIN_HOURS, "val": EXPECTED_VAL_HOURS, "test": EXPECTED_TEST_HOURS}[split]
    if cell_ids is None:
        cell_ids = np.arange(n_cells, dtype=np.int64)
    rng = np.random.default_rng(1000 + seed + (0 if source == "ny" else 50) + {"train": 0, "val": 1, "test": 2}[split])
    prediction = rng.normal(loc=5.0, scale=1.0, size=(hours, n_cells)).astype(np.float64)
    # Shared DC truth: same y for every source×seed; only predictions differ.
    truth_rng = np.random.default_rng(20261117 + {"train": 0, "val": 1, "test": 2}[split])
    target = truth_rng.normal(loc=5.0, scale=1.2, size=(hours, n_cells)).astype(np.float64)
    if cell_ids is None:
        cell_ids = np.arange(n_cells, dtype=np.int64)
    payload = {
        "prediction": prediction,
        "target": target,
        "time_index": _time_index(split) if time_index is None else np.asarray(time_index),
        "cell_ids": np.asarray(cell_ids),
        "source": np.asarray(source),
        "seed": np.asarray(seed, dtype=np.int64),
        "split": np.asarray(split),
        "unit": np.asarray(unit),
        "value_scale": np.asarray(1.0),
        "min_val": np.asarray(0.0),
        "norm_percentile": np.asarray("99.9"),
        "norm_fitted_on": np.asarray("full_raw_series"),
        "checkpoint_path": np.asarray("synthetic/placeholder.pt"),
        "checkpoint_sha256": np.asarray("synthetic"),
        "config_id": np.asarray("synthetic-test"),
        "generator_commit": np.asarray("synthetic"),
        "provenance_kind": np.asarray(provenance_kind),
    }
    if split == "train":
        if eligible is None:
            eligible = np.array([True] * n_cells)
        if raw_counts is None:
            raw_counts = np.where(eligible, 200.0, 10.0).astype(np.float64)
        payload["eligible"] = np.asarray(eligible, dtype=bool)
        payload["raw_train_counts"] = np.asarray(raw_counts, dtype=np.float64)
        payload["gate_threshold"] = np.asarray(float(GATE))
    if extras:
        payload.update(extras)
    for key in omit:
        payload.pop(key, None)
    return payload


def write_task_caches(cache_dir: Path, source: str, seed: int, **kwargs) -> None:
    for split in ("train", "val", "test"):
        write_cache_npz(cache_path(cache_dir, source, seed, split), _arrays_for(source, seed, split, **kwargs))


def write_complete_synth(cache_dir: Path, **kwargs) -> None:
    for source in SOURCES:
        for seed in PRELOCKED_SEEDS:
            write_task_caches(cache_dir, source, seed, **kwargs)


def worktree_source_stamp(root: Path = ROOT) -> dict:
    stamp = {}
    for rel in list(IMPLEMENTATION_FILES) + ["tests/test_val_select_kappa.py"]:
        path = Path(root) / rel
        stat = path.stat()
        stamp[rel] = {
            "sha256": sha256_path(path),
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
        }
    return stamp


def _copy_implementation_mirror(dest: Path) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for rel in IMPLEMENTATION_FILES:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    return dest


def rewrite_npz(path: Path, **updates) -> None:
    from src.val_select.io_cache import decode_npz

    data = decode_npz(path)
    data.update(updates)
    write_cache_npz(path, data)


class SpyAccess(CacheAccess):
    created: list["SpyAccess"] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpyAccess.created.append(self)


def reset_spy() -> None:
    SpyAccess.created = []


def test_loads() -> int:
    return sum(item.test_load_count for item in SpyAccess.created)


def test_cache_access_select_cannot_open_test(tmp_path: Path | None = None) -> None:
    cache_dir = Path(tmp_path or ROOT / ".tmp-test-select")
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_task_caches(cache_dir, "ny", 42)
    select = CacheAccess(cache_dir, role="select", allow_synthetic=True)
    select.load("ny", 42, "train")
    select.load("ny", 42, "val")
    failed = False
    try:
        select.load("ny", 42, "test")
    except PermissionError:
        failed = True
    _assert(failed, "select role must not open test")
    _assert(select.test_load_count == 0, "select role must not increment test loads")


def test_inventory_does_not_unpack(tmp_path: Path | None = None) -> None:
    cache_dir = Path(tmp_path or ROOT / ".tmp-test-inv")
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_complete_synth(cache_dir)
    import src.val_select.io_cache as io_mod

    calls = {"n": 0}
    original = io_mod.decode_npz

    def boom(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("inventory must not unpack arrays")

    io_mod.decode_npz = boom  # type: ignore[assignment]
    try:
        report = inventory_caches(cache_dir)
    finally:
        io_mod.decode_npz = original
    _assert(calls["n"] == 0, "inventory must not decode npz")
    _assert(report["unpacked_arrays"] is False, "inventory must declare it did not unpack")
    _assert(report["n_present"] == 18, "six tasks times three splits")
    test_items = [item for item in report["files"] if item["split"] == "test"]
    _assert(all("sha256" in item and "size" in item for item in test_items), "test inventory is stat/hash only")


def test_schema_rejects_wrong_seed_columns_unit_hours_gate(tmp_path: Path | None = None) -> None:
    cache_dir = Path(tmp_path or ROOT / ".tmp-test-schema")
    cache_dir.mkdir(parents=True, exist_ok=True)
    access = CacheAccess(cache_dir, role="select", allow_synthetic=True)

    write_cache_npz(
        cache_path(cache_dir, "ny", 42, "train"),
        _arrays_for("ny", 43, "train"),  # declared seed 43, requested 42
    )
    failed = False
    try:
        access.load("ny", 42, "train")
    except SchemaError:
        failed = True
    _assert(failed, "wrong declared seed must be refused")

    write_cache_npz(
        cache_path(cache_dir, "ny", 42, "train"),
        _arrays_for("ny", 42, "train", unit="normalized_unit"),
    )
    failed = False
    try:
        access.load("ny", 42, "train")
    except SchemaError:
        failed = True
    _assert(failed, "wrong unit must be refused")

    bad_hours = np.arange(LABEL_START_TRAIN, LABEL_START_TRAIN + 60, dtype=np.int64)
    payload = _arrays_for("ny", 42, "train", time_index=bad_hours)
    payload["prediction"] = payload["prediction"][:60]
    payload["target"] = payload["target"][:60]
    write_cache_npz(cache_path(cache_dir, "ny", 42, "train"), payload)
    failed = False
    try:
        access.load("ny", 42, "train")
    except SchemaError:
        failed = True
    _assert(failed, "wrong hour index / length must be refused")

    write_cache_npz(
        cache_path(cache_dir, "ny", 42, "train"),
        _arrays_for("ny", 42, "train", eligible=np.array([True, True]), raw_counts=np.array([200.0, 10.0])),
    )
    failed = False
    try:
        access.load("ny", 42, "train")
    except SchemaError:
        failed = True
    _assert(failed, "eligible that disagrees with recomputed raw-count gate must be refused")

    # Column-order mismatch against a reference already accepted from another task.
    write_complete_synth(cache_dir, cell_ids=np.array([10, 20]))
    access2 = CacheAccess(cache_dir, role="select", allow_synthetic=True, expected_cell_ids=np.array([10, 20]))
    access2.load("ny", 42, "train")
    write_cache_npz(
        cache_path(cache_dir, "chi", 42, "train"),
        _arrays_for("chi", 42, "train", cell_ids=np.array([20, 10])),
    )
    failed = False
    try:
        access2.load("chi", 42, "train")
    except SchemaError:
        failed = True
    _assert(failed, "permuted cell_ids must be refused")

    # Legacy missing metadata cannot be reconstructed.
    write_cache_npz(
        cache_path(cache_dir, "ny", 42, "train"),
        _arrays_for("ny", 42, "train", omit=("time_index", "source", "seed", "checkpoint_sha256")),
    )
    failed = False
    try:
        access.load("ny", 42, "train")
    except SchemaError as exc:
        failed = True
        _assert("do not guess" in str(exc) or "missing required" in str(exc), "legacy missing fields must demand proof")
    _assert(failed, "legacy caches missing provenance must be refused")


def test_lock_and_score_isolation_and_overwrite(tmp_path: Path | None = None) -> None:
    cache_dir = Path(tmp_path or ROOT / ".tmp-test-lock")
    if cache_dir.exists():
        import shutil

        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    write_complete_synth(cache_dir)
    lock_path = cache_dir / "selection.lock.json"
    lock = lock_all_selections(cache_dir, lock_path, allow_synthetic=True)
    _assert(lock["scorable"] is True, "complete synthetic set must be scorable")
    _assert(lock_path.is_file(), "scorable lock must be written")
    _assert("binding" in lock and "plan" in lock["binding"], "lock must bind the plan")
    _assert(lock["binding"]["implementation"]["fingerprint"], "lock must bind implementation")
    key = next(iter(lock["stage2"]))
    arm = "NEW_VAL"
    _assert(
        "fixed_weighted_mae_last_three" in lock["stage2"][key]["rolling"][arm],
        "fixed baselines must have observation-weighted last-three summaries",
    )
    _assert("near_constant" in lock["input_hashes"][0], "near-constant triggers must be recorded in the lock")

    reset_spy()
    run_dir = cache_dir / "run-a"
    report_path = cache_dir / "report.json"
    report = score_test(
        cache_dir,
        lock_path,
        report_path,
        allow_synthetic=True,
        run_dir=run_dir,
        command="synthetic-score-1",
        test_access_factory=SpyAccess,
        i_understand_this_is_exploratory=True,
    )
    _assert(test_loads() > 0, "a successful synthetic score may open test after verification")
    _assert(report["lock_unchanged"] is True, "lock must not be mutated")
    _assert((run_dir / "run_audit.json").is_file(), "exclusive run dir must store audit")
    locked = json.loads(lock_path.read_text())
    _assert("test_touched" not in locked, "lock must not claim a persistent test_touched flag")

    failed = False
    try:
        score_test(
            cache_dir,
            lock_path,
            report_path,
            allow_synthetic=True,
            run_dir=cache_dir / "run-b",
            command="synthetic-score-overwrite",
            i_understand_this_is_exploratory=True,
        )
    except FileExistsError:
        failed = True
    _assert(failed, "repeat scoring must not overwrite an existing report")

    failed = False
    try:
        score_test(
            cache_dir,
            lock_path,
            cache_dir / "report-2.json",
            allow_synthetic=True,
            run_dir=run_dir,
            command="synthetic-score-same-dir",
            i_understand_this_is_exploratory=True,
        )
    except FileExistsError:
        failed = True
    _assert(failed, "repeat scoring must not reuse an existing run directory")


def _score_should_refuse(cache_dir: Path, lock_path: Path, *, run_name: str, root=None) -> None:
    reset_spy()
    failed = False
    try:
        score_test(
            cache_dir,
            lock_path,
            cache_dir / f"{run_name}-report.json",
            allow_synthetic=True,
            run_dir=cache_dir / run_name,
            command=f"synthetic-refuse-{run_name}",
            test_access_factory=SpyAccess,
            i_understand_this_is_exploratory=True,
            root=root,
        )
    except (ValueError, FileExistsError, FileNotFoundError):
        failed = True
    _assert(failed, f"{run_name} must be refused")
    _assert(test_loads() == 0, f"{run_name}: test loader must be called 0 times")
    _assert(len(SpyAccess.created) == 0 or test_loads() == 0, f"{run_name}: spy must not have opened test")


def test_b1_tamper_train_val_plan_solver_config(tmp_path: Path | None = None) -> None:
    cache_dir = Path(tmp_path or ROOT / ".tmp-test-b1")
    if cache_dir.exists():
        import shutil

        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    write_complete_synth(cache_dir)
    lock_path = cache_dir / "selection.lock.json"
    lock_all_selections(cache_dir, lock_path, allow_synthetic=True)

    # Tamper train bytes, not the lock JSON.
    train_path = cache_path(cache_dir, "ny", 42, "train")
    original_train = train_path.read_bytes()
    train_path.write_bytes(original_train + b"\x00")
    _score_should_refuse(cache_dir, lock_path, run_name="tamper-train")
    train_path.write_bytes(original_train)

    val_path = cache_path(cache_dir, "chi", 43, "val")
    original_val = val_path.read_bytes()
    val_path.write_bytes(original_val + b"\x00")
    _score_should_refuse(cache_dir, lock_path, run_name="tamper-val")
    val_path.write_bytes(original_val)

    # Mutate only temporary mirrors. Use two independent copies so a plan
    # tamper cannot cause the solver case to fail for the wrong reason.
    original_binding = binding_payload()
    original_plan = original_binding["plan"]["sha256"]
    original_fp = original_binding["implementation"]["fingerprint"]
    original_solver = sha256_path(ROOT / "src" / "val_select" / "solvers.py")

    mirror_plan = _copy_implementation_mirror(cache_dir / "mirror-plan")
    (mirror_plan / "docs" / "VAL_SELECT_KAPPA_PLAN.md").write_text(
        (mirror_plan / "docs" / "VAL_SELECT_KAPPA_PLAN.md").read_text(encoding="utf-8") + "\n<!-- tamper -->\n",
        encoding="utf-8",
    )
    plan_now = binding_payload(root=mirror_plan)
    _assert(plan_now["plan"]["sha256"] != original_plan, "plan tamper must change the plan hash")
    _assert(plan_now["implementation"]["fingerprint"] != original_fp, "plan tamper must change the implementation fingerprint")
    _assert(
        sha256_path(mirror_plan / "src" / "val_select" / "solvers.py") == original_solver,
        "plan mirror must leave solvers.py unchanged",
    )
    _score_should_refuse(cache_dir, lock_path, run_name="tamper-plan", root=mirror_plan)

    mirror_solver = _copy_implementation_mirror(cache_dir / "mirror-solver")
    (mirror_solver / "src" / "val_select" / "solvers.py").write_text(
        (mirror_solver / "src" / "val_select" / "solvers.py").read_text(encoding="utf-8") + "\n# tamper\n",
        encoding="utf-8",
    )
    solver_now = binding_payload(root=mirror_solver)
    _assert(solver_now["plan"]["sha256"] == original_plan, "solver mirror must leave the plan hash unchanged")
    _assert(solver_now["implementation"]["fingerprint"] != original_fp, "solver tamper must change the implementation fingerprint")
    _assert(
        sha256_path(mirror_solver / "docs" / "VAL_SELECT_KAPPA_PLAN.md") == sha256_path(ROOT / "docs" / "VAL_SELECT_KAPPA_PLAN.md"),
        "solver mirror must leave the plan file unchanged",
    )
    _score_should_refuse(cache_dir, lock_path, run_name="tamper-solver", root=mirror_solver)

    import src.val_select.protocol as proto
    import src.val_select.snapshot as snap

    original_gate = proto.GATE
    try:
        proto.GATE = 999
        _score_should_refuse(cache_dir, lock_path, run_name="tamper-config")
    finally:
        proto.GATE = original_gate
        # snapshot.config_snapshot reads proto.GATE at call time; restore done.


def test_b2_missing_and_incomplete_locks(tmp_path: Path | None = None) -> None:
    cache_dir = Path(tmp_path or ROOT / ".tmp-test-b2")
    if cache_dir.exists():
        import shutil

        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    write_complete_synth(cache_dir)
    cache_path(cache_dir, "ny", 44, "train").unlink()
    lock_path = cache_dir / "selection.lock.json"
    lock = lock_all_selections(cache_dir, lock_path, allow_synthetic=True)
    _assert(lock["scorable"] is False, "missing task must not produce a scorable lock")
    _assert(not lock_path.exists(), "scorable lock path must not be written when incomplete")
    blocked_path = lock_path.with_name("selection.lock.blocked.json")
    _assert(blocked_path.is_file(), "diagnostic blocked lock may be written")
    _score_should_refuse(cache_dir, blocked_path, run_name="missing-task")

    # Restore the file and make a complete lock, then surgically drop structure and rehash.
    write_task_caches(cache_dir, "ny", 44)
    good_path = cache_dir / "good.lock.json"
    good = lock_all_selections(cache_dir, good_path, allow_synthetic=True)
    _assert(good["scorable"] is True, "restored set must lock")

    def sabotage(name: str, editor) -> Path:
        payload = json.loads(good_path.read_text(encoding="utf-8"))
        editor(payload)
        payload["lock_hash"] = canonical_hash({k: v for k, v in payload.items() if k != "lock_hash"})
        path = cache_dir / f"{name}.lock.json"
        write_json(path, payload)
        return path

    empty1 = sabotage("empty-stage1", lambda p: p.update(stage1={}))
    _score_should_refuse(cache_dir, empty1, run_name="empty-stage1")
    missing_stage = sabotage("missing-stage2", lambda p: p.pop("stage2"))
    _score_should_refuse(cache_dir, missing_stage, run_name="missing-stage2")
    missing_task = sabotage("drop-task", lambda p: p["stage1"].pop("chi:42"))
    _score_should_refuse(cache_dir, missing_task, run_name="drop-task")

    def empty_stage2_rounds(payload):
        for item in payload["stage2"].values():
            for arm in list(item.get("rolling") or {}):
                item["rolling"][arm]["rounds"] = []

    empty_rounds = sabotage("empty-stage2-rounds", empty_stage2_rounds)
    _score_should_refuse(cache_dir, empty_rounds, run_name="empty-stage2-rounds")

    def chosen_outside(payload):
        payload["stage1"]["ny:42"]["NEW_VAL"]["chosen"] = 3.0

    outside = sabotage("chosen-outside-grid", chosen_outside)
    _score_should_refuse(cache_dir, outside, run_name="chosen-outside-grid")

    failed = False
    try:
        lock_all_selections(cache_dir, cache_dir / "seed-drop.lock.json", seeds=(42,), allow_synthetic=True)
    except ValueError as exc:
        failed = True
        _assert("seed_freeze_id" in str(exc) and "not dropped automatically" in str(exc), exc.args[0])
    _assert(failed, "seed-set change without a new freeze version must be refused")


def test_filename_is_not_identity_proof(tmp_path: Path | None = None) -> None:
    cache_dir = Path(tmp_path or ROOT / ".tmp-test-name")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # File is named ny_seed42_train.npz but declares chi/43.
    write_cache_npz(cache_path(cache_dir, "ny", 42, "train"), _arrays_for("chi", 43, "train"))
    access = CacheAccess(cache_dir, role="select", allow_synthetic=True)
    failed = False
    try:
        access.load("ny", 42, "train")
    except SchemaError:
        failed = True
    _assert(failed, "filename must not be accepted as source/seed proof")


def test_sidecar_conflicts_and_equivalent_dual(tmp_path: Path | None = None) -> None:
    cache_dir = Path(tmp_path or ROOT / ".tmp-test-sidecar")
    cache_dir.mkdir(parents=True, exist_ok=True)
    access = CacheAccess(cache_dir, role="select", allow_synthetic=True)
    path = cache_path(cache_dir, "ny", 42, "train")
    payload = _arrays_for("ny", 42, "train")
    # Legal dual representation: payload arrays plus equivalent JSON sidecar types.
    sidecar = {
        "source": "ny",
        "seed": 42,
        "split": "train",
        "unit": UNIT_DC_CLIPPED_COUNT,
        "norm_percentile": 99.9,
        "eligible": [1, 1],
    }
    write_cache_npz(path, payload, sidecar=sidecar)
    loaded = access.load("ny", 42, "train")
    _assert(loaded["source"] == "ny" and loaded["seed"] == 42, "equivalent dual representation must load")

    write_cache_npz(path, _arrays_for("ny", 42, "train"), sidecar={"source": "chi"})
    failed = False
    try:
        access.load("ny", 42, "train")
    except SchemaError as exc:
        failed = True
        _assert("conflict" in str(exc), str(exc))
    _assert(failed, "npz ny / sidecar chi must be refused")


def _legal_recorded_train() -> dict:
    payload = _arrays_for("ny", 42, "train", provenance_kind="recorded")
    payload["checkpoint_sha256"] = np.asarray("ab" * 32)
    payload["generator_commit"] = np.asarray("a" * 40)
    payload["checkpoint_path"] = np.asarray("weights/model.pt")
    payload["config_id"] = np.asarray("finetune_ny2dc")
    payload["norm_fitted_on"] = np.asarray("full_raw_series")
    return payload


def test_schema_parameterized_rejects(tmp_path: Path | None = None) -> None:
    base = Path(tmp_path or ROOT / ".tmp-test-schema-param")
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)

    # Each case gets a fresh directory and file, starting from a legal baseline
    # and changing only one field. A leftover sidecar must not mask later cases.
    cases = []

    hash_synth = _legal_recorded_train()
    hash_synth["checkpoint_sha256"] = np.asarray("synthetic")
    cases.append(("hash-synthetic", hash_synth, None, "checkpoint_sha256"))

    hash_null = {k: v for k, v in _legal_recorded_train().items() if k != "checkpoint_sha256"}
    cases.append(("hash-null", hash_null, {"checkpoint_sha256": None}, "checkpoint_sha256"))

    cases.append(("seed-42.9", _arrays_for("ny", 42, "train", extras={"seed": np.asarray(42.9)}), None, "seed"))

    frac_time = _time_index("train").astype(np.float64)
    frac_time[0] = frac_time[0] + 0.5
    cases.append(("time-fractional", _arrays_for("ny", 42, "train", time_index=frac_time), None, "time_index"))

    cases.append(("raw-nan", _arrays_for("ny", 42, "train", raw_counts=np.array([200.0, np.nan])), None, "raw_train_counts"))
    cases.append(("raw-negative", _arrays_for("ny", 42, "train", raw_counts=np.array([200.0, -1.0])), None, "raw_train_counts"))

    for name, arrays, sidecar, field in cases:
        case_dir = base / name
        case_dir.mkdir()
        path = cache_path(case_dir, "ny", 42, "train")
        write_cache_npz(path, arrays, sidecar=sidecar)
        access = CacheAccess(case_dir, role="select", allow_synthetic=True)
        failed = False
        message = ""
        try:
            access.load("ny", 42, "train")
        except (SchemaError, ValueError) as exc:
            failed = True
            message = str(exc)
        _assert(failed, f"{name} must be refused")
        _assert(field in message, f"{name} must name {field} in the error, got {message!r}")


def test_val_norm_change_and_shared_truth(tmp_path: Path | None = None) -> None:
    cache_dir = Path(tmp_path or ROOT / ".tmp-test-truth")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    write_complete_synth(cache_dir)

    # Val of one task changes scale/norm; lock and score must refuse with test loads = 0.
    val_path = cache_path(cache_dir, "ny", 42, "val")
    payload = _arrays_for("ny", 42, "val", extras={"value_scale": np.asarray(2.0)})
    write_cache_npz(val_path, payload)
    lock_path = cache_dir / "scale.lock.json"
    failed = False
    try:
        lock_all_selections(cache_dir, lock_path, allow_synthetic=True)
    except (SchemaError, ValueError):
        failed = True
    _assert(failed, "val-only scale change must be refused at selection")
    _assert(not lock_path.exists(), "no scorable lock after val scale change")
    # Even a blocked lock, if written, must not open test.
    blocked = lock_path.with_name("scale.lock.blocked.json")
    if blocked.is_file():
        _score_should_refuse(cache_dir, blocked, run_name="val-scale-blocked")
    reset_spy()
    _assert(test_loads() == 0, "val scale change must not open test")

    # Restore a consistent set, lock, then change only CHI 43 val labels.
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    write_complete_synth(cache_dir)
    good_lock = cache_dir / "good-truth.lock.json"
    lock = lock_all_selections(cache_dir, good_lock, allow_synthetic=True)
    _assert(lock["scorable"] is True, "shared-y fixture must lock")

    chi_val = cache_path(cache_dir, "chi", 43, "val")
    from src.val_select.io_cache import decode_npz
    import src.val_select.select as select_mod

    data = decode_npz(chi_val)
    data["target"] = np.asarray(data["target"], dtype=np.float64) + 1.0
    write_cache_npz(chi_val, data)
    failed = False
    select_mod.reset_candidate_score_calls()
    changed_lock = cache_dir / "changed-val-y.lock.json"
    try:
        lock_all_selections(cache_dir, changed_lock, allow_synthetic=True)
    except (SchemaError, ValueError):
        failed = True
    _assert(failed, "changing only CHI 43 val labels must be refused at selection")
    _assert(select_mod.candidate_score_calls == 0, "shared-label change must not call candidate scoring")
    _assert(not changed_lock.exists(), "shared-label change must not write a scorable lock")
    blocked = changed_lock.with_name("changed-val-y.lock.blocked.json")
    _assert(not blocked.is_file() or not json.loads(blocked.read_text()).get("scorable"), "no valid lock after shared-label change")
    _score_should_refuse(cache_dir, good_lock, run_name="chi43-val-y")

    # Restore val, change only one test y: select/lock remains valid, score precheck refuses table write.
    write_task_caches(cache_dir, "chi", 43)
    # good_lock still matches the restored train/val hashes
    chi_test = cache_path(cache_dir, "chi", 43, "test")
    data = decode_npz(chi_test)
    data["target"] = np.asarray(data["target"], dtype=np.float64) + 1.0
    write_cache_npz(chi_test, data)
    reset_spy()
    failed = False
    try:
        score_test(
            cache_dir,
            good_lock,
            cache_dir / "changed-test-y-report.json",
            allow_synthetic=True,
            run_dir=cache_dir / "changed-test-y",
            command="synthetic-changed-test-y",
            test_access_factory=SpyAccess,
            i_understand_this_is_exploratory=True,
        )
    except (SchemaError, ValueError):
        failed = True
    _assert(failed, "changing only a test label must be refused at score precheck")
    _assert(not (cache_dir / "changed-test-y-report.json").exists(), "failed precheck must not write a score table")


def test_different_seeds_may_have_different_weight_hashes(tmp_path: Path | None = None) -> None:
    cache_dir = Path(tmp_path or ROOT / ".tmp-test-seed-hash")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    write_complete_synth(cache_dir)
    # Distinct checkpoint hashes per seed, identical within a task and identical DC labels.
    for source in SOURCES:
        for seed in PRELOCKED_SEEDS:
            digest = f"{seed:064x}"
            for split in ("train", "val", "test"):
                path = cache_path(cache_dir, source, seed, split)
                payload = _arrays_for(source, seed, split, extras={"checkpoint_sha256": np.asarray(digest)})
                write_cache_npz(path, payload)
    lock = lock_all_selections(cache_dir, cache_dir / "seed-hash.lock.json", allow_synthetic=True)
    _assert(lock["scorable"] is True, "different seeds must not be required to share a weight hash")



import hashlib
import zipfile


def _claimed_hash(name: str) -> str:
    return hashlib.sha256(f"claimed:{name}".encode("utf-8")).hexdigest()


def _write_native_npz(path: Path, *, prediction, target, value_scale, eligible=None, rows=None, cols=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prediction": np.asarray(prediction, dtype=np.float64),
        "target": np.asarray(target, dtype=np.float64),
        "value_scale": np.asarray(value_scale, dtype=np.float64),
    }
    if eligible is not None:
        payload["eligible"] = np.asarray(eligible, dtype=bool)
        payload["rows"] = np.asarray(rows)
        payload["cols"] = np.asarray(cols)
    np.savez_compressed(path, **payload)


def _bundle_readme() -> str:
    return (
        "PR20 hourly prediction bundle (synthetic fixture for adapter tests).\n"
        "train 第 t 行：标签小时 5832 + t，即 [5832, 5904)\n"
        "val 第 t 行：标签小时 5904 + t，即 [5904, 7344)\n"
        "test 第 t 行：标签小时 7344 + t，即 [7344, 8784)\n"
        "value_scale = 234.0 = max_val - min_val, min_val = 0.0, unit is dc_clipped_count.\n"
        "命名：{source}_seed{seed}_{split}.npz；文件名本身不能自证身份。\n"
    )


def _bundle_script(*, label_start: int = 5832, n_hours: int = 72) -> str:
    return (
        "from src.data.dataset import prepare_city_windows, split_x_y, trim_few_shot\n"
        f"LABEL_START = {label_start}\n"
        f"N_HOURS = {n_hours}\n"
        "min_val = 0.0\n"
        "max_val = 234.0\n"
        "value_scale = max_val - min_val\n"
        "# denorm: pred * value_scale + min_val\n"
        "train = clipped[LABEL_START:LABEL_START + N_HOURS]\n"
        "# val/test come from DataLoader(prepare_city_windows(...)); that helper is not defined here.\n"
    )


def build_synthetic_pr20_bundle(root: Path, *, n_cells: int = 2) -> dict:
    """Build a mini PR20-like bundle. Arrays are synthetic; not real experiment output."""
    bundle = root / "pr20_hourly_bundle"
    (bundle / "predictions" / "train").mkdir(parents=True)
    (bundle / "predictions" / "val").mkdir(parents=True)
    (bundle / "predictions" / "test").mkdir(parents=True)
    (bundle / "gate").mkdir(parents=True)
    (bundle / "source").mkdir(parents=True)

    rows = np.array([1, 2][:n_cells], dtype=np.int64)
    cols = np.array([1, 3][:n_cells], dtype=np.int64)
    eligible = np.array([True, False][:n_cells], dtype=bool)
    raw = np.zeros((N_TRAIN_HOURS, n_cells), dtype=np.float64)
    raw[:, 0] = 10.0
    raw[0, 0] = 264.0  # exceeds 234, clipped target should be 234
    if n_cells > 1:
        raw[:, 1] = 1.0
    raw_total = raw.sum(axis=0)
    _assert(bool((raw_total[0] >= GATE) and (n_cells == 1 or raw_total[1] < GATE)), "fixture gate must be mixed")
    clip_raw = np.clip(raw, 0.0, 234.0)
    train_y = clip_raw.copy()
    train_y[0, 0] = clip_raw[0, 0] + DOCUMENTED_TRAIN_Y_VS_CLIP_RAW_MAXABS
    _assert(
        float(np.max(np.abs(train_y - clip_raw))) == float(DOCUMENTED_TRAIN_Y_VS_CLIP_RAW_MAXABS),
        "fixture residual must match the documented maxabs",
    )
    _assert(not np.array_equal(train_y, clip_raw), "train y must not be byte-identical to clip(raw)")

    val_y = np.full((EXPECTED_VAL_HOURS, n_cells), 3.0, dtype=np.float64)
    test_y = np.full((EXPECTED_TEST_HOURS, n_cells), 4.0, dtype=np.float64)
    scale = float(DOCUMENTED_VALUE_SCALE)

    ckpt = {
        ("ny", 42): "finetune_ny2dc.pt",
        ("ny", 43): "finetune_ny2dc_seed43.pt",
        ("ny", 44): "finetune_ny2dc_seed44.pt",
        ("chi", 42): "finetune_chi2dc.pt",
        ("chi", 43): "finetune_chi2dc_seed43.pt",
        ("chi", 44): "finetune_chi2dc_seed44.pt",
    }
    (bundle / "source" / "checkpoint_sha256.txt").write_text(
        "\n".join(f"{_claimed_hash(name)}  {name}" for name in ckpt.values()) + "\n",
        encoding="utf-8",
    )
    (bundle / "README.txt").write_text(_bundle_readme(), encoding="utf-8")
    (bundle / "source" / "19_cell_affine_v2_at_7a04ca4.py").write_text(_bundle_script(), encoding="utf-8")
    (bundle / "gate" / "norm_and_gate.json").write_text(
        json.dumps({"value_scale": scale, "min_val": 0.0, "max_val": 234.0, "percentile": 99.9}, indent=2),
        encoding="utf-8",
    )
    np.save(bundle / "gate" / "raw_pickup_valid_5832_5904.npy", raw)
    np.save(bundle / "gate" / "raw_total_valid.npy", raw_total)

    members = []
    for source in SOURCES:
        for seed in PRELOCKED_SEEDS:
            rng = np.random.default_rng(1000 + seed + (0 if source == "ny" else 50))
            train_p = 5.0 + rng.normal(scale=0.2, size=(N_TRAIN_HOURS, n_cells))
            val_p = 5.0 + rng.normal(scale=0.2, size=(EXPECTED_VAL_HOURS, n_cells))
            test_p = 5.0 + rng.normal(scale=0.2, size=(EXPECTED_TEST_HOURS, n_cells))
            tpath = bundle / "predictions" / "train" / f"{source}_seed{seed}_train.npz"
            vpath = bundle / "predictions" / "val" / f"{source}_seed{seed}_val.npz"
            xpath = bundle / "predictions" / "test" / f"{source}_seed{seed}_test.npz"
            _write_native_npz(tpath, prediction=train_p, target=train_y, value_scale=scale, eligible=eligible, rows=rows, cols=cols)
            _write_native_npz(vpath, prediction=val_p, target=val_y, value_scale=scale)
            _write_native_npz(xpath, prediction=test_p, target=test_y, value_scale=scale)
            members.extend([
                f"predictions/train/{source}_seed{seed}_train.npz",
                f"predictions/val/{source}_seed{seed}_val.npz",
                f"predictions/test/{source}_seed{seed}_test.npz",
            ])

    extra = [
        "README.txt",
        "SHA256SUMS.txt",
        "source/19_cell_affine_v2_at_7a04ca4.py",
        "source/checkpoint_sha256.txt",
        "gate/norm_and_gate.json",
        "gate/raw_pickup_valid_5832_5904.npy",
        "gate/raw_total_valid.npy",
    ]
    # SHA256SUMS written after hashing everything except itself first, then hashed separately if needed.
    lines = []
    for rel in members + [x for x in extra if x != "SHA256SUMS.txt"]:
        digest = hashlib.sha256((bundle / rel).read_bytes()).hexdigest()
        lines.append(f"{digest}  {rel}")
    (bundle / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    zip_path = root / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in bundle.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(Path("pr20_hourly_bundle") / path.relative_to(bundle)))
    return {"bundle": bundle, "zip_path": zip_path, "n_cells": n_cells}


def _rebuild_bundle_zip(bundle: Path, zip_path: Path) -> None:
    """Rebuild a synthetic ZIP from its extracted tree. Never touches the accepted PR20 zip."""
    zip_path = Path(zip_path)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(Path("pr20_hourly_bundle") / path.relative_to(bundle)))


def _rewrite_sha256sums(bundle: Path, rel: str) -> None:
    """Update one SHA256SUMS line after a deliberate source-file mutation."""
    sums = bundle / "SHA256SUMS.txt"
    digest = hashlib.sha256((bundle / rel).read_bytes()).hexdigest()
    lines = []
    found = False
    for line in sums.read_text(encoding="utf-8").splitlines():
        if line.endswith("  " + rel) or line.endswith(" " + rel):
            lines.append(f"{digest}  {rel}")
            found = True
        else:
            lines.append(line)
    if not found:
        raise AssertionError(f"SHA256SUMS missing {rel}")
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_documented_bundle_adapter(tmp_path: Path | None = None) -> None:
    """Limited adapter tests: opt-in, evidence binding, and concentrated counterexamples."""
    root = Path(tmp_path or ROOT / ".tmp-test-bundle")
    root.mkdir(parents=True, exist_ok=True)

    built = build_synthetic_pr20_bundle(root / "good")
    out = root / "adapted"
    # no opt-in
    failed = False
    try:
        adapt_pr20_bundle(
            zip_path=built["zip_path"],
            extracted_root=built["bundle"],
            output_dir=out / "no-opt-in",
            allow_documented_from_bundle=False,
        )
    except BundleAdapterError as exc:
        failed = "opt-in" in str(exc) or "explicit" in str(exc)
    _assert(failed, "adapter must refuse without explicit opt-in")

    evidence = adapt_pr20_bundle(
        zip_path=built["zip_path"],
        extracted_root=built["bundle"],
        output_dir=out,
        allow_documented_from_bundle=True,
        open_test_arrays=False,
    )
    _assert(evidence["unknown"]["fine_tune_commit"] == "unknown", "FT commit must stay unknown")
    _assert(evidence["unknown"]["val_start_independent_code_proof"] is False, "val start is not code-proven")
    _assert(evidence["hour_chain"]["time_index_origin"] == TIME_INDEX_ORIGIN_DOCUMENTED, "time origin must be documented")
    _assert(evidence["shared_truth"]["train_y_byte_identical_to_clip_raw"] is False, "train y vs clip(raw) is not byte-identical")
    _assert(
        evidence["shared_truth"]["train_y_vs_clip_raw_maxabs"] == float(DOCUMENTED_TRAIN_Y_VS_CLIP_RAW_MAXABS),
        "clip-raw residual must be the documented maxabs",
    )
    _assert(evidence["shared_truth"]["train_targets_byte_identical"] is True, "same-split y must be byte-identical")
    _assert(evidence["original_arrays_unmodified"] is True, "adapter must not mutate arrays")
    cache_dir = Path(evidence["cache_dir"])
    # Original train member hash equals staged copy; arrays were not rewritten.
    for source in SOURCES:
        for seed in PRELOCKED_SEEDS:
            rel = f"predictions/train/{source}_seed{seed}_train.npz"
            staged = cache_dir / f"{source}_seed{seed}_train.npz"
            _assert(sha256_file(staged) == evidence["verified_train_val_members"][rel], "staged copy hash changed")
            _assert(sha256_file(built["bundle"] / rel) == sha256_file(staged), "source array was modified")
    from src.val_select.io_cache import sidecar_path
    side = json.loads(sidecar_path(cache_dir / "ny_seed42_train.npz").read_text(encoding="utf-8"))
    _assert(side["provenance_kind"] == PROVENANCE_DOCUMENTED, "sidecar provenance")
    _assert(side["generator_commit"] == DOCUMENTED_PREDICT_SCRIPT_REVISION, "short revision only")
    _assert(side["generator_commit_form"] == DOCUMENTED_PREDICT_SCRIPT_FORM, "form must be documented_short_revision")
    _assert(side["fine_tune_commit_status"] == "unknown", "FT status")
    _assert(side["checkpoint_sha256_status"] == "claimed", "weight hash is claimed")
    _assert(side["time_index_origin"] == TIME_INDEX_ORIGIN_DOCUMENTED, "documented range")
    _assert(side["input_protocol_id"] == INPUT_PROTOCOL_ID, "input protocol bound")
    _assert(all(not item["arrays_opened"] and not item["extracted"] for item in evidence["declared_test_checksums"].values()), "test arrays must stay unopened")

    # recorded path must still reject a short revision even if documented opt-in is on
    recorded_dir = root / "impersonate-recorded"
    recorded_dir.mkdir()
    from src.val_select.io_cache import write_cache_npz, cache_path
    payload = _arrays_for("ny", 42, "train", provenance_kind="recorded", extras={
        "generator_commit": np.asarray(DOCUMENTED_PREDICT_SCRIPT_REVISION),
        "checkpoint_sha256": np.asarray("a" * 64),
        "checkpoint_path": np.asarray("finetune_ny2dc.pt"),
        "config_id": np.asarray("configs/finetune_ny2dc.yaml"),
    })
    write_cache_npz(cache_path(recorded_dir, "ny", 42, "train"), payload)
    access = CacheAccess(recorded_dir, role="select", allow_synthetic=True, allow_documented_from_bundle=True)
    recorded_failed = False
    try:
        access.load("ny", 42, "train")
    except SchemaError as exc:
        recorded_failed = "40-character" in str(exc) or "recorded" in str(exc)
    _assert(recorded_failed, "recorded mode must still reject a short 7a04ca4 revision; no fallback")

    # documented sidecar without opt-in
    access2 = CacheAccess(cache_dir, role="select", allow_synthetic=True, allow_documented_from_bundle=False)
    opt_failed = False
    try:
        access2.load("ny", 42, "train")
    except SchemaError as exc:
        opt_failed = "opt-in" in str(exc) or "explicit" in str(exc)
    _assert(opt_failed, "loading documented caches without opt-in must fail")

    # happy-path lock binds evidence; select must not open test
    reset_spy()
    lock_path = cache_dir / "selection.lock.json"
    lock = lock_all_selections(
        cache_dir,
        lock_path,
        allow_synthetic=True,
        allow_documented_from_bundle=True,
    )
    _assert(lock["scorable"] is True, "complete documented synthetic bundle must be scorable")
    _assert(lock["input_protocol_id"] == INPUT_PROTOCOL_ID, "lock must bind input protocol")
    _assert(lock["input_evidence"]["zip_sha256"] == evidence["zip_sha256"], "lock must bind zip hash")
    _assert(lock["input_evidence"]["verified_train_val_members"] == evidence["verified_train_val_members"], "lock must bind member hashes")
    _assert(lock["input_evidence"]["adapter_implementation"]["fingerprint"] == evidence["adapter_implementation"]["fingerprint"], "lock must bind adapter fingerprint")
    _assert(test_loads() == 0, "lock-selections must not open test")

    # --- counterexamples, each from a fresh copy of the good extracted tree ---
    def _fresh(name: str):
        dest = root / name
        if dest.exists():
            shutil.rmtree(dest)
        built_i = build_synthetic_pr20_bundle(dest)
        return built_i

    # README vs script hour contradiction
    bad = _fresh("conflict-hours")
    readme = (bad["bundle"] / "README.txt").read_text(encoding="utf-8")
    (bad["bundle"] / "README.txt").write_text(readme.replace("[5904, 7344)", "[5905, 7345)"), encoding="utf-8")
    _rewrite_sha256sums(bad["bundle"], "README.txt")
    _rebuild_bundle_zip(bad["bundle"], bad["zip_path"])
    conflict = False
    conflict_msg = ""
    try:
        adapt_pr20_bundle(
            zip_path=bad["zip_path"],
            extracted_root=bad["bundle"],
            output_dir=root / "conflict-out",
            allow_documented_from_bundle=True,
        )
    except BundleAdapterError as exc:
        conflict = True
        conflict_msg = str(exc)
    _assert(conflict, "README/script/protocol hour contradiction must be refused")
    lowered = conflict_msg.lower()
    _assert(
        any(token in lowered for token in ("contradict", "disagree", "range", "hour", "formula")),
        conflict_msg,
    )

    # member tamper after a successful adapt: staged hash must block selection before scoring
    tampered = root / "tamper"
    if tampered.exists():
        shutil.rmtree(tampered)
    ev = adapt_pr20_bundle(
        zip_path=built["zip_path"],
        extracted_root=built["bundle"],
        output_dir=tampered,
        allow_documented_from_bundle=True,
    )
    victim = Path(ev["cache_dir"]) / "ny_seed42_train.npz"
    # Keep a valid NPZ so the failure is a member-hash mismatch, not a decode error.
    with np.load(victim, allow_pickle=False) as blob:
        arrays = {k: np.array(blob[k]) for k in blob.files}
    arrays["prediction"] = arrays["prediction"] + 0.0
    arrays["prediction"][0, 0] = arrays["prediction"][0, 0] + 1.0
    np.savez_compressed(victim, **arrays)
    reset_spy()
    bad_lock = lock_all_selections(
        Path(ev["cache_dir"]),
        tampered / "bad.lock.json",
        allow_synthetic=True,
        allow_documented_from_bundle=True,
    )
    _assert(bad_lock["scorable"] is False, "tampered member must not produce a scorable lock")
    _assert(test_loads() == 0, "tampered bundle must not open test during select")
    _assert(any("sha256" in err or "member" in err or "input_evidence" in err for err in bad_lock["blocked"]), str(bad_lock["blocked"]))

    # wrong gate: eligible flipped in the source train file
    gate_bad = _fresh("wrong-gate")
    import numpy as npz_fix  # keep numpy
    path = gate_bad["bundle"] / "predictions" / "train" / "ny_seed42_train.npz"
    with np.load(path, allow_pickle=False) as blob:
        arrays = {k: blob[k] for k in blob.files}
    arrays["eligible"] = np.array([False, False], dtype=bool)
    np.savez_compressed(path, **arrays)
    _rewrite_sha256sums(gate_bad["bundle"], "predictions/train/ny_seed42_train.npz")
    _rebuild_bundle_zip(gate_bad["bundle"], gate_bad["zip_path"])
    gate_failed = False
    try:
        adapt_pr20_bundle(
            zip_path=gate_bad["zip_path"],
            extracted_root=gate_bad["bundle"],
            output_dir=root / "wrong-gate-out",
            allow_documented_from_bundle=True,
        )
    except BundleAdapterError as exc:
        gate_failed = "eligible" in str(exc) or "gate" in str(exc)
    _assert(gate_failed, "wrong eligible/gate must be refused")

    # wrong coordinates
    coord = _fresh("wrong-coords")
    path = coord["bundle"] / "predictions" / "train" / "chi_seed43_train.npz"
    with np.load(path, allow_pickle=False) as blob:
        arrays = {k: blob[k] for k in blob.files}
    arrays["rows"] = arrays["rows"] + 10
    np.savez_compressed(path, **arrays)
    _rewrite_sha256sums(coord["bundle"], "predictions/train/chi_seed43_train.npz")
    _rebuild_bundle_zip(coord["bundle"], coord["zip_path"])
    coord_failed = False
    try:
        adapt_pr20_bundle(
            zip_path=coord["zip_path"],
            extracted_root=coord["bundle"],
            output_dir=root / "wrong-coords-out",
            allow_documented_from_bundle=True,
        )
    except BundleAdapterError as exc:
        coord_failed = "rows" in str(exc) or "cols" in str(exc) or "geometry" in str(exc)
    _assert(coord_failed, "wrong rows/cols must be refused")

    # impersonate recorded on a documented sidecar, then select must not open test
    fake = root / "fake-recorded"
    if fake.exists():
        shutil.rmtree(fake)
    ev = adapt_pr20_bundle(
        zip_path=built["zip_path"],
        extracted_root=built["bundle"],
        output_dir=fake,
        allow_documented_from_bundle=True,
    )
    from src.val_select.io_cache import sidecar_path as _side
    side_path = _side(Path(ev["cache_dir"]) / "ny_seed42_train.npz")
    sidecar = json.loads(side_path.read_text(encoding="utf-8"))
    sidecar["provenance_kind"] = "recorded"
    sidecar["generator_commit"] = DOCUMENTED_PREDICT_SCRIPT_REVISION
    side_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    reset_spy()
    impersonate_failed = False
    try:
        lock_all_selections(
            Path(ev["cache_dir"]),
            fake / "fake.lock.json",
            allow_synthetic=True,
            allow_documented_from_bundle=True,
        )
    except SchemaError as exc:
        impersonate_failed = "recorded" in str(exc) or "40-character" in str(exc) or "commit" in str(exc)
    _assert(impersonate_failed, "impersonating recorded must be refused by the unchanged recorded path")
    _assert(test_loads() == 0, "recorded impersonation must not open test")



def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _stage_synthetic_documented_tests(cache_dir: Path, bundle: Path, evidence: dict) -> None:
    """Copy synthetic test NPZs into the staged cache. Does not touch the real PR20 zip."""
    from src.val_select.bundle_adapter import documented_time_index
    from src.val_select.io_cache import cache_path, sidecar_path, sha256_file

    template = json.loads(sidecar_path(cache_dir / "ny_seed42_val.npz").read_text(encoding="utf-8"))
    for source in SOURCES:
        for seed in PRELOCKED_SEEDS:
            rel = f"predictions/test/{source}_seed{seed}_test.npz"
            src = bundle / rel
            dest = cache_path(cache_dir, source, int(seed), "test")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            declared = evidence["declared_test_checksums"][rel]["declared_sha256"]
            _assert(sha256_file(dest) == declared, "synthetic test copy must match the declared ZIP checksum")
            side = json.loads(json.dumps(template))
            side["source"] = source
            side["seed"] = int(seed)
            side["split"] = "test"
            side["time_index"] = documented_time_index("test").tolist()
            side["native_member"] = rel
            side["native_member_sha256"] = declared
            if "field_sources" in side:
                if "prediction" in side["field_sources"]:
                    side["field_sources"]["prediction"] = {"layer": "observed", "member": rel, "member_sha256": declared}
                if "target" in side["field_sources"]:
                    side["field_sources"]["target"] = {"layer": "observed", "member": rel, "member_sha256": declared}
            sidecar_path(dest).write_text(json.dumps(side, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_evidence_closure_b1_b2(tmp_path: Path | None = None) -> None:
    """B1/B2 evidence-closure counterexamples. Synthetic copies only; real zip is not used."""
    from src.val_select.io_cache import sidecar_path, sha256_file

    root = Path(tmp_path or ROOT / ".tmp-test-evidence-closure")
    root.mkdir(parents=True, exist_ok=True)

    def isolated(name: str):
        dest = root / name
        if dest.exists():
            shutil.rmtree(dest)
        built = build_synthetic_pr20_bundle(dest / "pkg")
        ev = adapt_pr20_bundle(
            zip_path=built["zip_path"],
            extracted_root=built["bundle"],
            output_dir=dest / "out",
            allow_documented_from_bundle=True,
        )
        return built, ev, Path(ev["cache_dir"])

    def mutate_evidence(cache_dir: Path, mutator) -> None:
        ev_path = cache_dir / "bundle_evidence.json"
        claimed = json.loads(ev_path.read_text(encoding="utf-8"))
        mutator(claimed)
        _write_json(ev_path, claimed)
        parent = cache_dir.parent / "bundle_evidence.json"
        if parent.is_file():
            _write_json(parent, claimed)

    def assert_blocked(cache_dir: Path, lock_path: Path, tokens: tuple[str, ...]) -> dict:
        reset_spy()
        reset_candidate_score_calls()
        lock = lock_all_selections(
            cache_dir,
            lock_path,
            allow_synthetic=True,
            allow_documented_from_bundle=True,
        )
        _assert(lock["scorable"] is False, "tampered evidence must not be scorable")
        _assert(select_mod.candidate_score_calls == 0, "candidate scoring must not run")
        _assert(test_loads() == 0, "select must not read test")
        blocked = " ".join(str(x) for x in lock.get("blocked", []))
        _assert(any(tok.lower() in blocked.lower() for tok in tokens), blocked)
        return lock

    # --- B1: stale evidence_hash after fingerprint-only edit ---
    built, ev, cache_dir = isolated("b1-fingerprint")
    def _fp(claimed):
        claimed["adapter_implementation"]["fingerprint"] = "0" * 64
        # deliberately keep the previous evidence_hash
    mutate_evidence(cache_dir, _fp)
    assert_blocked(cache_dir, cache_dir / "lock.json", ("evidence_hash", "fingerprint"))

    # --- B1: stale evidence_hash after README hash-only edit ---
    built, ev, cache_dir = isolated("b1-readme-hash")
    def _readme(claimed):
        claimed["source_files"]["README.txt"] = "b" * 64
    mutate_evidence(cache_dir, _readme)
    assert_blocked(cache_dir, cache_dir / "lock.json", ("evidence_hash", "README"))

    # --- B1: swap the ZIP after evidence is generated ---
    built, ev, cache_dir = isolated("b1-swap-zip")
    other = build_synthetic_pr20_bundle(root / "b1-swap-zip" / "other")
    (other["bundle"] / "README.txt").write_text(
        other["bundle"].joinpath("README.txt").read_text(encoding="utf-8") + "\n# zip swap marker\n",
        encoding="utf-8",
    )
    _rewrite_sha256sums(other["bundle"], "README.txt")
    _rebuild_bundle_zip(other["bundle"], other["zip_path"])
    shutil.copy2(other["zip_path"], ev["zip_path"])
    _assert(sha256_file(Path(ev["zip_path"])) != ev["zip_sha256"], "swapped zip must have a new hash")
    assert_blocked(cache_dir, cache_dir / "lock.json", ("zip", "sha256"))

    # --- B1: rewrite a document inside the same zip path after evidence ---
    built, ev, cache_dir = isolated("b1-swap-doc")
    readme = (built["bundle"] / "README.txt").read_text(encoding="utf-8")
    (built["bundle"] / "README.txt").write_text(readme + "\n# document swap marker\n", encoding="utf-8")
    _rewrite_sha256sums(built["bundle"], "README.txt")
    _rebuild_bundle_zip(built["bundle"], built["zip_path"])
    assert_blocked(cache_dir, cache_dir / "lock.json", ("zip", "README", "source file"))

    # --- B1: drop a required SHA256SUMS entry after evidence is generated ---
    built, ev, cache_dir = isolated("b1-drop-sums")
    sums = (built["bundle"] / "SHA256SUMS.txt").read_text(encoding="utf-8")
    lines = [ln for ln in sums.splitlines() if "ny_seed42_train.npz" not in ln]
    _assert(len(lines) < len(sums.splitlines()), "should have removed a required checksum")
    (built["bundle"] / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _rebuild_bundle_zip(built["bundle"], built["zip_path"])
    assert_blocked(cache_dir, cache_dir / "lock.json", ("SHA256SUMS", "missing", "required"))

    # --- B2: synthetic test byte tamper is rejected before NPZ decode ---
    built, ev, cache_dir = isolated("b2-test-bytes")
    reset_spy()
    reset_candidate_score_calls()
    lock = lock_all_selections(
        cache_dir,
        cache_dir / "ok.lock.json",
        allow_synthetic=True,
        allow_documented_from_bundle=True,
    )
    _assert(lock["scorable"] is True, "untampered isolated bundle must still lock")
    _assert(test_loads() == 0, "lock must not open test")
    _stage_synthetic_documented_tests(cache_dir, built["bundle"], ev)
    victim = cache_dir / "ny_seed42_test.npz"
    payload = bytearray(victim.read_bytes())
    payload[0] = (payload[0] + 1) % 256
    victim.write_bytes(bytes(payload))
    reset_spy()
    score_failed = False
    score_msg = ""
    try:
        score_test(
            cache_dir,
            cache_dir / "ok.lock.json",
            cache_dir / "report.json",
            allow_synthetic=True,
            allow_documented_from_bundle=True,
            run_dir=cache_dir / "run-b2",
            test_access_factory=SpyAccess,
            i_understand_this_is_exploratory=True,
        )
    except ValueError as exc:
        score_failed = True
        score_msg = str(exc)
    _assert(score_failed, "tampered test bytes must refuse scoring")
    _assert("refusing to decode" in score_msg or "byte sha256" in score_msg, score_msg)
    _assert(test_loads() == 0, "mismatched test bytes must not be NPZ-decoded")



def test_claimed_checkpoint_hash_bom_crlf() -> None:
    """BOM/CRLF text decoding is equivalent; 64-hex validation is not relaxed."""
    rows = [
        ("E279B4E6BAD1620A4295487A74F114EA3FDDE3422BEDE801F644EBA75BBC6C3E", "finetune_ny2dc.pt"),
        ("C25EAF82CEC11C71ACDDF2CE16989BA26DB41FCDB8949ECDB64BE554BC27E23E", "finetune_ny2dc_seed43.pt"),
        ("33E4B66E476FDCCC12EA867F74510ABA8D022BBE11A7C8D1A28DA798855DE497", "finetune_chi2dc.pt"),
    ]
    expected = {name: digest.lower() for digest, name in rows}
    body = "".join(f"{digest}  {name}\n" for digest, name in rows)
    encodings = {
        "no_bom_lf": body.encode("utf-8"),
        "no_bom_crlf": body.replace("\n", "\r\n").encode("utf-8"),
        "bom_lf": b"\xef\xbb\xbf" + body.encode("utf-8"),
        "bom_crlf": b"\xef\xbb\xbf" + body.replace("\n", "\r\n").encode("utf-8"),
    }
    parsed = {}
    for label, raw in encodings.items():
        text = decode_utf8_sig_text(raw)
        _assert(not text.startswith("\ufeff"), f"{label}: utf-8-sig must strip the BOM")
        parsed[label] = parse_claimed_checkpoint_hashes(text)
        _assert(parsed[label] == expected, f"{label} must parse the same claimed hashes")
    _assert(len({tuple(sorted(v.items())) for v in parsed.values()}) == 1, "all four encodings must be equivalent")

    # Plain utf-8 decode of a BOM file still works after the leftover U+FEFF strip,
    # and is equivalent to utf-8-sig. A raw 65-character hex token is still illegal.
    leftover = encodings["bom_crlf"].decode("utf-8")
    _assert(leftover.startswith("\ufeff"), "control: utf-8 keeps the BOM")
    _assert(parse_claimed_checkpoint_hashes(leftover) == expected, "leftover BOM must not drop the first hash")

    illegal_lines = [
        "E279B4E6BAD1620A4295487A74F114EA3FDDE3422BEDE801F644EBA75BBC6C3  too_short.pt",
        "E279B4E6BAD1620A4295487A74F114EA3FDDE3422BEDE801F644EBA75BBC6C3EE  too_long.pt",
        "G279B4E6BAD1620A4295487A74F114EA3FDDE3422BEDE801F644EBA75BBC6C3E  not_hex.pt",
        "finetune_ny2dc.pt",
        "",
    ]
    illegal_text = decode_utf8_sig_text(("\r\n".join(illegal_lines) + "\r\n").encode("utf-8"))
    rejected = parse_claimed_checkpoint_hashes(illegal_text)
    _assert(rejected == {}, f"illegal hashes must be rejected, got {rejected}")
    # Valid 64-hex on the same file as an illegal line still parses; the illegal line does not leak in.
    mixed = decode_utf8_sig_text(
        (b"\xef\xbb\xbf" + (
            "E279B4E6BAD1620A4295487A74F114EA3FDDE3422BEDE801F644EBA75BBC6C3EE  too_long.pt\r\n"
            "C25EAF82CEC11C71ACDDF2CE16989BA26DB41FCDB8949ECDB64BE554BC27E23E  finetune_ny2dc_seed43.pt\r\n"
        ).encode("utf-8"))
    )
    mixed_parsed = parse_claimed_checkpoint_hashes(mixed)
    _assert(mixed_parsed == {"finetune_ny2dc_seed43.pt": "c25eaf82cec11c71acddf2ce16989ba26db41fcdb8949ecdb64be554bc27e23e"}, mixed_parsed)
    _assert("too_long.pt" not in mixed_parsed, "a 65-character token must not pass the 64-hex check")


TESTS = [
    test_claimed_checkpoint_hash_bom_crlf,
    test_evidence_closure_b1_b2,
    test_documented_bundle_adapter,
    test_solvers_zero_and_identity,
    test_pr20_agreement_finite_strength,
    test_zero_strength_high_mean_low_variance_and_boundaries,
    test_pick_strength_ties,
    test_nonfinite_or_short_series_blocks,
    test_blocks_use_timestamps_and_refuse_gaps,
    test_cache_access_select_cannot_open_test,
    test_inventory_does_not_unpack,
    test_schema_rejects_wrong_seed_columns_unit_hours_gate,
    test_lock_and_score_isolation_and_overwrite,
    test_b1_tamper_train_val_plan_solver_config,
    test_b2_missing_and_incomplete_locks,
    test_filename_is_not_identity_proof,
    test_sidecar_conflicts_and_equivalent_dual,
    test_schema_parameterized_rejects,
    test_val_norm_change_and_shared_truth,
    test_different_seeds_may_have_different_weight_hashes,
]


def main() -> int:
    import inspect
    import tempfile

    before = worktree_source_stamp()
    print("worktree_stamp_before_tests:")
    for rel, item in before.items():
        print(f"  {rel} sha256={item['sha256']} mtime_ns={item['mtime_ns']} size={item['size']}")
    failures = []
    for fn in TESTS:
        tmp = Path(tempfile.mkdtemp(prefix="val-select-"))
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
