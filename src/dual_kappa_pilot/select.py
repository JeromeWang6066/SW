"""Train/val selection for the dual-kappa pilot. Never opens test arrays."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.dual_kappa_pilot.protocol import (
    ARMS,
    BUDGET_NOTE,
    EXPECTED_VAL_HOURS,
    FIXED_ARMS,
    IDENTITY,
    INPUT_PROTOCOL_ID,
    LABEL_BUDGET,
    N_GATE_EXPECTED,
    N_TRAIN_HOURS,
    N_VALID_CELLS_EXPECTED,
    PRELOCKED_SEEDS,
    PROTOCOL_ID,
    PROVENANCE_DOCUMENTED,
    SOURCES,
    budget_expand30,
    budget_nested10,
    canonical_strength_pair,
    candidate_records,
    is_identity,
    pick_strength,
)
from src.dual_kappa_pilot.snapshot import binding_payload
from src.dual_kappa_pilot.solvers import (
    apply_affine,
    fit_cells_dual,
    fit_identity,
    fit_new_k1,
    fit_old_m1,
    near_constant_report,
)
from src.dual_kappa_pilot.typology import lock_typology, typologies_must_agree
from src.val_select.bundle_adapter import verify_lock_input_evidence, verify_staged_cache_against_evidence
from src.val_select.io_cache import CacheAccess, canonical_hash, sha256_file, sidecar_path, write_json, cache_path
from src.val_select.metrics import gated_mae, score_arrays
from src.val_select.schema import SchemaError
from src.val_select.truth import assert_shared_dc_split, assert_task_split_consistency

candidate_score_calls = 0
test_inventory_hash_calls = 0
input_load_calls = 0


def reset_candidate_score_calls() -> None:
    global candidate_score_calls, test_inventory_hash_calls, input_load_calls
    candidate_score_calls = 0
    test_inventory_hash_calls = 0
    input_load_calls = 0


def _lock_path_exists(path: Path) -> bool:
    """True if the path exists in any form, including a broken symlink or a directory."""
    return os.path.lexists(str(path))


def reject_existing_lock_path(lock_path) -> Path:
    path = Path(lock_path)
    if _lock_path_exists(path):
        raise FileExistsError(
            f"refusing to use existing lock path {path}; "
            "selection locks are created exclusively and never overwritten"
        )
    return path


def write_json_exclusive(path: Path, payload: Any) -> None:
    """Create a new JSON file with O_CREAT|O_EXCL. Never overwrites an existing path."""
    path = Path(path)
    if _lock_path_exists(path):
        raise FileExistsError(f"refusing to overwrite existing lock path {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(str(path), flags, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


def _mark_candidate_score() -> None:
    global candidate_score_calls
    candidate_score_calls += 1


def expected_task_keys(sources=SOURCES, seeds=PRELOCKED_SEEDS) -> list[str]:
    return [f"{source}:{int(seed)}" for source in sources for seed in seeds]


def _eligible(train: dict[str, Any], *, require_real_counts: bool) -> np.ndarray:
    if train.get("eligible") is None:
        raise ValueError("train cache missing eligible gate flags")
    eligible = np.asarray(train["eligible"], dtype=bool)
    if train["prediction"].shape[0] != N_TRAIN_HOURS:
        raise ValueError(f"train hours {train['prediction'].shape[0]} != {N_TRAIN_HOURS}")
    if require_real_counts:
        if train["prediction"].shape[1] != N_VALID_CELLS_EXPECTED:
            raise ValueError(
                f"cell count {train['prediction'].shape[1]} != {N_VALID_CELLS_EXPECTED}"
            )
        if int(eligible.sum()) != N_GATE_EXPECTED:
            raise ValueError(f"gated cells {int(eligible.sum())} != {N_GATE_EXPECTED}")
    return eligible


def _fit_for_arm(arm: str, candidate: Any, train_pred, train_true, eligible):
    if arm == "FT":
        return fit_identity(train_pred, train_true, eligible)
    if arm == "OLD_FIXED":
        return fit_old_m1(train_pred, train_true, eligible)
    if arm == "NEW_FIXED":
        return fit_new_k1(train_pred, train_true, eligible)
    return fit_cells_dual(train_pred, train_true, eligible, candidate)


def _metrics(applied, truth, eligible, value_scale) -> dict[str, Any]:
    metrics = score_arrays(applied, truth, value_scale)
    return {
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mape": metrics["mape"],
        "gated_mae": gated_mae(applied, truth, eligible),
        "n_points": metrics["n_points"],
        "hours": metrics["hours"],
        "n_cells": metrics["n_cells"],
    }


def _curve_and_choice(
    budget: tuple[Any, ...],
    train_pred,
    train_true,
    eligible,
    val_pred,
    val_true,
    value_scale,
) -> tuple[list[dict[str, Any]], Any]:
    _mark_candidate_score()
    curve: list[dict[str, Any]] = []
    for candidate in budget:
        try:
            scale, shift = fit_cells_dual(train_pred, train_true, eligible, candidate)
            applied = apply_affine(val_pred, scale, shift)
            metrics = _metrics(applied, val_true, eligible, value_scale)
            status = "ok"
            detail = None
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(
                f"numerical/data failure at candidate={canonical_strength_pair(candidate)}: {exc}"
            ) from exc
        curve.append(
            {
                "candidate": canonical_strength_pair(candidate),
                "kappa_alpha": None if is_identity(candidate) else float(candidate[0]),
                "kappa_c": None if is_identity(candidate) else float(candidate[1]),
                "diagonal": False if is_identity(candidate) else float(candidate[0]) == float(candidate[1]),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "gated_mae": metrics["gated_mae"],
                "n_points": metrics["n_points"],
                "status": status,
                "detail": detail,
            }
        )
    typed = []
    for item in curve:
        cand = IDENTITY if item["candidate"] == IDENTITY else (float(item["kappa_alpha"]), float(item["kappa_c"]))
        typed.append({**item, "candidate": cand})
    chosen = pick_strength(typed)
    return curve, chosen


def _inventory_test_bytes(cache_dir: Path, sources, seeds) -> list[dict[str, Any]]:
    """Hash test *files* without decoding. Missing files are recorded, not invented."""
    global test_inventory_hash_calls
    out = []
    for source in sources:
        for seed in seeds:
            path = cache_path(Path(cache_dir), source, int(seed), "test")
            rec: dict[str, Any] = {
                "source": source,
                "seed": int(seed),
                "path": str(path),
                "present": path.is_file(),
                "sha256": None,
                "decoded": False,
            }
            if path.is_file():
                test_inventory_hash_calls += 1
                rec["sha256"] = sha256_file(path)
            out.append(rec)
    return out


def scorable_errors(lock: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not lock.get("scorable"):
        errors.append("lock is marked not scorable")
    if lock.get("missing"):
        errors.append(f"missing caches: {lock['missing']}")
    if lock.get("blocked"):
        errors.append(f"blocked: {lock['blocked']}")
    if lock.get("protocol_id") != PROTOCOL_ID:
        errors.append(f"protocol_id {lock.get('protocol_id')!r} != {PROTOCOL_ID}")
    sources = list(lock.get("sources") or [])
    seeds = list(lock.get("prelocked_seeds") or [])
    arms = list(lock.get("arms") or [])
    if sources != list(SOURCES):
        errors.append(f"sources {sources} != pre-locked {list(SOURCES)}")
    if [int(s) for s in seeds] != [int(s) for s in PRELOCKED_SEEDS]:
        errors.append("prelocked seed set does not match the frozen protocol set")
    if arms != list(ARMS):
        errors.append("arm list is incomplete or has extra arms")
    expected10 = [canonical_strength_pair(x) for x in budget_nested10()]
    expected30 = [canonical_strength_pair(x) for x in budget_expand30()]
    if list(lock.get("budget_nested10") or []) != expected10:
        errors.append("budget_nested10 does not exactly match the frozen 10-candidate grid")
    if list(lock.get("budget_expand30") or []) != expected30:
        errors.append("budget_expand30 does not exactly match the frozen 30-candidate grid")
    if len(expected30) != 30 or len(expected10) != 10:
        errors.append("internal grid length error")
    keys = expected_task_keys(sources or SOURCES, seeds or PRELOCKED_SEEDS)
    tasks = lock.get("tasks") or {}
    if set(tasks) != set(keys):
        errors.append("task set is incomplete")
    for key in keys:
        arms_here = tasks.get(key) or {}
        if set(arms_here) != set(ARMS):
            errors.append(f"{key} is missing arms")
            continue
        for arm in ARMS:
            if "chosen" not in arms_here[arm]:
                errors.append(f"{key} {arm} missing chosen strength")
        single = arms_here.get("SINGLE_VAL10") or {}
        double = arms_here.get("DOUBLE_VAL30") or {}
        if [row.get("candidate") for row in (single.get("curve") or [])] != expected10:
            errors.append(f"{key} SINGLE_VAL10 curve is not the frozen 10-candidate budget")
        if [row.get("candidate") for row in (double.get("curve") or [])] != expected30:
            errors.append(f"{key} DOUBLE_VAL30 curve is not the frozen 30-candidate budget")
    if "typology" not in lock or not lock["typology"]:
        errors.append("train typologies are missing")
    if "stop_rule" not in lock:
        errors.append("pre-locked STOP rule is missing")
    return errors


def assert_lock_scorable(lock: dict[str, Any]) -> None:
    errors = scorable_errors(lock)
    if errors:
        raise ValueError("lock is not scorable: " + "; ".join(errors))


def lock_all_selections(
    cache_dir,
    lock_path,
    sources=SOURCES,
    seeds=PRELOCKED_SEEDS,
    *,
    allow_synthetic: bool = False,
    allow_documented_from_bundle: bool = False,
    seed_freeze_id: str | None = None,
    root=None,
) -> dict[str, Any]:
    """Fit and select on train/val only. Writes a scorable lock iff complete.

    Test arrays are never decoded. Optional test *file* sha256 values are
    recorded by inventory hashing only, for later compare-before-decode.

    If lock_path already exists in any form, this function raises before any
    cache load or candidate scoring. The final write uses an exclusive create
    and cannot overwrite, including in a late race.
    """
    lock_path = reject_existing_lock_path(lock_path)
    requested_seeds = [int(s) for s in (seeds if seeds is not None else PRELOCKED_SEEDS)]
    requested_sources = list(sources if sources is not None else SOURCES)
    if requested_seeds != [int(s) for s in PRELOCKED_SEEDS] or requested_sources != list(SOURCES):
        if not seed_freeze_id:
            raise ValueError(
                "requested source/seed set differs from the frozen protocol set "
                f"sources={list(SOURCES)} seeds={list(PRELOCKED_SEEDS)}. "
                "Report the change and create a new versioned freeze with seed_freeze_id; "
                "seeds are not dropped automatically."
            )

    access = CacheAccess(
        cache_dir,
        role="select",
        allow_synthetic=allow_synthetic,
        allow_documented_from_bundle=allow_documented_from_bundle,
    )
    missing: list[dict[str, Any]] = []
    blocked: list[str] = []
    input_hashes: list[dict[str, Any]] = []
    loaded_trains: list[dict[str, Any]] = []
    loaded_vals: list[dict[str, Any]] = []
    loaded_tasks: list[dict[str, Any]] = []

    for source in requested_sources:
        for seed in requested_seeds:
            try:
                global input_load_calls
                input_load_calls += 1
                train = access.load(source, seed, "train")
                if access.expected_cell_ids is None:
                    access.expected_cell_ids = np.asarray(train["cell_ids"])
                    access.expected_cells = int(access.expected_cell_ids.size)
                input_load_calls += 1
                val = access.load(source, seed, "val")
            except (FileNotFoundError, SchemaError, ValueError, PermissionError) as exc:
                missing.append({"source": source, "seed": int(seed), "error": str(exc)})
                continue
            try:
                assert_task_split_consistency(train, val)
                if int(val["prediction"].shape[0]) != EXPECTED_VAL_HOURS and not allow_synthetic:
                    raise ValueError(f"val hours {val['prediction'].shape[0]} != {EXPECTED_VAL_HOURS}")
                eligible = _eligible(train, require_real_counts=not allow_synthetic)
            except (SchemaError, ValueError) as exc:
                blocked.append(f"{source}:{seed}: {exc}")
                continue
            loaded_trains.append(train)
            loaded_vals.append(val)
            input_hashes.append(
                {
                    "source": source,
                    "seed": int(seed),
                    "train_sha256": train["_sha256"],
                    "val_sha256": val["_sha256"],
                    "train_sidecar_sha256": train.get("_sidecar_sha256"),
                    "val_sidecar_sha256": val.get("_sidecar_sha256"),
                    "train_manifest": train["manifest"],
                    "val_manifest": val["manifest"],
                    "near_constant": near_constant_report(train["prediction"], train["eligible"]),
                }
            )
            loaded_tasks.append(
                {
                    "source": source,
                    "seed": int(seed),
                    "key": f"{source}:{seed}",
                    "train": train,
                    "val": val,
                    "eligible": eligible,
                    "value_scale": float(val["value_scale"]),
                }
            )

    input_evidence = None
    evidence_errors: list[str] = []
    tasks: dict[str, Any] = {}
    typology = None
    if not missing and not blocked:
        try:
            assert_shared_dc_split(loaded_trains, split="train", compare_gate=True)
            assert_shared_dc_split(loaded_vals, split="val", compare_gate=False)
        except SchemaError as exc:
            evidence_errors.append(f"shared DC train/val labels disagree: {exc}")
        provenances = {str(item["train"]["provenance_kind"]) for item in loaded_tasks} | {
            str(item["val"]["provenance_kind"]) for item in loaded_tasks
        }
        if PROVENANCE_DOCUMENTED in provenances:
            if provenances != {PROVENANCE_DOCUMENTED}:
                evidence_errors.append(
                    "mixed provenance is not allowed; documented_from_bundle cannot mix with recorded or synthetic"
                )
            if not allow_documented_from_bundle:
                evidence_errors.append("documented_from_bundle requires explicit opt-in; no fallback")
            ev_path = Path(cache_dir) / "bundle_evidence.json"
            if not ev_path.is_file():
                evidence_errors.append("documented_from_bundle is missing bundle_evidence.json next to the staged cache")
            else:
                import json as _json

                input_evidence = _json.loads(ev_path.read_text(encoding="utf-8"))
                evidence_errors.extend(
                    verify_lock_input_evidence(
                        {
                            "input_protocol_id": INPUT_PROTOCOL_ID,
                            "provenance_kind": PROVENANCE_DOCUMENTED,
                            "input_evidence": input_evidence,
                        },
                        cache_dir=cache_dir,
                        zip_path=input_evidence.get("zip_path"),
                        root=root,
                    )
                )
                evidence_errors.extend(verify_staged_cache_against_evidence(cache_dir, input_evidence))
        # All train/val schema, shared truth, gate, and evidence must pass
        # before any candidate or fixed-arm scoring.
        if evidence_errors:
            loaded_tasks = []
            blocked.extend(evidence_errors)
        typologies = []
        for item in loaded_tasks:
            typologies.append(lock_typology(item["train"]["target"], item["eligible"]))
        if typologies:
            typology = typologies[0]
            for other in typologies[1:]:
                typologies_must_agree(typology, other, context="shared train DC truth")
        for item in loaded_tasks:
            train = item["train"]
            val = item["val"]
            eligible = item["eligible"]
            value_scale = item["value_scale"]
            arm_results: dict[str, Any] = {}
            # Fixed arms first, then the two val-tuned budgets.
            for arm, fitter, label in (
                ("FT", fit_identity, IDENTITY),
                ("OLD_FIXED", fit_old_m1, "1"),
                ("NEW_FIXED", fit_new_k1, "1,1"),
            ):
                _mark_candidate_score()
                scale, shift = fitter(train["prediction"], train["target"], eligible)
                applied = apply_affine(val["prediction"], scale, shift)
                scored = _metrics(applied, val["target"], eligible, value_scale)
                arm_results[arm] = {
                    "kind": "fixed",
                    "chosen": label,
                    "val": scored,
                }
            curve10, chosen10 = _curve_and_choice(
                budget_nested10(),
                train["prediction"],
                train["target"],
                eligible,
                val["prediction"],
                val["target"],
                value_scale,
            )
            curve30, chosen30 = _curve_and_choice(
                budget_expand30(),
                train["prediction"],
                train["target"],
                eligible,
                val["prediction"],
                val["target"],
                value_scale,
            )
            arm_results["SINGLE_VAL10"] = {
                "kind": "tuned",
                "budget": "budget_nested10",
                "chosen": canonical_strength_pair(chosen10),
                "chose_identity": canonical_strength_pair(chosen10) == IDENTITY,
                "curve": curve10,
            }
            arm_results["DOUBLE_VAL30"] = {
                "kind": "tuned",
                "budget": "budget_expand30",
                "chosen": canonical_strength_pair(chosen30),
                "chose_identity": canonical_strength_pair(chosen30) == IDENTITY,
                "curve": curve30,
                "nested10_would_choose": canonical_strength_pair(chosen10),
            }
            tasks[item["key"]] = arm_results

    test_file_inventory = _inventory_test_bytes(Path(cache_dir), requested_sources, requested_seeds)
    binding = binding_payload(root)
    complete = (not missing) and (not blocked) and set(tasks) == set(expected_task_keys(requested_sources, requested_seeds))
    lock: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "input_protocol_id": INPUT_PROTOCOL_ID if input_evidence else None,
        "provenance_kind": (PROVENANCE_DOCUMENTED if input_evidence else ("synthetic" if allow_synthetic else None)),
        "scorable": False,
        "sources": list(requested_sources),
        "prelocked_seeds": list(requested_sources and requested_seeds),
        "arms": list(ARMS),
        "budget_nested10": [canonical_strength_pair(x) for x in budget_nested10()],
        "budget_expand30": [canonical_strength_pair(x) for x in budget_expand30()],
        "candidates_nested10": candidate_records(budget_nested10()),
        "candidates_expand30": candidate_records(budget_expand30()),
        "budget_note": BUDGET_NOTE,
        "label_budget": LABEL_BUDGET,
        "tasks": tasks,
        "typology": typology,
        "stop_rule": {
            "mae_rel_min": 0.001,
            "rmse_ratio_max": 1.005,
            "n_tasks_le_old_min": 4,
            "eq_abs": 1e-10,
            "eq_rel": 1e-12,
            "references": ["OLD_FIXED", "SINGLE_VAL10"],
            "aggregation": "per-source arithmetic mean of three seeds",
            "note": (
                "Engineering screen, not a significance test. Full-precision ratios. "
                "All four MAE comparisons, all four RMSE comparisons, and the 4/6 "
                "task rule must hold together. Do not retune after seeing test."
            ),
        },
        "input_hashes": input_hashes,
        "input_evidence": input_evidence,
        "test_file_sha256_at_lock": test_file_inventory,
        "missing": missing,
        "blocked": blocked,
        "binding": binding,
        "nongate_remains_ft": True,
        "no_clip_negative": True,
    }
    # typo guard: prelocked_seeds assignment above used a confusing expression; fix.
    lock["prelocked_seeds"] = list(requested_seeds)
    if complete:
        try:
            assert_lock_scorable({**lock, "scorable": True})
            lock["scorable"] = True
        except ValueError as exc:
            lock["scorable"] = False
            lock["blocked"] = list(blocked) + [str(exc)]
    lock["lock_hash"] = canonical_hash({k: v for k, v in lock.items() if k != "lock_hash"})
    # Exclusive create: a race that appears after the opening check still cannot overwrite.
    write_json_exclusive(Path(lock_path), lock)
    return lock
