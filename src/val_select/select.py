"""Non-test selection for all three stages. Never opens test caches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.val_select.blocks import four_contiguous_blocks, rolling_rounds, weighted_block_mae
from src.val_select.bundle_adapter import verify_lock_input_evidence, verify_staged_cache_against_evidence
from src.val_select.io_cache import CacheAccess, canonical_hash, write_json
from src.val_select.metrics import gated_mae, score_arrays
from src.val_select.protocol import (
    ARM_FIXED_STRENGTH,
    ARM_METHOD,
    ARMS,
    CANDIDATES,
    FIXED_ARMS,
    IDENTITY,
    LABEL_BUDGET,
    N_GATE_EXPECTED,
    N_TRAIN_HOURS,
    N_VALID_CELLS_EXPECTED,
    N_VAL_BLOCKS,
    PRELOCKED_SEEDS,
    PROTOCOL_ID,
    SOURCES,
    TUNABLE_ARMS,
    canonical_strength,
    pick_strength,
    INPUT_PROTOCOL_ID,
    PROVENANCE_DOCUMENTED,
)
from src.val_select.snapshot import binding_payload
from src.val_select.solvers import apply_affine, fit_cells, near_constant_report
from src.val_select.truth import assert_shared_dc_split, assert_task_split_consistency


candidate_score_calls = 0


def reset_candidate_score_calls() -> None:
    """Test hook: number of candidate/fixed-arm scoring calls since the last reset."""
    global candidate_score_calls
    candidate_score_calls = 0


def _mark_candidate_score() -> None:
    global candidate_score_calls
    candidate_score_calls += 1



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


def _curve_and_choice(
    method: str,
    train_pred: np.ndarray,
    train_true: np.ndarray,
    eligible: np.ndarray,
    val_pred: np.ndarray,
    val_true: np.ndarray,
    value_scale: float,
    positions: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    _mark_candidate_score()
    if positions is not None:
        val_pred = val_pred[positions]
        val_true = val_true[positions]
    curve = []
    for strength in CANDIDATES:
        try:
            scale, shift = fit_cells(train_pred, train_true, eligible, method, strength)
            applied = apply_affine(val_pred, scale, shift)
            metrics = score_arrays(applied, val_true, value_scale)
            gated = gated_mae(applied, val_true, eligible)
            status = "ok"
            detail = None
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(
                f"numerical/data failure at strength={canonical_strength(strength)}: {exc}"
            ) from exc
        curve.append(
            {
                "strength": canonical_strength(strength) if strength != IDENTITY else IDENTITY,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "gated_mae": gated,
                "n_points": metrics["n_points"],
                "status": status,
                "detail": detail,
            }
        )
    typed = []
    for item in curve:
        strength = IDENTITY if item["strength"] == IDENTITY else float(item["strength"])
        typed.append({**item, "strength": strength})
    chosen = pick_strength(typed)
    return curve, chosen


def _fixed_metrics(
    method: str,
    strength: Any,
    train_pred: np.ndarray,
    train_true: np.ndarray,
    eligible: np.ndarray,
    val_pred: np.ndarray,
    val_true: np.ndarray,
    value_scale: float,
) -> dict[str, Any]:
    _mark_candidate_score()
    scale, shift = fit_cells(train_pred, train_true, eligible, method, strength)
    applied = apply_affine(val_pred, scale, shift)
    metrics = score_arrays(applied, val_true, value_scale)
    return {
        "strength": canonical_strength(strength),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mape": metrics["mape"],
        "gated_mae": gated_mae(applied, val_true, eligible),
        "n_points": metrics["n_points"],
    }


def expected_task_keys(sources, seeds) -> list[str]:
    return [f"{source}:{int(seed)}" for source in sources for seed in seeds]


def scorable_errors(lock: dict[str, Any]) -> list[str]:
    """Structural completeness required before any test load."""
    errors: list[str] = []
    if not lock.get("scorable"):
        errors.append("lock is marked not scorable")
    if lock.get("missing"):
        errors.append(f"missing caches: {lock['missing']}")
    if lock.get("blocked"):
        errors.append(f"blocked: {lock['blocked']}")
    sources = list(lock.get("sources") or [])
    seeds = list(lock.get("prelocked_seeds") or [])
    arms = list(lock.get("arms") or [])
    candidates = list(lock.get("candidates") or [])
    if sources != list(SOURCES):
        errors.append(f"sources {sources} != pre-locked {list(SOURCES)}")
    if [int(s) for s in seeds] != [int(s) for s in PRELOCKED_SEEDS]:
        errors.append("prelocked seed set does not match the frozen protocol set")
    if arms != list(ARMS):
        errors.append("arm list is incomplete")
    expected_candidates = [canonical_strength(s) for s in CANDIDATES]
    if candidates != expected_candidates:
        errors.append("candidate set does not exactly match the frozen grid")
    keys = expected_task_keys(sources or SOURCES, seeds or PRELOCKED_SEEDS)
    stage1 = lock.get("stage1") or {}
    stage2 = lock.get("stage2") or {}
    stage3 = lock.get("stage3") or {}
    if not stage1:
        errors.append("stage1 is empty")
    if set(stage1) != set(keys):
        errors.append("stage1 task set is incomplete")
    if not stage2 or set(stage2) != set(keys):
        errors.append("stage2 is missing or incomplete")
    if not stage3:
        errors.append("stage3 is missing")
    for key in keys:
        arms_here = stage1.get(key) or {}
        if set(arms_here) != set(ARMS):
            errors.append(f"stage1 {key} is missing arms")
        else:
            for arm in ARMS:
                if "chosen" not in arms_here[arm]:
                    errors.append(f"stage1 {key} {arm} missing chosen strength")
                elif not _chosen_in_grid(arms_here[arm]["chosen"], expected_candidates):
                    errors.append(f"stage1 {key} {arm} chosen {arms_here[arm]['chosen']!r} is outside the candidate grid")
        s2 = stage2.get(key) or {}
        rolling = s2.get("rolling") or {}
        if set(rolling) != set(TUNABLE_ARMS):
            errors.append(f"stage2 {key} is missing tunable rolling tables")
        else:
            for arm in TUNABLE_ARMS:
                rounds = (rolling.get(arm) or {}).get("rounds") or []
                if len(rounds) != N_VAL_BLOCKS - 1:
                    errors.append(f"stage2 {key} {arm} does not have {N_VAL_BLOCKS - 1} complete rolling rounds")
                else:
                    for index, row in enumerate(rounds):
                        if int(row.get("round", -1)) != index or int(row.get("score_block", -1)) != index + 1:
                            errors.append(f"stage2 {key} {arm} round {index} is incomplete")
                        if "chosen" in row and not _chosen_in_grid(row["chosen"], expected_candidates):
                            errors.append(f"stage2 {key} {arm} rolling chosen is outside the candidate grid")
    for source in sources or SOURCES:
        item = stage3.get(source)
        if not item:
            errors.append(f"stage3 missing {source}")
        elif item.get("status") != "ok":
            errors.append(f"stage3 {source} is blocked or incomplete")
        elif set((item.get("arms") or {})) != set(TUNABLE_ARMS):
            errors.append(f"stage3 {source} is missing tunable methods")
        else:
            if [int(s) for s in (item.get("seeds") or [])] != [int(s) for s in seeds]:
                errors.append(f"stage3 {source} seed set does not match the pre-locked seeds")
            for arm in TUNABLE_ARMS:
                arm_item = (item.get("arms") or {}).get(arm) or {}
                if not _chosen_in_grid(arm_item.get("chosen"), expected_candidates):
                    errors.append(f"stage3 {source} {arm} chosen is outside the candidate grid")
                if [int(s) for s in (arm_item.get("seeds") or [])] != [int(s) for s in seeds]:
                    errors.append(f"stage3 {source} {arm} seed set does not match the pre-locked seeds")
    return errors


def _chosen_in_grid(chosen, expected_candidates) -> bool:
    if chosen is None:
        return False
    return canonical_strength(chosen) in {canonical_strength(item) for item in expected_candidates}


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
    """Complete every non-test selection, then write a scorable lock if and only if complete.

    Missing pre-locked seeds are recorded and cannot produce a scorable lock.
    They are never dropped after seeing scores. Test caches are not opened.
    """
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
    stage1: dict[str, Any] = {}
    stage2: dict[str, Any] = {}
    input_hashes: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    available_by_source: dict[str, list[int]] = {source: [] for source in requested_sources}
    reference_cell_ids = None
    loaded_trains: list[dict[str, Any]] = []
    loaded_vals: list[dict[str, Any]] = []
    loaded_tasks: list[dict[str, Any]] = []

    for source in requested_sources:
        for seed in requested_seeds:
            try:
                train = access.load(source, seed, "train")
                if reference_cell_ids is None:
                    reference_cell_ids = np.asarray(train["cell_ids"])
                    access.expected_cell_ids = reference_cell_ids
                    access.expected_cells = int(reference_cell_ids.size)
                val = access.load(source, seed, "val")
            except FileNotFoundError as exc:
                missing.append({"source": source, "seed": int(seed), "error": str(exc)})
                continue
            assert_task_split_consistency(train, val)
            eligible = _eligible(train, require_real_counts=not allow_synthetic)
            value_scale = float(val["value_scale"])
            blocks = four_contiguous_blocks(val["time_index"])
            rounds = rolling_rounds(blocks)
            loaded_trains.append(train)
            loaded_vals.append(val)
            manifests.append({"split": "train", **train["manifest"]})
            manifests.append({"split": "val", **val["manifest"]})
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
                    "val_blocks": [
                        {k: b[k] for k in ("block", "t_start", "t_end", "n_effective")}
                        for b in blocks
                    ],
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
                    "value_scale": value_scale,
                    "blocks": blocks,
                    "rounds": rounds,
                }
            )

    input_evidence = None
    evidence_errors: list[str] = []
    if not missing:
        assert_shared_dc_split(loaded_trains, split="train", compare_gate=True)
        assert_shared_dc_split(loaded_vals, split="val", compare_gate=False)
        provenances = {str(item["train"]["provenance_kind"]) for item in loaded_tasks} | {
            str(item["val"]["provenance_kind"]) for item in loaded_tasks
        }
        if PROVENANCE_DOCUMENTED in provenances:
            if provenances != {PROVENANCE_DOCUMENTED}:
                evidence_errors.append("mixed provenance is not allowed; documented_from_bundle cannot mix with recorded or synthetic")
            if not allow_documented_from_bundle:
                evidence_errors.append("documented_from_bundle requires explicit opt-in; no fallback")
            ev_path = Path(cache_dir) / "bundle_evidence.json"
            if not ev_path.is_file():
                evidence_errors.append("documented_from_bundle is missing bundle_evidence.json next to the staged cache")
            else:
                import json as _json
                input_evidence = _json.loads(ev_path.read_text(encoding="utf-8"))
                # Rematerialize from the current ZIP and current adapter. Never
                # compare the claimed evidence object to itself.
                evidence_errors.extend(
                    verify_lock_input_evidence(
                        {"input_protocol_id": INPUT_PROTOCOL_ID, "provenance_kind": PROVENANCE_DOCUMENTED, "input_evidence": input_evidence},
                        cache_dir=cache_dir,
                        zip_path=input_evidence.get("zip_path"),
                        root=root,
                    )
                )
                evidence_errors.extend(verify_staged_cache_against_evidence(cache_dir, input_evidence))
        # All pre-locked train/val files passed schema, split identity, and shared
        # DC truth/gate checks, and (if documented) the bundle evidence. Only now
        # may any candidate or fixed-arm scoring run.
        if evidence_errors:
            # Do not score candidates. Fall through to lock assembly with empty stages.
            loaded_tasks = []
        for item in loaded_tasks:
            source = item["source"]
            seed = item["seed"]
            key = item["key"]
            train = item["train"]
            val = item["val"]
            eligible = item["eligible"]
            value_scale = item["value_scale"]
            blocks = item["blocks"]
            rounds = item["rounds"]
            arm_results: dict[str, Any] = {}
            for arm in ARMS:
                method = ARM_METHOD[arm]
                if arm in FIXED_ARMS:
                    strength = ARM_FIXED_STRENGTH[arm]
                    scored = _fixed_metrics(
                        method,
                        strength,
                        train["prediction"],
                        train["target"],
                        eligible,
                        val["prediction"],
                        val["target"],
                        value_scale,
                    )
                    arm_results[arm] = {
                        "kind": "fixed",
                        "method": method,
                        "chosen": scored["strength"],
                        "val": scored,
                    }
                else:
                    curve, chosen = _curve_and_choice(
                        method,
                        train["prediction"],
                        train["target"],
                        eligible,
                        val["prediction"],
                        val["target"],
                        value_scale,
                    )
                    arm_results[arm] = {
                        "kind": "tuned",
                        "method": method,
                        "chosen": canonical_strength(chosen),
                        "curve": curve,
                        "chose_identity": canonical_strength(chosen) == IDENTITY,
                    }
            stage1[key] = arm_results

            block_tables: dict[str, Any] = {}
            rolling_tables: dict[str, Any] = {}
            for arm in TUNABLE_ARMS:
                method = ARM_METHOD[arm]
                per_block = []
                for block in blocks:
                    curve, chosen = _curve_and_choice(
                        method,
                        train["prediction"],
                        train["target"],
                        eligible,
                        val["prediction"],
                        val["target"],
                        value_scale,
                        positions=block["positions"],
                    )
                    per_block.append(
                        {
                            "block": block["block"],
                            "n_effective": block["n_effective"],
                            "best": canonical_strength(chosen),
                            "curve": curve,
                        }
                    )
                block_tables[arm] = per_block

                roll_rows = []
                scored_parts: list[tuple[float, int]] = []
                fixed_parts: dict[str, list[tuple[float, int]]] = {fixed_arm: [] for fixed_arm in FIXED_ARMS}
                for rnd in rounds:
                    curve, chosen = _curve_and_choice(
                        method,
                        train["prediction"],
                        train["target"],
                        eligible,
                        val["prediction"],
                        val["target"],
                        value_scale,
                        positions=rnd["prefix_positions"],
                    )
                    chosen_for_fit = IDENTITY if canonical_strength(chosen) == IDENTITY else float(chosen)
                    scale, shift = fit_cells(
                        train["prediction"],
                        train["target"],
                        eligible,
                        method,
                        chosen_for_fit,
                    )
                    next_pred = val["prediction"][rnd["score_positions"]]
                    next_true = val["target"][rnd["score_positions"]]
                    applied = apply_affine(next_pred, scale, shift)
                    metrics = score_arrays(applied, next_true, value_scale)
                    scored_parts.append((metrics["mae"], int(rnd["score_n"])))
                    fixed_on_next = {}
                    for fixed_arm in FIXED_ARMS:
                        fx = _fixed_metrics(
                            ARM_METHOD[fixed_arm],
                            ARM_FIXED_STRENGTH[fixed_arm],
                            train["prediction"],
                            train["target"],
                            eligible,
                            next_pred,
                            next_true,
                            value_scale,
                        )
                        fixed_on_next[fixed_arm] = fx
                        fixed_parts[fixed_arm].append((fx["mae"], int(rnd["score_n"])))
                    roll_rows.append(
                        {
                            "round": rnd["round"],
                            "prefix_blocks": rnd["prefix_blocks"],
                            "prefix_n": rnd["prefix_n"],
                            "score_block": rnd["score_block"],
                            "score_n": rnd["score_n"],
                            "chosen": canonical_strength(chosen),
                            "curve": curve,
                            "next_block": metrics,
                            "fixed_on_next": fixed_on_next,
                        }
                    )
                rolling_tables[arm] = {
                    "rounds": roll_rows,
                    "weighted_mae_last_three": weighted_block_mae(scored_parts),
                    "fixed_weighted_mae_last_three": {
                        arm_name: weighted_block_mae(parts) for arm_name, parts in fixed_parts.items()
                    },
                }
            stage2[key] = {
                "blocks": [{k: b[k] for k in ("block", "t_start", "t_end", "n_effective")} for b in blocks],
                "per_block_best": block_tables,
                "rolling": rolling_tables,
                "note": (
                    "Stage 2 rolling evaluates the selection rule on later val blocks. "
                    "It is not ranked against Stage 1 full-val selected scores. "
                    "Final test uses the Stage 1 full-val choice and must match Stage 1. "
                    "Fixed baselines use the same prefix budget and are summarized on the "
                    "last three scored blocks by effective observation counts."
                ),
            }
            available_by_source[source].append(int(seed))

    prelocked = list(requested_seeds)
    blocked: list[str] = []
    if missing:
        blocked.append("missing_caches")
    if evidence_errors:
        blocked.append("input_evidence")
        blocked.extend(evidence_errors)

    stage3: dict[str, Any] = {}
    for source in requested_sources:
        have = available_by_source[source]
        if set(have) != set(prelocked):
            stage3[source] = {
                "status": "blocked",
                "reason": "pre-locked seed set is incomplete; refusing to drop seeds after the fact",
                "prelocked_seeds": prelocked,
                "available_seeds": have,
            }
            if "incomplete_seed_set" not in blocked:
                blocked.append("incomplete_seed_set")
            continue
        methods: dict[str, Any] = {}
        for arm in TUNABLE_ARMS:
            mean_curve = []
            for strength in CANDIDATES:
                label = canonical_strength(strength)
                losses = []
                for seed in prelocked:
                    curve = stage1[f"{source}:{seed}"][arm]["curve"]
                    match = [item for item in curve if item["strength"] == label]
                    if len(match) != 1:
                        raise ValueError(f"missing curve point {label} for {source} seed {seed} {arm}")
                    losses.append(float(match[0]["mae"]))
                mean_mae = float(sum(losses) / len(losses))
                mean_curve.append({"strength": label, "mae": mean_mae, "per_seed_mae": losses, "n_seeds": len(losses)})
            typed = [
                {"strength": IDENTITY if item["strength"] == IDENTITY else float(item["strength"]), "mae": item["mae"]}
                for item in mean_curve
            ]
            chosen = pick_strength(typed)
            methods[arm] = {
                "chosen": canonical_strength(chosen),
                "curve": mean_curve,
                "seeds": prelocked,
                "note": "shared strength only; per-seed a,b/c are still fit on that seed's 72h train window",
            }
        stage3[source] = {"status": "ok", "seeds": prelocked, "arms": methods}

    binding = binding_payload(root)
    lock_path = Path(lock_path)
    lock: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "input_protocol_id": INPUT_PROTOCOL_ID if input_evidence is not None else None,
        "provenance_kind": PROVENANCE_DOCUMENTED if input_evidence is not None else None,
        "input_evidence": input_evidence,
        "label_budget": LABEL_BUDGET,
        "sources": list(requested_sources),
        "prelocked_seeds": prelocked,
        "seed_freeze_id": seed_freeze_id,
        "arms": list(ARMS),
        "candidates": [canonical_strength(s) for s in CANDIDATES],
        "binding": binding,
        "input_hashes": input_hashes,
        "manifests": manifests,
        "missing": missing,
        "blocked": blocked,
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "note": (
            "All non-test selections are frozen in this file. Test scoring must use "
            "score_test after verifying this hash and rematerializing plan, implementation, "
            "config, and train/val manifests, in order Stage1 -> Stage2 -> Stage3, "
            "and must not revise any choice. This file is not mutated after scoring; "
            "exclusive run directories record whether test was opened."
        ),
    }
    completeness = scorable_errors({**lock, "scorable": True})
    # scorable_errors treats scorable=False as an error, so evaluate structure with a
    # temporary True flag, then also require no missing/blocked.
    lock["scorable"] = not completeness
    lock["scorable_errors"] = completeness
    lock["lock_hash"] = canonical_hash({k: v for k, v in lock.items() if k != "lock_hash"})
    if lock["scorable"]:
        if lock_path.exists():
            raise FileExistsError(f"refusing to overwrite existing lock {lock_path}")
        write_json(lock_path, lock)
    else:
        blocked_path = lock_path.with_name(lock_path.stem + ".blocked.json")
        write_json(blocked_path, lock)
        lock["_written_path"] = str(blocked_path)
        lock["_scorable_lock_written"] = False
    return lock
