"""PR20 hourly-bundle adapter: documented_from_bundle input mode.

This module never changes recorded provenance rules and never invents a
full git commit or a rehashed checkpoint. Arrays are read-only: they are
not rewritten, reordered, renormalized, or clipped. The bundled predict
script is read as text and is never executed.

documented_from_bundle is an explicit opt-in. There is no fallback to
recorded or synthetic provenance.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from src.val_select.io_cache import canonical_hash, sha256_file, sidecar_path, write_json
from src.val_select.protocol import (
    CHECKPOINT_HASH_STATUS,
    DOCUMENTED_MAX_VAL,
    DOCUMENTED_MIN_VAL,
    DOCUMENTED_PREDICT_SCRIPT_FORM,
    DOCUMENTED_PREDICT_SCRIPT_REVISION,
    DOCUMENTED_TRAIN_Y_VS_CLIP_RAW_MAXABS,
    DOCUMENTED_VALUE_SCALE,
    EXPECTED_TEST_HOURS,
    EXPECTED_TEST_START,
    EXPECTED_VAL_HOURS,
    EXPECTED_VAL_START,
    FINE_TUNE_COMMIT_STATUS,
    GATE,
    INPUT_PROTOCOL_ID,
    LABEL_START_TRAIN,
    NORM_FITTED_ON,
    NORM_PERCENTILE,
    N_TRAIN_HOURS,
    PRELOCKED_SEEDS,
    PROVENANCE_DOCUMENTED,
    SOURCES,
    TIME_INDEX_ORIGIN_DOCUMENTED,
    UNIT_DC_CLIPPED_COUNT,
    VAL_TEST_RANGE_STATUS,
)
from src.val_select.schema import SchemaError, expected_time_index
from src.val_select.snapshot import implementation_snapshot, sha256_path

# Half-open documented ranges from the bundle README.
DOCUMENTED_RANGES = {
    "train": (LABEL_START_TRAIN, LABEL_START_TRAIN + N_TRAIN_HOURS),
    "val": (EXPECTED_VAL_START, EXPECTED_VAL_START + EXPECTED_VAL_HOURS),
    "test": (EXPECTED_TEST_START, EXPECTED_TEST_START + EXPECTED_TEST_HOURS),
}

RANGE_RE = re.compile(
    r"(train|val|test)\s+第 t 行：标签小时\s+(\d+)\s*\+\s*t.*?即\s*\[(\d+),\s*(\d+)\)",
    re.IGNORECASE,
)
LABEL_START_RE = re.compile(r"^LABEL_START\s*=\s*(\d+)\s*$", re.MULTILINE)
N_HOURS_RE = re.compile(r"^N_HOURS\s*=\s*(\d+)\s*$", re.MULTILINE)
SHA256_LINE_RE = re.compile(r"^([0-9a-fA-F]{64})\s+(\S+)\s*$")

# Trust anchor for the accepted PR20 hourly bundle. Member hashes are taken from a
# ZIP whose container sha256 equals this value; an extracted tree plus a rewritten
# SHA256SUMS cannot impersonate the package.
ACCEPTED_PR20_ZIP_SHA256 = "99594eaa2b200fe4b8be51823e5716d6152de3186ccf05afb21b4bba81ef6845"

OPTIONAL_SOURCE_MEMBERS = (
    "source/configs/default.yaml",
    "source/configs/finetune_ny2dc.yaml",
    "source/configs/finetune_chi2dc.yaml",
)

REQUIRED_SIDECAR_FIELD_SOURCES = (
    "prediction",
    "target",
    "value_scale",
    "min_val",
    "unit",
    "time_index",
    "source_seed_split",
    "generator_commit",
    "fine_tune_commit",
    "checkpoint_sha256",
    "norm_fitted_on",
    "raw_train_counts",
    "eligible",
    "cell_ids",
)


class BundleAdapterError(SchemaError):
    """Bundle documentation or evidence is inconsistent or incomplete."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(values).tobytes())


def parse_readme_ranges(readme_text: str) -> dict[str, tuple[int, int]]:
    found: dict[str, tuple[int, int]] = {}
    for match in RANGE_RE.finditer(readme_text):
        split, start_plus, start, end = match.group(1).lower(), int(match.group(2)), int(match.group(3)), int(match.group(4))
        if start_plus != start:
            raise BundleAdapterError(
                f"README {split} start {start} disagrees with the '{start_plus} + t' formula"
            )
        found[split] = (start, end)
    if set(found) != {"train", "val", "test"}:
        raise BundleAdapterError(f"README is missing documented hour ranges, found {sorted(found)}")
    return found


def parse_script_train_slice(script_text: str) -> tuple[int, int, bool]:
    """Read-only parse of the bundled predict script. Never exec the file.

    Returns (LABEL_START, N_HOURS, imports_prepare_city_windows).
    The prepare_city_windows import is a repo dependency that is not defined
    in the bundled script, so val/test starts cannot be proved from this file
    alone.
    """
    start_match = LABEL_START_RE.search(script_text)
    hours_match = N_HOURS_RE.search(script_text)
    if not start_match or not hours_match:
        raise BundleAdapterError("bundled script is missing LABEL_START or N_HOURS")
    imports_prep = "prepare_city_windows" in script_text
    defines_prep = re.search(r"^def prepare_city_windows\b", script_text, re.MULTILINE) is not None
    return int(start_match.group(1)), int(hours_match.group(1)), imports_prep and not defines_prep


def documented_time_index(split: str) -> np.ndarray:
    start, end = DOCUMENTED_RANGES[split]
    return np.arange(start, end, dtype=np.int64)


def assert_documented_hour_chain(readme_text: str, script_text: str) -> dict[str, Any]:
    """Compare README ranges with the bundled script and protocol defaults.

    Train hours are supported by LABEL_START/N_HOURS in the script (read as
    text). Val/test starts are documented in the README and match the protocol
    defaults, but the script only *imports* prepare_city_windows — that helper
    is not in the bundle — so this does not independently prove the val start
    from executable generation-chain code.
    """
    readme_ranges = parse_readme_ranges(readme_text)
    label_start, n_hours, missing_prep = parse_script_train_slice(script_text)
    errors: list[str] = []
    for split, expected in DOCUMENTED_RANGES.items():
        if readme_ranges[split] != expected:
            errors.append(
                f"README {split} range {readme_ranges[split]} contradicts documented/protocol {expected}"
            )
    if (label_start, label_start + n_hours) != DOCUMENTED_RANGES["train"]:
        errors.append(
            f"script LABEL_START={label_start} N_HOURS={n_hours} contradicts documented train {DOCUMENTED_RANGES['train']}"
        )
    for split in ("train", "val", "test"):
        if not np.array_equal(documented_time_index(split), expected_time_index(split)):
            errors.append(f"documented {split} time_index disagrees with protocol expected_time_index")
    if errors:
        raise BundleAdapterError("documented hour chain contradiction: " + "; ".join(errors))
    return {
        "time_index_origin": TIME_INDEX_ORIGIN_DOCUMENTED,
        "readme_ranges": {k: list(v) for k, v in readme_ranges.items()},
        "script_label_start": label_start,
        "script_n_hours": n_hours,
        "script_imports_prepare_city_windows_but_does_not_define_it": missing_prep,
        "val_test_start_code_independent_proof": False,
        "val_test_range_status": VAL_TEST_RANGE_STATUS,
        "note": (
            "Train slice is supported by the bundled script constants LABEL_START/N_HOURS "
            "(text only, script not executed). Val/test starts are documented in the README "
            "and are compatible with protocol defaults. Because prepare_city_windows is an "
            "imported repo dependency not shipped in the bundle, this is not an independent "
            "code proof of the actual val start."
        ),
    }


def load_declared_checksums(sums_text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in sums_text.splitlines():
        match = SHA256_LINE_RE.match(line.strip())
        if match:
            out[match.group(2).replace("\\", "/")] = match.group(1).lower()
    return out


def required_nontest_members() -> tuple[str, ...]:
    members = []
    for source in SOURCES:
        for seed in PRELOCKED_SEEDS:
            for split in ("train", "val"):
                members.append(f"predictions/{split}/{source}_seed{seed}_{split}.npz")
    members.extend(
        [
            "README.txt",
            "SHA256SUMS.txt",
            "source/19_cell_affine_v2_at_7a04ca4.py",
            "source/checkpoint_sha256.txt",
            "gate/norm_and_gate.json",
            "gate/raw_pickup_valid_5832_5904.npy",
            "gate/raw_total_valid.npy",
        ]
    )
    return tuple(members)


def required_test_declaration_members() -> tuple[str, ...]:
    return tuple(
        f"predictions/test/{source}_seed{seed}_test.npz"
        for source in SOURCES
        for seed in PRELOCKED_SEEDS
    )


def required_sha256sums_entries() -> tuple[str, ...]:
    # SHA256SUMS does not list itself. Test entries are declared from the sums
    # text; their arrays are not read from the ZIP at adapt/select time.
    return tuple(m for m in required_nontest_members() if m != "SHA256SUMS.txt") + required_test_declaration_members()


def resolve_zip_member(zf: zipfile.ZipFile, rel: str) -> str | None:
    rel = rel.replace("\\", "/")
    candidates = (rel, f"pr20_hourly_bundle/{rel}")
    names = {name.replace("\\", "/"): name for name in zf.namelist()}
    for cand in candidates:
        if cand in names:
            return names[cand]
    return None


def sha256_zip_member_bytes(zip_path: Path, rel: str) -> str:
    """Hash one non-test ZIP member from the container. Does not write to disk."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        name = resolve_zip_member(zf, rel)
        if name is None:
            raise BundleAdapterError(f"ZIP is missing required non-test member {rel}")
        digest = hashlib.sha256()
        with zf.open(name, "r") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def read_zip_member_text(zip_path: Path, rel: str) -> str:
    with zipfile.ZipFile(zip_path, "r") as zf:
        name = resolve_zip_member(zf, rel)
        if name is None:
            raise BundleAdapterError(f"ZIP is missing required member {rel}")
        with zf.open(name, "r") as handle:
            return handle.read().decode("utf-8")


def evidence_content(evidence: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in evidence.items() if k != "evidence_hash"}


def compute_evidence_hash(evidence: dict[str, Any]) -> str:
    return canonical_hash(evidence_content(evidence))


def rematerialize_zip_trust_anchors(zip_path: Path) -> dict[str, Any]:
    """Recompute trust anchors from the current ZIP container.

    Hashes the ZIP bytes and every required non-test whitelist member inside it.
    Test members are not hashed or extracted; only SHA256SUMS text is read for
    their declared checksums.
    """
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise BundleAdapterError(f"ZIP trust anchor is missing: {zip_path}")
    zip_sha = sha256_file(zip_path)
    sums_text = read_zip_member_text(zip_path, "SHA256SUMS.txt")
    declared = load_declared_checksums(sums_text)
    missing = [rel for rel in required_sha256sums_entries() if rel not in declared]
    if missing:
        raise BundleAdapterError("SHA256SUMS is missing required entries: " + ", ".join(missing))

    source_files: dict[str, str] = {}
    verified: dict[str, str] = {}
    member_hashes: dict[str, str] = {}
    for rel in required_nontest_members():
        got = sha256_zip_member_bytes(zip_path, rel)
        member_hashes[rel] = got
        if rel == "SHA256SUMS.txt":
            source_files[rel] = got
            continue
        if declared[rel] != got:
            raise BundleAdapterError(f"ZIP member {rel} sha256 {got} does not match SHA256SUMS {declared[rel]}")
        if rel.startswith("predictions/"):
            verified[rel] = got
        else:
            source_files[rel] = got
    for rel in OPTIONAL_SOURCE_MEMBERS:
        try:
            got = sha256_zip_member_bytes(zip_path, rel)
        except BundleAdapterError:
            continue
        member_hashes[rel] = got
        source_files[rel] = got
        if rel in declared and declared[rel] != got:
            raise BundleAdapterError(f"ZIP member {rel} sha256 {got} does not match SHA256SUMS {declared[rel]}")

    declared_test = {rel: declared[rel] for rel in required_test_declaration_members()}
    return {
        "zip_path": str(zip_path),
        "zip_sha256": zip_sha,
        "source_files": source_files,
        "verified_train_val_members": verified,
        "declared_test_sha256": declared_test,
        "member_hashes": member_hashes,
        "is_accepted_pr20_zip": zip_sha == ACCEPTED_PR20_ZIP_SHA256,
    }


def zip_member_inventory(zip_path: Path) -> list[dict[str, Any]]:
    """Central-directory metadata only. Does not unpack arrays."""
    rows = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            rows.append({
                "filename": info.filename,
                "file_size": info.file_size,
                "compress_size": info.compress_size,
                "CRC": f"{info.CRC:08x}",
                "is_dir": info.is_dir(),
            })
    return rows


def decode_utf8_sig_text(data: bytes) -> str:
    """Decode source-document bytes with utf-8-sig (UTF-8 BOM removed if present).

    Newlines are not rewritten here; callers use str.splitlines() so LF, CRLF,
    and CR are equivalent. This does not change file/ZIP member byte hashes.
    """
    return data.decode("utf-8-sig")


def parse_claimed_checkpoint_hashes(text: str) -> dict[str, str]:
    """Parse claimed weight hashes. The 64-character hex rule is unchanged.

    Text should come from decode_utf8_sig_text. A leftover leading U+FEFF
    (plain utf-8 decode of a BOM file) is stripped so BOM/no-BOM × LF/CRLF
    are equivalent. A 65-character token is still rejected.
    """
    if text.startswith("﻿"):
        text = text[1:]
    out: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            out[parts[1]] = parts[0].lower()
    return out


def cell_ids_from_rows_cols(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return np.array([f"{int(r)}:{int(c)}" for r, c in zip(rows.tolist(), cols.tolist())], dtype=object)


def _copy_unmodified(src: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    # Refuse in-place mutation: the copy must stay byte-identical to the source.
    if sha256_file(src) != sha256_file(dest):
        raise BundleAdapterError(f"copy of {src} is not byte-identical")
    return sha256_file(dest)


def adapt_pr20_bundle(
    *,
    zip_path: Path,
    extracted_root: Path,
    output_dir: Path,
    allow_documented_from_bundle: bool,
    open_test_arrays: bool = False,
    expect_clip_residual: bool = True,
) -> dict[str, Any]:
    """Build documented_from_bundle sidecars and an evidence manifest.

    Train/val NPZ files are copied unmodified. Test members are not extracted
    and arrays are not opened. Test checksums are recorded as declared from
    SHA256SUMS / the ZIP central directory, pending score-time verification.
    """
    if not allow_documented_from_bundle:
        raise BundleAdapterError("documented_from_bundle requires explicit opt-in; no fallback")
    if open_test_arrays:
        raise BundleAdapterError("adapter is not allowed to open real test arrays")

    zip_path = Path(zip_path)
    extracted_root = Path(extracted_root)
    # Accept either the bundle root or its parent.
    if (extracted_root / "README.txt").is_file():
        bundle = extracted_root
    elif (extracted_root / "pr20_hourly_bundle" / "README.txt").is_file():
        bundle = extracted_root / "pr20_hourly_bundle"
    else:
        raise BundleAdapterError(f"README.txt not found under {extracted_root}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"

    readme_path = bundle / "README.txt"
    script_path = bundle / "source" / "19_cell_affine_v2_at_7a04ca4.py"
    sums_path = bundle / "SHA256SUMS.txt"
    gate_json_path = bundle / "gate" / "norm_and_gate.json"
    gate_raw_path = bundle / "gate" / "raw_pickup_valid_5832_5904.npy"
    gate_total_path = bundle / "gate" / "raw_total_valid.npy"
    ckpt_path = bundle / "source" / "checkpoint_sha256.txt"
    for required in (readme_path, script_path, sums_path, gate_json_path, gate_raw_path, gate_total_path, ckpt_path, zip_path):
        if not Path(required).is_file():
            raise BundleAdapterError(f"required bundle file missing: {required}")

    # The ZIP container is the trust anchor. Extracted files must match ZIP
    # members; a rewritten SHA256SUMS next to a mutated tree cannot impersonate
    # the original package.
    anchors = rematerialize_zip_trust_anchors(zip_path)
    zip_sha = anchors["zip_sha256"]
    for rel, expected in anchors["member_hashes"].items():
        extracted = bundle / rel
        if not extracted.is_file():
            raise BundleAdapterError(f"extracted tree is missing whitelist member {rel}")
        got = sha256_file(extracted)
        if got != expected:
            raise BundleAdapterError(
                f"extracted {rel} sha256 {got} does not match ZIP member {expected}; "
                "the extracted tree is not the trust anchor"
            )

    readme_text = readme_path.read_text(encoding="utf-8")
    script_text = script_path.read_text(encoding="utf-8")
    hour_chain = assert_documented_hour_chain(readme_text, script_text)

    declared = load_declared_checksums(read_zip_member_text(zip_path, "SHA256SUMS.txt"))
    zip_members = zip_member_inventory(zip_path)
    gate_meta = json.loads(gate_json_path.read_text(encoding="utf-8"))
    claimed_ckpts = parse_claimed_checkpoint_hashes(decode_utf8_sig_text(ckpt_path.read_bytes()))

    documented_scale = float(gate_meta.get("value_scale", DOCUMENTED_VALUE_SCALE))
    documented_min = float(gate_meta.get("min_val", DOCUMENTED_MIN_VAL))
    documented_max = float(gate_meta.get("max_val", DOCUMENTED_MAX_VAL))
    if documented_scale != documented_max - documented_min:
        raise BundleAdapterError(
            f"value_scale {documented_scale} != documented max-min {documented_max}-{documented_min}; "
            "unit/scale semantics come from metadata, not from observing the value range"
        )
    if documented_scale != float(DOCUMENTED_VALUE_SCALE) or documented_min != float(DOCUMENTED_MIN_VAL):
        raise BundleAdapterError("bundle norm metadata does not match the frozen documented scale/min")

    raw_pickup = np.load(gate_raw_path, allow_pickle=False)
    raw_total = np.load(gate_total_path, allow_pickle=False)
    if raw_pickup.ndim != 2 or raw_pickup.shape[0] != N_TRAIN_HOURS:
        raise BundleAdapterError(f"gate raw_pickup shape {raw_pickup.shape} is not ({N_TRAIN_HOURS}, cells)")
    recomputed_total = raw_pickup.sum(axis=0)
    if not np.array_equal(recomputed_total, raw_total):
        raise BundleAdapterError("raw_total_valid.npy does not match sum of raw_pickup")
    eligible_from_raw = recomputed_total >= float(GATE)

    # Geometry is taken from the first verified train file and inherited by val.
    reference_train_rel = f"predictions/train/{SOURCES[0]}_seed{PRELOCKED_SEEDS[0]}_train.npz"
    reference_train_path = bundle / reference_train_rel
    with np.load(reference_train_path, allow_pickle=False) as blob:
        ref_rows = np.asarray(blob["rows"])
        ref_cols = np.asarray(blob["cols"])
        ref_eligible = np.asarray(blob["eligible"], dtype=bool)
        ref_y = np.asarray(blob["target"], dtype=np.float64)
    if not np.array_equal(ref_eligible, eligible_from_raw):
        raise BundleAdapterError("reference train eligible does not match recomputed raw gate")
    cell_ids = cell_ids_from_rows_cols(ref_rows, ref_cols)
    clip_raw = np.clip(raw_pickup, documented_min, documented_max)
    y_vs_clip = float(np.max(np.abs(ref_y - clip_raw)))
    if expect_clip_residual:
        if y_vs_clip != float(DOCUMENTED_TRAIN_Y_VS_CLIP_RAW_MAXABS):
            raise BundleAdapterError(
                f"train y vs clip(raw,0,{documented_max}) maxabs={y_vs_clip} "
                f"!= documented {DOCUMENTED_TRAIN_Y_VS_CLIP_RAW_MAXABS}; "
                "these arrays are not byte-identical"
            )
    byte_identical_to_clip = bool(np.array_equal(ref_y, clip_raw))
    if byte_identical_to_clip:
        raise BundleAdapterError("train y must not be treated as byte-identical to clip(raw)")

    source_files = {
        "README.txt": sha256_file(readme_path),
        "SHA256SUMS.txt": sha256_file(sums_path),
        "source/19_cell_affine_v2_at_7a04ca4.py": sha256_file(script_path),
        "source/checkpoint_sha256.txt": sha256_file(ckpt_path),
        "gate/norm_and_gate.json": sha256_file(gate_json_path),
        "gate/raw_pickup_valid_5832_5904.npy": sha256_file(gate_raw_path),
        "gate/raw_total_valid.npy": sha256_file(gate_total_path),
    }
    # Configs/logs if present (hashed; test metrics inside logs are not expanded).
    for rel in (
        "source/configs/default.yaml",
        "source/configs/finetune_ny2dc.yaml",
        "source/configs/finetune_chi2dc.yaml",
    ):
        path = bundle / rel
        if path.is_file():
            source_files[rel] = sha256_file(path)

    verified_members: dict[str, str] = {}
    sidecars: list[dict[str, Any]] = []
    split_y_hashes: dict[str, str] = {}

    for source in SOURCES:
        for seed in PRELOCKED_SEEDS:
            for split in ("train", "val"):
                rel = f"predictions/{split}/{source}_seed{seed}_{split}.npz"
                src = bundle / rel
                if not src.is_file():
                    raise BundleAdapterError(f"missing train/val member {rel}")
                got = sha256_file(src)
                if rel not in declared:
                    raise BundleAdapterError(f"SHA256SUMS is missing required entry for {rel}")
                declared_hash = declared[rel]
                if declared_hash != got:
                    raise BundleAdapterError(f"SHA256SUMS mismatch for {rel}")
                if anchors["verified_train_val_members"].get(rel) != got:
                    raise BundleAdapterError(f"extracted {rel} does not match the ZIP whitelist member hash")
                verified_members[rel] = got
                dest = cache_dir / f"{source}_seed{seed}_{split}.npz"
                copied = _copy_unmodified(src, dest)
                if copied != got:
                    raise BundleAdapterError(f"copy hash changed for {rel}")

                with np.load(src, allow_pickle=False) as blob:
                    keys = list(blob.files)
                    prediction = np.asarray(blob["prediction"], dtype=np.float64)
                    target = np.asarray(blob["target"], dtype=np.float64)
                    value_scale = float(np.asarray(blob["value_scale"]))
                    train_extra = {}
                    if split == "train":
                        train_extra = {
                            "eligible": np.asarray(blob["eligible"], dtype=bool),
                            "rows": np.asarray(blob["rows"]),
                            "cols": np.asarray(blob["cols"]),
                        }

                if value_scale != documented_scale:
                    raise BundleAdapterError(
                        f"{rel} value_scale={value_scale} != documented metadata scale {documented_scale}"
                    )
                if split == "train":
                    if not np.array_equal(train_extra["eligible"], eligible_from_raw):
                        raise BundleAdapterError(f"{rel} eligible disagrees with recomputed raw gate")
                    if not np.array_equal(train_extra["rows"], ref_rows) or not np.array_equal(train_extra["cols"], ref_cols):
                        raise BundleAdapterError(f"{rel} rows/cols disagree with the reference train geometry")
                    if not np.array_equal(target, ref_y):
                        raise BundleAdapterError(f"{rel} train target is not byte-identical across tasks")
                else:
                    # val geometry is inherited; val has no rows/cols/eligible in the native npz.
                    if "eligible" in keys or "rows" in keys or "cols" in keys:
                        raise BundleAdapterError(f"{rel} unexpectedly contains train geometry keys")

                y_hash = _sha256_array(target)
                split_y_hashes.setdefault(split, y_hash)
                if split_y_hashes[split] != y_hash:
                    raise BundleAdapterError(f"{split} targets are not byte-identical across tasks ({rel})")

                time_index = documented_time_index(split)
                ckpt_name = {
                    ("ny", 42): "finetune_ny2dc.pt",
                    ("ny", 43): "finetune_ny2dc_seed43.pt",
                    ("ny", 44): "finetune_ny2dc_seed44.pt",
                    ("chi", 42): "finetune_chi2dc.pt",
                    ("chi", 43): "finetune_chi2dc_seed43.pt",
                    ("chi", 44): "finetune_chi2dc_seed44.pt",
                }[(source, int(seed))]
                claimed_hash = claimed_ckpts.get(ckpt_name)
                if not claimed_hash:
                    raise BundleAdapterError(f"no claimed checkpoint hash for {ckpt_name}")
                config_id = "configs/finetune_ny2dc.yaml" if source == "ny" else "configs/finetune_chi2dc.yaml"

                field_sources = {
                    "prediction": {"layer": "observed", "member": rel, "member_sha256": got},
                    "target": {"layer": "observed", "member": rel, "member_sha256": got, "array_sha256": y_hash},
                    "value_scale": {
                        "layer": "observed_and_documented",
                        "npz_value": value_scale,
                        "metadata": "gate/norm_and_gate.json",
                        "metadata_sha256": source_files["gate/norm_and_gate.json"],
                        "script_support": "value_scale = max_val - min_val; denorm is pred * value_scale + min_val (script read as text, not executed)",
                    },
                    "min_val": {"layer": "documented", "value": documented_min, "metadata": "gate/norm_and_gate.json"},
                    "unit": {
                        "layer": "documented",
                        "value": UNIT_DC_CLIPPED_COUNT,
                        "from": "README + script denorm of percentile-clipped counts; not inferred from the numeric range alone",
                    },
                    "time_index": {
                        "layer": "documented",
                        "origin": TIME_INDEX_ORIGIN_DOCUMENTED,
                        "readme": "README.txt",
                        "readme_sha256": source_files["README.txt"],
                        "script_support": (
                            "LABEL_START/N_HOURS for the train slice only. "
                            "Val/test starts are not independently proved by the bundled script "
                            "because prepare_city_windows is an imported dependency not defined in it."
                        ),
                    },
                    "source_seed_split": {
                        "layer": "documented",
                        "note": "declared in the sidecar from the requested task + README naming; the filename is not self-proving",
                    },
                    "generator_commit": {
                        "layer": "documented",
                        "value": DOCUMENTED_PREDICT_SCRIPT_REVISION,
                        "form": DOCUMENTED_PREDICT_SCRIPT_FORM,
                    },
                    "fine_tune_commit": {"layer": "unknown", "status": FINE_TUNE_COMMIT_STATUS},
                    "checkpoint_sha256": {
                        "layer": "claimed",
                        "status": CHECKPOINT_HASH_STATUS,
                        "source": "source/checkpoint_sha256.txt",
                        "source_sha256": source_files["source/checkpoint_sha256.txt"],
                        "path": ckpt_name,
                        "value": claimed_hash,
                        "rehash": "unknown_files_absent",
                    },
                    "norm_fitted_on": {
                        "layer": "documented",
                        "value": NORM_FITTED_ON,
                        "independent_recompute": "unknown_full_raw_absent",
                    },
                    "raw_train_counts": {
                        "layer": "observed",
                        "member": "gate/raw_pickup_valid_5832_5904.npy",
                        "member_sha256": source_files["gate/raw_pickup_valid_5832_5904.npy"],
                        "recomputed_total_matches": True,
                        "note": "packager-extracted from local TaxiDC_pickup.npy; not stored in the predict npz",
                    },
                    "eligible": {
                        "layer": "observed",
                        "from": "train npz + recomputed raw_total >= 150",
                    },
                    "cell_ids": {
                        "layer": "observed_inherited" if split == "val" else "observed",
                        "from_train_member": reference_train_rel,
                        "from_train_sha256": verified_members.get(reference_train_rel, sha256_file(reference_train_path)),
                    },
                }

                sidecar = {
                    "source": source,
                    "seed": int(seed),
                    "split": split,
                    "unit": UNIT_DC_CLIPPED_COUNT,
                    "cell_ids": cell_ids.tolist(),
                    "time_index": time_index.tolist(),
                    "time_index_origin": TIME_INDEX_ORIGIN_DOCUMENTED,
                    "value_scale": documented_scale,
                    "min_val": documented_min,
                    "documented_max_val": documented_max,
                    "norm_percentile": NORM_PERCENTILE,
                    "norm_fitted_on": NORM_FITTED_ON,
                    "checkpoint_path": ckpt_name,
                    "checkpoint_sha256": claimed_hash,
                    "checkpoint_sha256_status": CHECKPOINT_HASH_STATUS,
                    "config_id": config_id,
                    "generator_commit": DOCUMENTED_PREDICT_SCRIPT_REVISION,
                    "generator_commit_form": DOCUMENTED_PREDICT_SCRIPT_FORM,
                    "fine_tune_commit_status": FINE_TUNE_COMMIT_STATUS,
                    "provenance_kind": PROVENANCE_DOCUMENTED,
                    "input_protocol_id": INPUT_PROTOCOL_ID,
                    "field_sources": field_sources,
                    "native_npz_keys": keys,
                    "native_member": rel,
                    "native_member_sha256": got,
                    "original_arrays_unmodified": True,
                    "gate_threshold": float(GATE),
                }
                if split == "train":
                    sidecar["eligible"] = train_extra["eligible"].astype(bool).tolist()
                    sidecar["raw_train_counts"] = raw_total.astype(np.float64).tolist()
                    sidecar["train_y_vs_clip_raw_maxabs"] = y_vs_clip
                    sidecar["train_y_byte_identical_to_clip_raw"] = False
                sidecar_path(dest).write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                sidecars.append({"path": str(dest), "sidecar": str(sidecar_path(dest)), "member": rel, "sha256": got})

    # Test members: declared checksums only. Do not extract or load arrays.
    declared_test: dict[str, Any] = {}
    for source in SOURCES:
        for seed in PRELOCKED_SEEDS:
            rel = f"predictions/test/{source}_seed{seed}_test.npz"
            zip_name = f"pr20_hourly_bundle/{rel}"
            member_meta = next((m for m in zip_members if m["filename"] in {rel, zip_name, "pr20_hourly_bundle/" + rel}), None)
            if rel not in declared:
                raise BundleAdapterError(f"SHA256SUMS is missing required test declaration for {rel}")
            declared_test[rel] = {
                "declared_sha256": declared[rel],
                "declaration_status": "declared_pending_score_verification",
                "zip_central_directory": member_meta,
                "arrays_opened": False,
                "extracted": False,
            }

    evidence = {
        "input_protocol_id": INPUT_PROTOCOL_ID,
        "provenance_kind": PROVENANCE_DOCUMENTED,
        "opt_in": True,
        "original_arrays_unmodified": True,
        "zip_path": str(zip_path),
        "zip_sha256": zip_sha,
        "extracted_bundle": str(bundle),
        "hour_chain": hour_chain,
        "source_files": source_files,
        "verified_train_val_members": verified_members,
        "declared_test_checksums": declared_test,
        "gate": {
            "raw_pickup_sha256": source_files["gate/raw_pickup_valid_5832_5904.npy"],
            "raw_total_sha256": source_files["gate/raw_total_valid.npy"],
            "norm_json_sha256": source_files["gate/norm_and_gate.json"],
            "recomputed_n_gate": int(eligible_from_raw.sum()),
            "raw_total_min": float(raw_total.min()),
            "raw_total_max": float(raw_total.max()),
        },
        "shared_truth": {
            "train_target_sha256": split_y_hashes.get("train"),
            "val_target_sha256": split_y_hashes.get("val"),
            "train_targets_byte_identical": True,
            "val_targets_byte_identical": True,
            "train_y_vs_clip_raw_maxabs": y_vs_clip,
            "train_y_byte_identical_to_clip_raw": False,
            "note": (
                "Same-split targets are byte-identical across the six tasks. "
                "Train y vs clip(raw, 0, max_val) is NOT byte-identical "
                f"(maxabs={y_vs_clip})."
            ),
        },
        "geometry_inheritance": {
            "val_inherits_from_train_member": reference_train_rel,
            "val_inherits_from_train_sha256": verified_members[reference_train_rel],
            "rows_range": [int(ref_rows.min()), int(ref_rows.max())],
            "cols_range": [int(ref_cols.min()), int(ref_cols.max())],
        },
        "unknown": {
            "fine_tune_commit": FINE_TUNE_COMMIT_STATUS,
            "checkpoint_rehash": "unknown_files_absent",
            "full_raw_norm_recompute": "unknown_full_raw_absent",
            "val_start_independent_code_proof": False,
        },
        "adapter_implementation": implementation_snapshot(),
        "sidecars": sidecars,
        "cache_dir": str(cache_dir),
    }
    evidence["evidence_hash"] = canonical_hash({k: v for k, v in evidence.items() if k != "evidence_hash"})
    evidence_path = output_dir / "bundle_evidence.json"
    write_json(evidence_path, evidence)
    # Pointer next to the staged cache for lock_all_selections to pick up.
    write_json(cache_dir / "bundle_evidence.json", evidence)
    return evidence



def verify_staged_cache_against_evidence(cache_dir: Path, evidence: dict[str, Any]) -> list[str]:
    """Re-hash staged train/val copies against the verified member map. Never opens test."""
    from src.val_select.io_cache import cache_path
    from src.val_select.protocol import PRELOCKED_SEEDS, SOURCES

    errors: list[str] = []
    verified = evidence.get("verified_train_val_members") or {}
    cache_dir = Path(cache_dir)
    for source in SOURCES:
        for seed in PRELOCKED_SEEDS:
            for split in ("train", "val"):
                rel = f"predictions/{split}/{source}_seed{seed}_{split}.npz"
                path = cache_path(cache_dir, source, seed, split)
                if rel not in verified:
                    errors.append(f"evidence is missing verified hash for {rel}")
                    continue
                if not path.is_file():
                    errors.append(f"staged cache missing for {rel}")
                    continue
                got = sha256_file(path)
                if got != verified[rel]:
                    errors.append(f"staged {rel} sha256 {got} does not match verified member {verified[rel]}")
    declared = evidence.get("declared_test_checksums") or {}
    for rel, item in declared.items():
        if item.get("arrays_opened") or item.get("extracted"):
            errors.append(f"evidence claims test arrays were opened for {rel}")
        test_like = cache_dir / Path(rel).name
        # Presence of a synthetic test fixture in cache_dir is allowed for unit tests,
        # but this verifier must not unpack or hash-open those arrays.
    return errors


def verify_sidecar_field_source_mapping(cache_dir: Path, claimed: dict[str, Any]) -> list[str]:
    """Check staged sidecars still carry the documented field-source mapping."""
    from src.val_select.io_cache import cache_path

    errors: list[str] = []
    verified = claimed.get("verified_train_val_members") or {}
    cache_dir = Path(cache_dir)
    for source in SOURCES:
        for seed in PRELOCKED_SEEDS:
            for split in ("train", "val"):
                path = cache_path(cache_dir, source, seed, split)
                side = sidecar_path(path)
                if not side.is_file():
                    errors.append(f"missing sidecar for {source}:{seed}:{split}")
                    continue
                data = json.loads(side.read_text(encoding="utf-8"))
                mapping = data.get("field_sources") or {}
                for key in REQUIRED_SIDECAR_FIELD_SOURCES:
                    if key not in mapping:
                        errors.append(f"sidecar {source}:{seed}:{split} is missing field_sources.{key}")
                if data.get("time_index_origin") != TIME_INDEX_ORIGIN_DOCUMENTED:
                    errors.append(f"sidecar {source}:{seed}:{split} time_index_origin is not documented_range_from_bundle")
                if data.get("generator_commit_form") != DOCUMENTED_PREDICT_SCRIPT_FORM:
                    errors.append(f"sidecar {source}:{seed}:{split} generator_commit_form is not documented_short_revision")
                if data.get("fine_tune_commit_status") != FINE_TUNE_COMMIT_STATUS:
                    errors.append(f"sidecar {source}:{seed}:{split} fine_tune_commit_status must remain unknown")
                rel = f"predictions/{split}/{source}_seed{seed}_{split}.npz"
                if data.get("native_member_sha256") != verified.get(rel):
                    errors.append(f"sidecar {source}:{seed}:{split} native member hash does not match claimed verified members")
    return errors


def verify_declared_test_file_bytes(lock: dict[str, Any], cache_dir: Path) -> tuple[list[str], dict[str, Any]]:
    """Compare staged test *file bytes* to declared_sha256 before any NPZ decode.

    This is not a substitute for schema validation. A matching byte hash does not
    decode the file; a mismatch refuses to decode.
    """
    from src.val_select.io_cache import cache_path

    errors: list[str] = []
    audit: dict[str, Any] = {
        "order": ["file_byte_sha256", "npz_decode_not_started"],
        "note": "sha256_file reads raw bytes; it does not call np.load or decode_npz",
        "compared": [],
        "decoded": False,
    }
    declared = (lock.get("input_evidence") or {}).get("declared_test_checksums") or {}
    for rel in required_test_declaration_members():
        item = declared.get(rel)
        if not item or not item.get("declared_sha256"):
            errors.append(f"missing declared test checksum for {rel}")
            continue
        source, rest = rel.split("/")[-1].split("_seed")
        seed_s, _ = rest.split("_", 1)
        path = cache_path(Path(cache_dir), source, int(seed_s), "test")
        if not path.is_file():
            errors.append(f"test file missing for byte-sha check of {rel}; not decoded")
            audit["compared"].append({"rel": rel, "present": False, "decoded": False})
            continue
        got = sha256_file(path)
        expected = item["declared_sha256"]
        match = got == expected
        audit["compared"].append({
            "rel": rel,
            "path": str(path),
            "byte_sha256": got,
            "declared_sha256": expected,
            "match": match,
            "decoded": False,
        })
        if not match:
            errors.append(
                f"test byte sha256 for {rel} is {got}, declared {expected}; refusing to decode NPZ"
            )
    return errors, audit


def verify_lock_input_evidence(
    lock: dict[str, Any],
    *,
    cache_dir: Path | None = None,
    zip_path: Path | None = None,
    root: Path | None = None,
) -> list[str]:
    """Rematerialize trust anchors and compare them to the claimed lock evidence.

    The claimed evidence object is never compared to itself. Stored evidence_hash
    is recomputed from content; a matching hash string does not skip a full
    canonical comparison against the current ZIP and the current adapter.
    Test arrays are not read.
    """
    errors: list[str] = []
    claimed = lock.get("input_evidence")
    if not claimed:
        if lock.get("input_protocol_id") == INPUT_PROTOCOL_ID or lock.get("provenance_kind") == PROVENANCE_DOCUMENTED:
            errors.append("documented_from_bundle lock is missing input_evidence")
        return errors

    recomputed = compute_evidence_hash(claimed)
    stored = claimed.get("evidence_hash")
    if stored != recomputed:
        errors.append(
            "stored evidence_hash does not match the recomputed canonical hash of the evidence content"
        )

    try:
        anchors = rematerialize_zip_trust_anchors(Path(zip_path or claimed.get("zip_path") or ""))
    except (BundleAdapterError, OSError, TypeError, ValueError) as exc:
        errors.append(f"failed to rematerialize ZIP trust anchors: {exc}")
        return errors

    claimed_zip = claimed.get("zip_sha256")
    if claimed_zip != anchors["zip_sha256"]:
        errors.append("current ZIP sha256 does not match the claimed evidence zip_sha256")
    if claimed_zip == ACCEPTED_PR20_ZIP_SHA256 or anchors["zip_sha256"] == ACCEPTED_PR20_ZIP_SHA256:
        if claimed_zip != ACCEPTED_PR20_ZIP_SHA256 or anchors["zip_sha256"] != ACCEPTED_PR20_ZIP_SHA256:
            errors.append(
                "accepted PR20 ZIP sha256 99594eaa2b200fe4b8be51823e5716d6152de3186ccf05afb21b4bba81ef6845 "
                "is the package trust anchor; the current ZIP or claimed hash does not match it"
            )

    if claimed.get("verified_train_val_members") != anchors["verified_train_val_members"]:
        errors.append("verified train/val member hashes do not match current ZIP whitelist members")

    claimed_sources = claimed.get("source_files") or {}
    for rel, got in anchors["source_files"].items():
        if rel not in claimed_sources:
            errors.append(f"claimed evidence is missing source-file hash for {rel}")
        elif claimed_sources[rel] != got:
            errors.append(f"source file {rel} hash does not match the current ZIP member")
    # Any claimed source hash not rematerialized still has to be checked if we can;
    # extra claimed entries that are absent from the ZIP are a mismatch.
    for rel in claimed_sources:
        if rel not in anchors["source_files"] and rel in set(required_nontest_members()) | set(OPTIONAL_SOURCE_MEMBERS):
            errors.append(f"claimed source file {rel} is not in the current ZIP whitelist")

    current_impl = implementation_snapshot(root)
    claimed_fp = (claimed.get("adapter_implementation") or {}).get("fingerprint")
    if claimed_fp != current_impl["fingerprint"]:
        errors.append("adapter implementation fingerprint does not match the current source")

    declared = claimed.get("declared_test_checksums") or {}
    for rel in required_test_declaration_members():
        item = declared.get(rel)
        if not item or not item.get("declared_sha256"):
            errors.append(f"missing declared test checksum for {rel}")
            continue
        if item["declared_sha256"] != anchors["declared_test_sha256"].get(rel):
            errors.append(f"declared test checksum for {rel} does not match SHA256SUMS inside the current ZIP")
        if item.get("declaration_status") != "declared_pending_score_verification":
            errors.append(f"test checksum for {rel} is not in declared-pending state")
        if item.get("arrays_opened") or item.get("extracted"):
            errors.append(f"lock claims test arrays were opened for {rel}")

    if cache_dir is not None:
        errors.extend(verify_sidecar_field_source_mapping(cache_dir, claimed))
    return errors
