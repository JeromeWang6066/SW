"""Prediction-cache I/O with a hard select-vs-test separation.

Select mode may open only train and val files. Test files are opened only by
the dedicated scoring entry after a selection lock is verified. Inventory
records size and sha256 only and never unpacks arrays.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.val_select.protocol import PRELOCKED_SEEDS, SOURCES
from src.val_select.schema import SchemaError, merge_payload_and_sidecar, validate_payload

ALLOWED_SELECT_SPLITS = frozenset({"train", "val"})
ALLOWED_TEST_SPLITS = frozenset({"test"})
ALLOWED_REFIT_SPLITS = frozenset({"train", "val"})

CANONICAL_JSON_SEPARATORS = (",", ":")


def cache_path(cache_dir: Path, source: str, seed: int, split: str) -> Path:
    return Path(cache_dir) / f"{source}_seed{seed}_{split}.npz"


def sidecar_path(path: Path) -> Path:
    return Path(path).with_suffix(Path(path).suffix + ".manifest.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=CANONICAL_JSON_SEPARATORS, ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _require_split(split: str, allowed: frozenset[str], *, role: str) -> None:
    if split not in allowed:
        raise PermissionError(f"{role} is not allowed to open the {split!r} cache")


def _merge_sidecar(data: dict[str, Any], path: Path) -> dict[str, Any]:
    side = sidecar_path(path)
    if not side.is_file():
        return data
    meta = json.loads(side.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise SchemaError(f"sidecar {side} is not a JSON object")
    try:
        return merge_payload_and_sidecar(data, meta)
    except SchemaError as exc:
        raise SchemaError(f"{side}: {exc}") from exc


def decode_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


class CacheAccess:
    """Role-restricted cache reader. Select access cannot touch test files."""

    def __init__(
        self,
        cache_dir: Path,
        role: str,
        *,
        allow_synthetic: bool = False,
        allow_documented_from_bundle: bool = False,
        expected_cells: int | None = None,
        expected_cell_ids: np.ndarray | None = None,
        on_test_load: Callable[[str, int, str], None] | None = None,
    ) -> None:
        if role not in {"select", "score_test", "inventory", "refit"}:
            raise ValueError(f"unknown cache role {role}")
        self.cache_dir = Path(cache_dir)
        self.role = role
        self.allow_synthetic = allow_synthetic
        self.allow_documented_from_bundle = allow_documented_from_bundle
        self.expected_cells = expected_cells
        self.expected_cell_ids = expected_cell_ids
        self.on_test_load = on_test_load
        self.test_load_count = 0
        self.test_load_calls: list[tuple[str, int, str]] = []

    def path(self, source: str, seed: int, split: str) -> Path:
        if self.role == "select":
            _require_split(split, ALLOWED_SELECT_SPLITS, role="select")
        elif self.role == "score_test":
            _require_split(split, ALLOWED_TEST_SPLITS, role="score_test")
        elif self.role == "refit":
            _require_split(split, ALLOWED_REFIT_SPLITS, role="refit")
        elif self.role == "inventory":
            raise PermissionError("inventory cannot open or unpack cache arrays")
        return cache_path(self.cache_dir, source, seed, split)

    def load(self, source: str, seed: int, split: str) -> dict[str, Any]:
        path = self.path(source, seed, split)
        if not path.is_file():
            raise FileNotFoundError(path)
        if self.role == "select" and "_test" in path.name:
            raise PermissionError(f"select mode refused test-like path {path}")
        if split == "test":
            self.test_load_count += 1
            self.test_load_calls.append((source, int(seed), split))
            if self.on_test_load is not None:
                self.on_test_load(source, int(seed), split)
        raw = decode_npz(path)
        raw = _merge_sidecar(raw, path)
        raw["_path"] = str(path)
        raw["_sha256"] = sha256_file(path)
        raw["_sidecar_sha256"] = sha256_file(sidecar_path(path)) if sidecar_path(path).is_file() else None
        validated = validate_payload(
            raw,
            requested_source=source,
            requested_seed=int(seed),
            requested_split=split,
            allow_synthetic=self.allow_synthetic,
            allow_documented_from_bundle=self.allow_documented_from_bundle,
            expected_cells=self.expected_cells,
            expected_cell_ids=self.expected_cell_ids,
        )
        validated["_path"] = raw["_path"]
        validated["_sha256"] = raw["_sha256"]
        validated["_sidecar_sha256"] = raw["_sidecar_sha256"]
        validated["manifest"]["file_sha256"] = raw["_sha256"]
        validated["manifest"]["sidecar_sha256"] = raw["_sidecar_sha256"]
        return validated


def file_stat_record(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    record = {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
    }
    side = sidecar_path(path)
    if side.is_file():
        record["sidecar"] = {
            "path": str(side),
            "size": int(side.stat().st_size),
            "sha256": sha256_file(side),
        }
    else:
        record["sidecar"] = None
    return record


def inventory_caches(cache_dir: Path, sources=None, seeds=None) -> dict[str, Any]:
    """Stat and hash only. Never unpacks prediction arrays, including test."""
    sources = list(sources or SOURCES)
    seeds = list(seeds or PRELOCKED_SEEDS)
    access = CacheAccess(cache_dir, role="inventory")
    records = []
    for source in sources:
        for seed in seeds:
            for split in ("train", "val", "test"):
                path = cache_path(Path(cache_dir), source, seed, split)
                record = file_stat_record(path)
                record.update({"source": source, "seed": int(seed), "split": split})
                records.append(record)
    present = [item for item in records if item["exists"]]
    return {
        "cache_dir": str(Path(cache_dir)),
        "n_present": len(present),
        "n_expected": len(records),
        "unpacked_arrays": False,
        "note": "inventory records size and sha256 only; test predictions are not decompressed",
        "files": records,
        # Role object exists only to document that inventory cannot load arrays.
        "inventory_role": access.role,
    }


def write_cache_npz(path: Path, arrays: dict[str, Any], sidecar: dict[str, Any] | None = None) -> Path:
    """Write a schema-capable cache. Used by synthetic tests, not to invent real provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = {}
    for key, value in arrays.items():
        if isinstance(value, (str, bytes)):
            packed[key] = np.asarray(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            packed[key] = np.asarray(value)
        else:
            packed[key] = np.asarray(value)
    np.savez_compressed(path, **packed)
    if sidecar:
        sidecar_path(path).write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
