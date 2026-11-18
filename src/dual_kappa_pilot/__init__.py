"""Dual-kappa pilot v1.1: task-level (kappa_alpha, kappa_c) selection.

This package does not rewrite the frozen val-select-kappa-v3 protocol.
Vendored src/val_select is a byte-identical snapshot used only for I/O,
schema, evidence, metrics, and the inherited single-kappa / old solvers.
"""

from src.dual_kappa_pilot.protocol import (
    ARMS,
    PROTOCOL_ID,
    IDENTITY,
    PRELOCKED_SEEDS,
    SOURCES,
)

__all__ = ["ARMS", "IDENTITY", "PRELOCKED_SEEDS", "PROTOCOL_ID", "SOURCES"]
