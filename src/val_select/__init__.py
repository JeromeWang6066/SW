"""Val-based strength selection for centered affine calibration.

New protocol on top of PR20 solvers. Does not rewrite the PR20 leave-one-day
training-window protocol. Selection code must never open test caches.
"""

from src.val_select.protocol import (
    ARMS,
    CANDIDATE_STRENGTHS,
    IDENTITY,
    PRELOCKED_SEEDS,
    SOURCES,
)

__all__ = ["ARMS", "CANDIDATE_STRENGTHS", "IDENTITY", "PRELOCKED_SEEDS", "SOURCES"]
