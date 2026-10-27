#!/usr/bin/env python3
"""Synthetic tensor-shape checks for ST-Net / KD interfaces (no real data)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    print("=== SW synthetic shape smoke ===")

    import numpy as np

    from src.data.splits import few_shot_split_indices, upstream_style_cut_lengths
    from src.eval.metrics import summarize
    from src.models.kd import KDWrapper, freeze_module
    from src.models.st_net import STNet, expected_io_shapes
    from src.utils.seed import set_seed

    set_seed(0)

    # Documented CrossTReS/STBaT window: lag of 6 hours, single channel
    shapes = expected_io_shapes(batch=2, history_len=6, height=8, width=8, n_valid=40)
    assert shapes["x"] == (2, 6, 8, 8)
    print("  expected_io_shapes OK:", shapes)

    student = STNet(in_channels=1, residual_blocks=3)
    teacher = STNet(in_channels=1, residual_blocks=3)
    freeze_module(teacher)
    assert all(not p.requires_grad for p in teacher.parameters())

    wrap = KDWrapper(student, teacher, lambda_kd=0.5)
    # Fake logits on valid regions — mimics upstream (B, 1, N_valid) style
    s_out = __import__("torch").zeros(2, 1, 40)
    t_out = __import__("torch").ones(2, 1, 40)
    kd = wrap.kd_loss(s_out, t_out)
    assert kd.ndim == 0
    task = __import__("torch").tensor(0.25)
    total = wrap.total_loss(task, s_out, t_out)
    assert abs(float(total) - (0.25 + 0.5 * float(kd))) < 1e-6
    print("  KDWrapper kd_loss/total_loss shapes OK")

    # forward must still refuse pretend predictions
    try:
        student(__import__("torch").zeros(*shapes["x"]))
        print("  ERROR: STNet.forward should raise NotImplementedError")
        return 1
    except NotImplementedError:
        print("  STNet.forward correctly refuses (NotImplementedError)")

    T = 8784  # matches verified upstream year length
    cuts = upstream_style_cut_lengths(T, few_shot_days=3)
    assert cuts["test"] == 60 * 24
    assert cuts["val"] == 60 * 24
    assert cuts["few_shot_train"] == 3 * 24
    print("  upstream_style_cut_lengths OK:", cuts)

    # Schema helper still usable for approximate month protocol
    approx_T = (2 + 2) * 30 * 24 + 3 * 24
    splits = few_shot_split_indices(approx_T, few_shot_days=3)
    assert splits.train.stop - splits.train.start == 72

    # Metrics wiring only
    a = np.zeros((4, 4))
    b = np.ones((4, 4))
    _ = summarize(a, b)
    print("Synthetic shape smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
