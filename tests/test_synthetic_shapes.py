"""Shape-contract tests without real city data or training."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from src.data.splits import few_shot_split_indices, split_x_y, trim_few_shot, upstream_style_cut_lengths
from src.models.kd import KDWrapper, freeze_module
from src.models.st_net import STNet, expected_io_shapes


def test_expected_io_shapes() -> None:
    s = expected_io_shapes(batch=4, history_len=6, height=21, width=20, n_valid=207)
    assert s["x"] == (4, 6, 21, 20)
    assert s["y_valid"] == (4, 1, 207)


def test_stnet_forward_shape() -> None:
    m = STNet(in_channels=1, residual_blocks=1, lstm_hidden=128)
    x = torch.zeros(2, 6, 8, 8)
    y = m(x)
    assert y.shape == (2, 1, 64)
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask.view(-1)[:10] = True
    y_m = m(x, spatial_mask=mask)
    assert y_m.shape == (2, 1, 10)


def test_kd_loss_shapes_and_freeze() -> None:
    student = STNet()
    teacher = STNet()
    wrap = KDWrapper(student, teacher, lambda_kd=1.0)
    assert all(not p.requires_grad for p in wrap.teacher.parameters())
    s = torch.randn(3, 1, 16)
    t = torch.randn(3, 1, 16)
    loss = wrap.kd_loss(s, t)
    assert loss.shape == ()
    total = wrap.total_loss(torch.tensor(0.1), s, t)
    assert total.shape == ()


def test_kd_pearson_mask_hook() -> None:
    student = STNet()
    teacher = STNet()
    mask = torch.ones(1, 1, 8) * 0.5
    wrap = KDWrapper(student, teacher, lambda_kd=1.0, pearson_mask=mask)
    s = torch.ones(2, 1, 8)
    t = torch.zeros(2, 1, 8)
    # residual=1, mask=0.5, mean over batch then sum over 8 cells → 4
    assert abs(float(wrap.kd_loss(s, t)) - 4.0) < 1e-6
    wrap.train()
    assert wrap.student.training
    assert not wrap.teacher.training
    x = torch.zeros(2, 6, 8, 8)
    student_out, teacher_out = wrap(x)
    assert student_out.shape == teacher_out.shape == (2, 1, 64)


def test_upstream_cut_lengths_match_stbat() -> None:
    cuts = upstream_style_cut_lengths(8784, few_shot_days=3)
    assert cuts["val"] == cuts["test"] == 1440
    assert cuts["few_shot_train"] == 72
    assert cuts["remainder_train"] == 8784 - 1440 - 1440


def test_split_x_y_window_counts() -> None:
    data = np.zeros((200, 2, 2), dtype=np.float32)
    tx, ty, vx, vy, sx, sy = split_x_y(data, val_num=40, test_num=40)
    assert tx.shape == (200 - 6 - 40 - 40, 6, 2, 2)
    assert ty.shape[0] == tx.shape[0] and ty.shape[1:] == (1, 2, 2)
    assert vx.shape[0] == vy.shape[0] == 40
    assert sx.shape[0] == sy.shape[0] == 40
    trimmed_x, trimmed_y = trim_few_shot(tx, ty, few_shot_days=3)
    assert len(trimmed_x) == len(trimmed_y) == 72


def test_few_shot_schema_helper() -> None:
    T = (2 + 2) * 30 * 24 + 3 * 24
    sp = few_shot_split_indices(T, few_shot_days=3)
    assert sp.train.stop - sp.train.start == 72
