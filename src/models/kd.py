"""Knowledge distillation helpers.

Group 2: frozen teacher + standard KD loss.
Group 3: optional non-learnable weekday Pearson ρ̃ multiplier (hook only).

Upstream references (patterns only — different data domain):
  - TransGTR ``engine.py`` / ``train_distil.py``: teacher.eval + MSE distill
  - ST-MLP ``run_student.py`` / ``models/losses.py``: task + KD combination
See ``docs/UPSTREAM_MAP.md`` §3.
"""

from __future__ import annotations

from typing import Optional

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


def freeze_module(module: "nn.Module") -> "nn.Module":  # noqa: F821
    """Freeze all parameters (Group 2/3 teacher)."""
    if nn is None:
        raise ImportError("PyTorch is required")
    module.eval()
    for p in module.parameters():
        p.requires_grad = False
    return module


class KDWrapper(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Student + frozen teacher.

    ``pearson_mask`` is an optional non-learnable multiplier broadcastable to
    the KD residual (region / hour). Group 3 will plug ``pearson_weekday_profile``
    here — formula TBD; do not invent a final matching algorithm.
    """

    def __init__(
        self,
        student: "nn.Module",  # noqa: F821
        teacher: "nn.Module",  # noqa: F821
        lambda_kd: float = 1.0,
        pearson_mask: Optional["torch.Tensor"] = None,  # noqa: F821
    ) -> None:
        if nn is None:
            raise ImportError("PyTorch is required")
        super().__init__()
        self.student = student
        self.teacher = freeze_module(teacher)
        self.lambda_kd = float(lambda_kd)
        if pearson_mask is not None:
            self.register_buffer("pearson_mask", pearson_mask)
        else:
            self.pearson_mask = None

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
        raise NotImplementedError(
            "KDWrapper.forward is TODO. Implement student forward + teacher "
            "no_grad forward when STNet is ready."
        )

    def kd_loss(
        self,
        student_out: "torch.Tensor",  # noqa: F821
        teacher_out: "torch.Tensor",  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        """Masked MSE between student and teacher outputs.

        If ``pearson_mask`` is set (Group 3), multiply the per-element residual
        energy by the mask before reduction. Matching / multi-channel profile
        details remain TODO.
        """
        if torch is None:
            raise ImportError("PyTorch is required")
        residual = (student_out - teacher_out) ** 2
        if self.pearson_mask is not None:
            residual = residual * self.pearson_mask
        return residual.mean()

    def total_loss(
        self,
        task_loss: "torch.Tensor",  # noqa: F821
        student_out: "torch.Tensor",  # noqa: F821
        teacher_out: "torch.Tensor",  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        return task_loss + self.lambda_kd * self.kd_loss(student_out, teacher_out)
