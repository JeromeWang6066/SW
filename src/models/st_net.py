"""ST-Net stub — Phase-1 backbone.

Point of truth for architecture details:
  https://github.com/KL4805/CrossTReS  (see ``src/model.py`` → ``class STNet``)

Cross-check: https://github.com/ShaunHao/stbat ``model.py`` → ``STNet``

Do not invent layer shapes that diverge from upstream without documenting why.
See ``docs/UPSTREAM_MAP.md``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - smoke environments without torch
    torch = None  # type: ignore
    nn = None  # type: ignore


def expected_io_shapes(
    batch: int = 2,
    history_len: int = 6,
    height: int = 8,
    width: int = 8,
    n_valid: Optional[int] = None,
    channels: int = 1,
) -> Dict[str, Tuple[int, ...]]:
    """Document upstream-style tensor shapes for smoke tests (no real forward).

    CrossTReS / STBaT use ``lag = [-6,...,-1]`` → ``history_len=6``.
    Output after spatial mask is typically ``(B, 1, N_valid)``.
    """
    if n_valid is None:
        n_valid = height * width
    if channels == 1:
        x = (batch, history_len, height, width)
    else:
        x = (batch, history_len * channels, height, width)
    return {
        "x": x,
        "y_grid": (batch, 1, height, width),
        "y_valid": (batch, 1, n_valid),
    }


class STNet(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Placeholder ST-Net.

    Intended structure (CrossTReS ``src/model.py``):
      Conv2d(in→64) → ResUnit×``num_convs`` → LSTM(64→128) → Linear → sigmoid.
    Replace ``forward`` with a faithful port; keep ``expected_io_shapes`` in sync.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 64,
        residual_blocks: int = 3,
        lstm_hidden: int = 128,
        pred_len: int = 1,
    ) -> None:
        if nn is None:
            raise ImportError("PyTorch is required for STNet. pip install torch")
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.residual_blocks = residual_blocks
        self.lstm_hidden = lstm_hidden
        self.pred_len = pred_len
        # Honest stub: a single identity-ish proj so construction/smoke works.
        # Real ResUnit/LSTM weights land when porting CrossTReS.
        self._proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self._implemented = False

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
        """Forward is not implemented for real training yet.

        Parameters
        ----------
        x :
            Upstream-style windowed grid batch, e.g. ``(B, L, H, W)`` with
            ``L=6`` (see ``expected_io_shapes``). Finalize channel layout when
            porting CrossTReS ``STNet.forward``.
        """
        raise NotImplementedError(
            "STNet.forward is TODO. Port from KL4805/CrossTReS src/model.py "
            "(class STNet) before any training run. Refusing to return pretend "
            "predictions. See docs/UPSTREAM_MAP.md."
        )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, channels={self.channels}, "
            f"residual_blocks={self.residual_blocks}, lstm_hidden={self.lstm_hidden}, "
            f"implemented={self._implemented}"
        )
