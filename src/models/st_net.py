"""ST-Net stub — Phase-1 backbone.

Point of truth for architecture details:
  https://github.com/KL4805/CrossTReS  (see ``src/model.py``)

Do not invent layer shapes that diverge from upstream without documenting why.
"""

from __future__ import annotations

from typing import Optional

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - smoke environments without torch
    torch = None  # type: ignore
    nn = None  # type: ignore


class STNet(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Placeholder ST-Net.

    Intended structure (from CrossTReS / STBaT descriptions): residual CNN
    blocks + LSTM temporal module. Replace ``forward`` with a faithful port.
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
        # Honest stub: a single identity-ish proj so shape smoke tests can run later.
        self._proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self._implemented = False

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
        """Forward is not implemented for real training yet.

        Parameters
        ----------
        x :
            Expected windowed grid batch, e.g. ``(B, L, H, W)`` or
            ``(B, L, C, H, W)`` — finalize when porting CrossTReS.
        """
        raise NotImplementedError(
            "STNet.forward is TODO. Port from KL4805/CrossTReS model.py "
            "before any training run. Refusing to return pretend predictions."
        )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, channels={self.channels}, "
            f"residual_blocks={self.residual_blocks}, lstm_hidden={self.lstm_hidden}, "
            f"implemented={self._implemented}"
        )
