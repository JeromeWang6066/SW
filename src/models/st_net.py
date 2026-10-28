"""CrossTReS ST-Net (Phase-1 backbone).

Ported from https://github.com/KL4805/CrossTReS ``src/model.py`` ``class STNet``
and ``class ResUnit`` (the ``forward`` path, not ``functional_forward``).
Cross-check: https://github.com/ShaunHao/stbat ``model.py`` ``class STNet``.

Upstream hardcodes conv width 64 and LSTM hidden 128. Those are the defaults
here. ``functional_forward`` (meta-learning) is not ported; Phase-1 does not
use region re-weighting.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - smoke environments without torch
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


def expected_io_shapes(
    batch: int = 2,
    history_len: int = 6,
    height: int = 8,
    width: int = 8,
    n_valid: Optional[int] = None,
    channels: int = 1,
) -> Dict[str, Tuple[int, ...]]:
    """Document upstream-style tensor shapes.

    CrossTReS / STBaT use ``lag = [-6,...,-1]`` → ``history_len=6``.
    With one flow channel, ``X`` is ``(B, history_len, H, W)``.
    After the spatial mask, ``STNet`` returns ``(B, 1, N_valid)``.
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


class ResUnit(nn.Module if nn is not None else object):  # type: ignore[misc]
    """CrossTReS residual unit: BN-ReLU-Conv ×2, then add the input."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        if nn is None:
            raise ImportError("PyTorch is required for ResUnit. pip install torch")
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
        z = self.bn1(x)
        z = F.relu(z)
        z = self.conv1(z)
        z = self.bn2(z)
        z = F.relu(z)
        z = self.conv2(z)
        return z + x


class STNet(nn.Module if nn is not None else object):  # type: ignore[misc]
    """CrossTReS ``STNet``: per-lag conv stack, LSTM over lags, sigmoid on valid cells.

    Parameters match the previous stub names (``in_channels``, ``residual_blocks``,
    ``lstm_hidden``) so configs and smoke tests keep working. ``spatial_mask``
    is optional at init; ``forward`` can also receive one. A missing mask keeps
    every grid cell (useful for shape tests). Training passes the upstream
    ``sum(time) > 0`` mask.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 64,
        residual_blocks: int = 3,
        lstm_hidden: int = 128,
        pred_len: int = 1,
        spatial_mask: Optional["torch.Tensor"] = None,  # noqa: F821
    ) -> None:
        if nn is None:
            raise ImportError("PyTorch is required for STNet. pip install torch")
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.residual_blocks = residual_blocks
        self.lstm_hidden = lstm_hidden
        self.pred_len = pred_len
        self.num_channels = in_channels
        self.num_convs = residual_blocks
        self._implemented = True

        self.conv1 = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.layers = nn.ModuleList(
            [ResUnit(channels, channels) for _ in range(residual_blocks)]
        )
        self.lstm = nn.LSTM(channels, lstm_hidden)
        self.linear = nn.Linear(lstm_hidden * 2, pred_len)

        if spatial_mask is not None:
            mask = torch.as_tensor(spatial_mask).bool()
            self.register_buffer("spatial_mask", mask)
        else:
            self.spatial_mask = None

    def forward(
        self,
        x: "torch.Tensor",  # noqa: F821
        spatial_mask: Optional["torch.Tensor"] = None,  # noqa: F821
        return_feat: bool = False,
    ):
        """Return ``(B, pred_len, N_valid)``. With ``return_feat``, also the pre-mask features."""
        if spatial_mask is None:
            spatial_mask = self.spatial_mask
        if spatial_mask is None:
            spatial_mask = torch.ones(
                x.shape[2], x.shape[3], dtype=torch.bool, device=x.device
            )
        else:
            spatial_mask = spatial_mask.to(device=x.device)
            if spatial_mask.dtype != torch.bool:
                spatial_mask = spatial_mask.bool()
            spatial_mask = spatial_mask.reshape(x.shape[2], x.shape[3])

        num_lag = x.shape[1] // self.num_channels
        batch_size = x.shape[0]
        outs = []
        for i in range(num_lag):
            step = x[:, i * self.num_channels : (i + 1) * self.num_channels, :, :]
            z = self.conv1(step)
            for layer in self.layers:
                z = layer(z)
            z = z.permute(0, 2, 3, 1).reshape(batch_size, -1, self.channels).contiguous()
            outs.append(z.reshape(-1, self.channels))

        # (lag, B*H*W, C) — each cell is its own sequence, as in CrossTReS.
        seq = torch.stack(outs, dim=0)
        temporal_out, (temporal_hid, _) = self.lstm(seq)
        temporal_out = temporal_out[-1:, :]
        temporal = torch.cat(
            [
                temporal_out.reshape(batch_size, -1, self.lstm_hidden),
                temporal_hid.reshape(batch_size, -1, self.lstm_hidden),
            ],
            dim=-1,
        )
        temporal_valid = temporal[:, spatial_mask.reshape(-1), :]
        output = torch.sigmoid(self.linear(temporal_valid)).permute(0, 2, 1)
        if return_feat:
            return temporal, output
        return output

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, channels={self.channels}, "
            f"residual_blocks={self.residual_blocks}, lstm_hidden={self.lstm_hidden}, "
            f"implemented={self._implemented}"
        )
