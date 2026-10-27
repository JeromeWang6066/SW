"""Model stubs: ST-Net (CrossTReS) and KD wrappers."""

from .kd import KDWrapper, freeze_module
from .st_net import STNet

__all__ = ["STNet", "KDWrapper", "freeze_module"]
