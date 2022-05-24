import logging
import torch.nn as nn
from models.hrnet import get_pose_net

logger = logging.getLogger(__name__)


class HRNetBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.body = get_pose_net(cfg, is_train=True)

    def forward(self, x):
        y = self.body(x)
        return y


def build_backbone(cfg):
    
    backbone = HRNetBackbone(cfg)
    
    return backbone