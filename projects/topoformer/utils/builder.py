import torch.nn as nn
from mmcv.utils import Registry, build_from_cfg

BEV_CONSTRUCTOR = Registry('BEV Constructor')

def build_bev_constructor(cfg, default_args=None):
    """Builder for BEV Constructor."""
    return build_from_cfg(cfg, BEV_CONSTRUCTOR, default_args)
