import torch
from lib import optimizers
import lib.optimizers.optimizers_utils as optimizers_utils

@optimizers_utils.register_optimizer
def Adam(params, cfg):
    return torch.optim.Adam(params, cfg.optimizer.lr)