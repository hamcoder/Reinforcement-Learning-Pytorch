import torch.nn as nn

from config.config import Config


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_dim = Config.STATE_DIM
        self.action_dim = Config.ACTION_DIM
        
    def forward(self, x):
        raise NotImplementedError("build model: not implemented!")