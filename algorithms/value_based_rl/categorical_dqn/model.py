import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import Config
from framework.algorithm.model import Model as Model_Base


class Model(Model_Base):
    def __init__(self):
        super().__init__()

        self.v_min = Config.V_MIN
        self.v_max = Config.V_MAX
        self.atom_size = Config.ATOM_SIZE
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        )

        self.layers = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim * self.atom_size)
        )

    def forward(self, x):
        hist = self.hist(x)
        out = torch.sum(hist * self.support.to(device=hist.device), dim=2)
        return out

    def hist(self, x):
        q_atoms = self.layers(x).view(-1, self.action_dim, self.atom_size)
        hist = F.softmax(q_atoms, dim=-1)
        hist = hist.clamp(min=1e-3)
        return hist