import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import Config
from framework.algorithm.layer import NoisyLinear
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

        self.feature_layer = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU()
        )

        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, self.action_dim * self.atom_size)

        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, self.atom_size)

    def forward(self, x):
        hist = self.hist(x)
        out = torch.sum(hist * self.support.to(device=hist.device), dim=2)
        return out

    def hist(self, x):
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(-1, self.action_dim, self.atom_size)
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        hist = F.softmax(q_atoms, dim=-1)
        hist = hist.clamp(min=1e-3)
        return hist

    def reset_noise(self):
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()