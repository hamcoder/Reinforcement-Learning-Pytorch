import torch.nn as nn
import torch.nn.functional as F

from framework.algorithm.layer import NoisyLinear
from framework.algorithm.model import Model as Model_Base


class Model(Model_Base):
    def __init__(self):
        super().__init__()

        self.feature_layer = nn.Linear(self.state_dim, 128)
        self.noisy_layer1 = NoisyLinear(128, 128)
        self.noisy_layer2 = NoisyLinear(128, self.action_dim)

    def forward(self, x):
        feature = F.relu(self.feature_layer(x))
        hidden = F.relu(self.noisy_layer1(feature))
        out = self.noisy_layer2(hidden)
        return out

    def reset_noise(self):
        self.noisy_layer1.reset_noise()
        self.noisy_layer2.reset_noise()