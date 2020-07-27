import torch.nn as nn

from framework.algorithm.model import Model as Model_Base


class Model(Model_Base):
    def __init__(self):
        super().__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU()
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )

        self.value_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        feature = self.feature_layer(x)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        out = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return out