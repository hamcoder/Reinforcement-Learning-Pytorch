import torch.nn as nn

from framework.algorithm.model import Model as Model_Base


class Model(Model_Base):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.softmax(self.layers(x))
        return out