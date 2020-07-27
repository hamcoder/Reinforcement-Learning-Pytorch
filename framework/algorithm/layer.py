import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    def __init__(self, in_dim, out_dim, std_init=0.5):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.register_buffer("weight_epsilon", torch.Tensor(out_dim, in_dim))

        self.bias_mu = nn.Parameter(torch.Tensor(out_dim))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_dim))
        self.register_buffer("bias_epsilon", torch.Tensor(out_dim))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_dim)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_dim)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_dim)
        )

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_dim)
        epsilon_out = self.scale_noise(self.out_dim)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon
        )

    @staticmethod
    def scale_noise(size):
        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size))
        return x.sign().mul(x.abs().sqrt())