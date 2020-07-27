import torch
import torch.nn.functional as F

from config.config import Config
from algorithms.dqn.agent import Agent as Agent_Base


class Agent(Agent_Base):
    def __init__(self, env):
        super().__init__(env)

        self.v_min = Config.V_MIN
        self.v_max = Config.V_MAX
        self.atom_size = Config.ATOM_SIZE
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

    def _compute_loss(self, samples):
        state = torch.FloatTensor(samples['state']).to(self.device)
        next_state = torch.FloatTensor(samples['next_state']).to(self.device)
        action = torch.LongTensor(samples['action']).to(self.device)
        reward = torch.FloatTensor(samples['reward'].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(samples['done'].reshape(-1, 1)).to(self.device)

        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            next_action = self.q_network(next_state).argmax(1)
            next_hist = self.target_q_network.hist(next_state)
            batch_size = next_hist.size(0)
            next_hist = next_hist[range(batch_size), next_action]

            t_z = reward + (1 - done) * self.gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (batch_size - 1) * self.atom_size, self.batch_size
                ).long().unsqueeze(1).expand(batch_size, self.atom_size).to(self.device)
            )

            proj_hist = torch.zeros(next_hist.size(), device=self.device)
            proj_hist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_hist * (u.float() - b)).view(-1)
            )
            proj_hist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_hist * (b - l.float())).view(-1)
            )

        hist = self.q_network.hist(state)
        log_p = torch.log(hist[range(hist.size(0)), action])

        loss = -(proj_hist * log_p).sum(1).mean()
        return loss