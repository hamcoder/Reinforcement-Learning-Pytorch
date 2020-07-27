import torch
import torch.optim as optim
import torch.nn.functional as F

from config.config import Config
from algorithms.dqn.agent import Agent as Agent_Base


class Agent(Agent_Base):
    def __init__(self, env):
        super().__init__(env)

    def _select_action(self, state):
        if self.phase == 'train':
            self.target_q_network.reset_noise()
            action = self.target_q_network(
                torch.FloatTensor(state).to(self.device)
            ).argmax()
        else:
            action = self.q_network(
                torch.FloatTensor(state).to(self.device)
            ).argmax()
        action = action.detach().cpu().numpy()
        return action

    def _update_model(self):
        samples = self.memory.sample()

        self.q_network.reset_noise()
        self.target_q_network.reset_noise()

        loss = self._compute_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()