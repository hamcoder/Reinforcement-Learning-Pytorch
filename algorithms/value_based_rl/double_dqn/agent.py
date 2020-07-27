import torch
import torch.nn.functional as F

from algorithms.dqn.agent import Agent as Agent_Base


class Agent(Agent_Base):
    def __init__(self, env):
        super().__init__(env)

    def _compute_loss(self, samples):
        state = torch.FloatTensor(samples['state']).to(self.device)
        next_state = torch.FloatTensor(samples['next_state']).to(self.device)
        action = torch.LongTensor(samples['action'].reshape(-1, 1)).to(self.device)
        reward = torch.FloatTensor(samples['reward'].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(samples['done'].reshape(-1, 1)).to(self.device)

        q_value = self.q_network(state).gather(1, action)

        next_q_value = self.target_q_network(next_state).gather(
            1, self.q_network(next_state).argmax(dim=1, keepdim=True)
        ).detach()
        mask = 1 - done
        target_q_value = (reward + self.gamma * next_q_value * mask)

        loss = F.smooth_l1_loss(q_value, target_q_value)
        
        return loss