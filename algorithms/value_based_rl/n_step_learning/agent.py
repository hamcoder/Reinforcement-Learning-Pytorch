import importlib

import torch
import torch.nn.functional as F

from config.config import Config
from algorithms.dqn.agent import Agent as Agent_Base


class Agent(Agent_Base):
    def __init__(self, env):
        super().__init__(env)
        self.n_step = Config.N_STEP

        if self.phase == "train":
            self._load_n_memory()

    def _load_n_memory(self):
        print('loading n step memory...')

        self.n_memory = importlib.import_module(self.lib_memory).ReplayBuffer(self.memory_size, self.n_step)

        print('n step memory load finished!')

    def _store_sample(self, state, action, next_state, reward, done):
        transition = (state, action, next_state, reward, done)
        one_step_transition = self.n_memory.store(*transition)
        if one_step_transition:
            self.memory.store(*one_step_transition)

    def _update_model(self):
        #samples = self.memory.sample()
        #indices = samples['indices']
        #loss = self._compute_loss(samples, self.gamma)

        #samples = self.n_memory.sample_idx(indices)
        samples = self.n_memory.sample()
        gamma = self.gamma ** self.n_step
        loss = self._compute_loss(samples, gamma)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _compute_loss(self, samples, gamma):
        state = torch.FloatTensor(samples['state']).to(self.device)
        next_state = torch.FloatTensor(samples['next_state']).to(self.device)
        action = torch.LongTensor(samples['action'].reshape(-1, 1)).to(self.device)
        reward = torch.FloatTensor(samples['reward'].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(samples['done'].reshape(-1, 1)).to(self.device)

        q_value = self.q_network(state).gather(1, action)
        next_q_value = self.target_q_network(
            next_state
        ).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target_q_value = (reward + gamma * next_q_value * mask)

        loss = F.smooth_l1_loss(q_value, target_q_value)

        return loss