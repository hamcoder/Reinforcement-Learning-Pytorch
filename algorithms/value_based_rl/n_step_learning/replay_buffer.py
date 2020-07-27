import random
import numpy as np
from collections import namedtuple, deque

from config.config import Config
from framework.data_structure.segment_tree import SegmentTree
from algorithms.dqn.replay_buffer import ReplayBuffer as ReplayBufferBase

class ReplayBuffer(ReplayBufferBase):
    def __init__(self, memory_size, n_step=1):
        super().__init__(memory_size)

        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.gamma = Config.GAMMA

    def store(self, state, action, next_state, reward, done):
        transition = (state, action, next_state, reward, done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            return ()

        next_state, reward, done = self._get_n_step_info()
        state, action = self.n_step_buffer[0][:2]

        self.state[self.position] = state
        self.next_state[self.position] = next_state
        self.action[self.position] = action
        self.reward[self.position] = reward
        self.done[self.position] = done
        self.position = (self.position + 1) % self.memory_size
        self.size = min(self.size + 1, self.memory_size)

        return self.n_step_buffer[0]

    def sample(self):
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(state=self.state[idxs],
                    next_state=self.next_state[idxs],
                    action=self.action[idxs],
                    reward=self.reward[idxs],
                    done=self.done[idxs],
                    indices=idxs)

    def sample_idx(self, idxs):
        return dict(
            state=self.state[idxs],
            next_state=self.next_state[idxs],
            action=self.action[idxs],
            reward=self.reward[idxs],
            done=self.done[idxs])

    def _get_n_step_info(self):
        next_state, reward, done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            s, r, d = transition[-3:]
            reward  = r + self.gamma * reward * (1 - d)
            next_state, done = (s, d) if d else (next_state, reward)

        return next_state, reward, done