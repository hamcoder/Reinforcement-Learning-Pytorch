import random
import numpy as np
from collections import namedtuple

from config.config import Config

class ReplayBuffer(object):
    state_dim = Config.STATE_DIM
    batch_size = Config.BATCH_SIZE

    def __init__(self, memory_size=100000):
        self.state = np.zeros([memory_size, self.state_dim])
        self.next_state = np.zeros([memory_size, self.state_dim])
        self.action = np.zeros([memory_size])
        self.reward = np.zeros([memory_size])
        self.done = np.zeros([memory_size])
        self.memory_size = memory_size
        self.position, self.size = 0, 0

    def store(self, state, action, next_state, reward, done):
        self.state[self.position] = state
        self.next_state[self.position] = next_state
        self.action[self.position] = action
        self.reward[self.position] = reward
        self.done[self.position] = done
        self.position = (self.position + 1) % self.memory_size
        self.size = min(self.size + 1, self.memory_size)

    def sample(self):
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(state=self.state[idxs],
                    next_state=self.next_state[idxs],
                    action=self.action[idxs],
                    reward=self.reward[idxs],
                    done=self.done[idxs])

    def __len__(self):
        return self.size