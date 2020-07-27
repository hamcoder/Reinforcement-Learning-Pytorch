import random
import numpy as np
from collections import namedtuple

from config.config import Config
from framework.data_structure.segment_tree import SegmentTree
from algorithms.dqn.replay_buffer import ReplayBuffer as ReplayBufferBase

class ReplayBuffer(ReplayBufferBase):
    def __init__(self, memory_size):
        super().__init__(memory_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = Config.ALPHA

        tree_capacity = 1
        while tree_capacity < self.memory_size:
            tree_capacity *= 2

        self.segment_tree = SegmentTree(tree_capacity)

    def store(self, state, action, next_state, reward, done):
        super().store(state, action, next_state, reward, done)
        self.segment_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.memory_size

    def sample(self, beta):
        idxs = self._sample_proportional()

        state = self.state[idxs]
        next_state = self.next_state[idxs]
        action = self.action[idxs]
        reward = self.reward[idxs]
        done = self.done[idxs]
        weights = np.array([self._calculate_weight(i, beta) for i in idxs])

        return dict(
            state=state,
            next_state=next_state,
            action=action,
            reward=reward,
            done=done,
            weights=weights,
            indices=idxs
        )

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.segment_tree[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self):
        indices = []
        p_total = self.segment_tree.sum()
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.segment_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx, beta):
        p_min = self.segment_tree.min() / self.segment_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        p_sample = self.segment_tree[idx] / self.segment_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight
