import torch
import torch.optim as optim
import torch.nn.functional as F

from config.config import Config
from algorithms.dqn.agent import Agent as Agent_Base


class Agent(Agent_Base):
    def __init__(self, env):
        super().__init__(env)
        self.prior_eps = Config.PRIOR_EPS
        self.beta = Config.BETA

    def train(self, frame_num, plotting_interval=200):
        state = self.env.reset()
        score = 0
        scores = []
        losses = []
        update_cnt = 0

        for frame_idx in range(1, frame_num + 1):
            action = self._select_action(state)
            next_state, reward, done = self._step(action)
            self._store_sample(state, action, next_state, reward, done)

            state = next_state
            score += reward

            fraction = min(frame_idx / frame_num, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0

            if len(self.memory) >= self.batch_size:
                loss = self._update_model()
                losses.append(loss)
                update_cnt += 1

                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                            self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )

                if update_cnt % self.target_update == 0:
                    self._target_hard_update()
                    self._save_checkpoint(self.q_network.state_dict())

            if frame_idx % plotting_interval == 0:
                score_avg = 0
                for s in scores:
                    score_avg += s
                score_avg /= len(scores)
                print("frame_idx: {}, score_avg: {}".format(frame_idx, score_avg))
                self._plot(scores)

        self.env.close()

        self._save_checkpoint(self.q_network.state_dict())

    def _update_model(self):
        samples = self.memory.sample(self.beta)
        weights = torch.FloatTensor(
            samples['weights'].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        element_loss = self._compute_loss(samples)
        loss = torch.mean(element_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_for_prior = element_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        return loss.item()

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

        loss = F.smooth_l1_loss(q_value, target_q_value, reduction="none")
        
        return loss