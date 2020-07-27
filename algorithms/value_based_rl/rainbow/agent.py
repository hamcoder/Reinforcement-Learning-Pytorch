import torch
import torch.nn.functional as F

from config.config import Config
from algorithms.dqn.agent import Agent as Agent_Base


class Agent(Agent_Base):
    def __init__(self, env):
        super().__init__(env)

        self.prior_eps = Config.PRIOR_EPS
        self.beta = Config.BETA

        self.v_min = Config.V_MIN
        self.v_max = Config.V_MAX
        self.atom_size = Config.ATOM_SIZE
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

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
        samples = self.memory.sample(self.beta)
        weights = torch.FloatTensor(
            samples['weights'].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        self.q_network.reset_noise()
        self.target_q_network.reset_noise()

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

        loss = -(proj_hist * log_p).sum(1)
        return loss