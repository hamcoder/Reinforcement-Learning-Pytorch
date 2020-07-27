import os
import gym
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation

import torch
import torch.optim as optim
import torch.nn.functional as F

from config.config import Config
from framework.algorithm.agent import Agent as Agent_Base


class Agent(Agent_Base):
    def __init__(self, env):
        super().__init__(env)

        if self.phase == "train":
            self._load_memory()

        self._load_model()

        if self.phase == "train":
            self.optimizer = optim.Adam(self.q_network.parameters())

    def _load_memory(self):
        print('loading memory...')

        self.memory = importlib.import_module(self.lib_memory).ReplayBuffer(self.memory_size)

        print('memory load finished!')

    def _load_pretrained(self, resume, model):
        if os.path.isfile(resume):
            print(("=> loading checkpoint '{}'".format(resume)))

            checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint)

            print("=> loaded checkpoint!")
        else:
            print(("=> no checkpoint found at '{}'".format(resume)))

    def _load_model(self):
        print('loading model...')

        self.q_network = importlib.import_module(self.lib_model).Model().to(self.device)

        if self.resume:
            self._load_pretrained(self.resume, self.q_network)

        if self.phase == 'train':
            self.target_q_network = importlib.import_module(self.lib_model).Model().to(self.device)
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            self.target_q_network.eval()

        print('model load finished!')

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

    def test(self):
        self.env = gym.wrappers.Monitor(self.env, os.path.join(self.save_dir, "videos"), force=True)

        state = self.env.reset()
        done = False
        score = 0
        frames = []

        while not done:
            frames.append(self.env.render(mode='rgb_array'))
            action = self._select_action(state)
            next_state, reward, done = self._step(action)
            state = next_state
            score += reward

        self.env.close()

        print("score: ", score)
        self.display_frames_as_gif(frames)

    def _select_action(self, state):
        if self.phase == 'train' and self.epsilon > np.random.random():
            action = self.env.action_space.sample()
        else:
            action = self.q_network(
                torch.FloatTensor(state).to(self.device)
            ).argmax()
            action = action.detach().cpu().numpy()
        return action

    def _step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward, done

    def _store_sample(self, state, action, next_state, reward, done):
        transition = (state, action, next_state, reward, done)
        self.memory.store(*transition)

    def _update_model(self):
        samples = self.memory.sample()
        
        loss = self._compute_loss(samples)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _compute_loss(self, samples):
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
        target_q_value = (reward + self.gamma * next_q_value * mask)

        loss = F.smooth_l1_loss(q_value, target_q_value)
        
        return loss

    def _target_hard_update(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def _plot(self, scores):
        df = pd.DataFrame({'x': range(len(scores)), 'y': scores})
        plt.style.use('seaborn-darkgrid')
        palette = plt.get_cmap('Set1')
        num = 0
        for column in df.drop('x', axis=1):
            num += 1
            plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)
        plt.title("CartPole", fontsize=14)
        plt.xlabel("episode", fontsize=12)
        plt.ylabel("score", fontsize=12)

        plt.savefig('score.png')

    def _save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        path = os.path.join(self.save_dir, 'model_dict')
        os.makedirs(path, exist_ok=True)
        filename = self.model_name + '_' + filename
        torch.save(state, os.path.join(path, filename))

    def display_frames_as_gif(self, frames):
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
        anim.save('./dqn_cart_pole_result.gif', writer='imagemagick', fps=30)