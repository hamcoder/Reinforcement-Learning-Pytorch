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
from torch.distributions import Categorical, Bernoulli

from config.config import Config
from framework.algorithm.agent import Agent as Agent_Base


class Agent(Agent_Base):
    def __init__(self, env):
        super().__init__(env)

        if self.phase == "train":
            self._load_memory()

        self._load_model()

        if self.phase == "train":
            self.optimizer = optim.Adam(self.policy_network.parameters())

    def _load_memory(self):
        print('loading memory...')

        self.memory = importlib.import_module(self.lib_memory).ReplayBuffer()

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

        self.policy_network = importlib.import_module(self.lib_model).Model().to(self.device)

        if self.resume:
            self._load_pretrained(self.resume, self.policy_network)

        print('model load finished!')

    def train(self, frame_num, plotting_interval=200):
        state = self.env.reset()
        score = 0
        scores = []
        update_cnt = 0
        episode = 0
        states, rewards, actions = [], [], []

        for frame_idx in range(1, frame_num + 1):
            action = self._select_action(state)
            next_state, reward, done = self._step(action)
            states.append(state)
            rewards.append(reward)
            actions.append(action)

            state = next_state
            score += reward

            if done:
                episode += 1
                rewards = self._discount_rewards(rewards, self.gamma)
                self._store_sample(states, rewards, actions)

                if episode % self.batch_size == 0:
                    self._update_model()
                    self.memory.reset()

                state = self.env.reset()
                scores.append(score)
                score = 0
                states, rewards, actions = [], [], []

                if update_cnt % self.target_update == 0:
                    self._save_checkpoint(self.policy_network.state_dict())

            if frame_idx % plotting_interval == 0:
                score_avg = 0
                for s in scores:
                    score_avg += s
                score_avg /= len(scores)
                print("frame_idx: {}, score_avg: {}".format(frame_idx, score_avg))
                self._plot(scores)

        self.env.close()

        self._save_checkpoint(self.policy_network.state_dict())

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
        if self.phase == 'train':
            probs = self.policy_network(torch.FloatTensor(state).to(self.device))
            sampler = Categorical(probs)
            action = sampler.sample().item()
        else:
            action = torch.argmax(self.policy_network(torch.FloatTensor(state).to(device))).item()
        return action

    def _step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward, done

    def _discount_rewards(self, rewards, gamma=0.99):
        # Cumulative discounted sum
        r = np.array([gamma ** i * rewards[i]
                      for i in range(len(rewards))])
        r = r[::-1].cumsum()[::-1]
        return r

    def _store_sample(self, state, reward, action):
        transition = (state, reward, action)
        self.memory.store(*transition)

    def _update_model(self):
        samples = self.memory.sample()

        loss = self._compute_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _compute_loss(self, samples):
        state = torch.FloatTensor(samples['state']).to(self.device)
        action = torch.LongTensor(samples['action']).to(self.device)
        reward = torch.FloatTensor(samples['reward']).to(self.device)

        probs = self.policy_network(state)
        sampler = Categorical(probs)
        log_probs = -sampler.log_prob(action)
        loss = torch.mean(log_probs * reward)

        return loss

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