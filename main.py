import os
import gym
import importlib
import numpy as np

import torch

from config.config import Config


class Processor():
    def __init__(self):
        self.env_name = Config.ENV_NAME
        self.lib_agent = Config.LIB_AGENT
        self.phase = Config.PHASE
        self.frame_num = Config.FRAME_NUM
        
        self._load_environment()
        self._load_random_seed()
        self._load_agent()

    def _load_environment(self):
        print('loading environment...')

        self.env = gym.make(self.env_name)
        Config.ACTION_DIM = self.env.action_space.n
        Config.STATE_DIM = self.env.observation_space.shape[0]

        print('environment load finished!')
    
    def _load_random_seed(self):
        seed = 777
        
        def seed_torch(seed):
            torch.manual_seed(seed)
            if torch.backends.cudnn.enabled:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

        np.random.seed(seed)
        seed_torch(seed)
        self.env.seed(seed)

    def _load_agent(self):
        print('loading agent...')

        self.agent = importlib.import_module(self.lib_agent).Agent(self.env)

        print('agent load finished!')

    def start(self):
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        if self.phase == 'train':
            self.agent.train(self.frame_num)
        elif self.phase == 'test':
            self.agent.test()
        else:
            raise ValueError

def main():
    processor = Processor()
    processor.start()

if __name__ == '__main__':
    main()