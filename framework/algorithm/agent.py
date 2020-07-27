from config.config import Config

import torch


class Agent(object):

    def __init__(self, env):
        self.env = env
        self.lib_memory = Config.LIB_MEMORY
        self.lib_model = Config.LIB_MODEL

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.phase = Config.PHASE

        self.model_name = Config.MODEL_NAME
        self.resume = Config.RESUME

        self.memory_size = Config.MEMORY_SIZE
        self.batch_size = Config.BATCH_SIZE

        self.epsilon = Config.MAX_EPSILON
        self.epsilon_decay = Config.EPSILON_DECAY
        self.max_epsilon = Config.MAX_EPSILON
        self.min_epsilon = Config.MIN_EPSILON

        self.target_update = Config.TARGET_UPDATE

        self.gamma = Config.GAMMA

        self.save_dir = Config.SAVE_DIR

    def train(self):
        raise NotImplementedError("build model: not implemented!")

    def test(self):
        raise NotImplementedError("build model: not implemented!")