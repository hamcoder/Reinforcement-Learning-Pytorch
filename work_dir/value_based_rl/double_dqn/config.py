class Config:

    ENV_NAME = "CartPole-v0"

    PHASE = "test"

    FRAME_NUM = 20000
    MEMORY_SIZE = 1000
    BATCH_SIZE = 32
    TARGET_UPDATE = 100
    EPSILON_DECAY = 1 / 2000
    MIN_EPSILON = 0.1
    MAX_EPSILON = 1.0
    GAMMA = 0.99

    STATE_DIM = 0
    ACTION_DIM = 0

    MODEL_NAME = "double_dqn"
    RESUME = "./work_dir/double_dqn/model/double_dqn_checkpoint.pth.tar"

    SAVE_DIR = "./work_dir/double_dqn"

    LIB_AGENT = "algorithms.double_dqn.agent"
    LIB_MODEL = "algorithms.dqn.model"
    LIB_MEMORY = "algorithms.dqn.replay_buffer"