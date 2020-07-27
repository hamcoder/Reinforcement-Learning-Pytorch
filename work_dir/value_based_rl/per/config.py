class Config:

    ENV_NAME = "CartPole-v0"

    PHASE = "test"

    ALPHA = 0.2
    BETA = 0.6
    PRIOR_EPS = 1e-6

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

    MODEL_NAME = "per"
    RESUME = "./work_dir/per/model_dict/per_checkpoint.pth.tar"

    SAVE_DIR = "./work_dir/per"

    LIB_AGENT = "algorithms.priority_experience_replay.agent"
    LIB_MODEL = "algorithms.dqn.model"
    LIB_MEMORY = "algorithms.priority_experience_replay.replay_buffer"