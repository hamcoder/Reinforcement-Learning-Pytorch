class Config:

    ENV_NAME = "CartPole-v0"

    STATE_DIM = 0
    ACTION_DIM = 0

    PHASE = "test"

    # train
    FRAME_NUM = 20000
    TARGET_UPDATE = 200

    # experience replay
    MEMORY_SIZE = 1000
    BATCH_SIZE = 32

    # priority experience replay
    ALPHA = 0.2
    BETA = 0.6
    PRIOR_EPS = 1e-6

    # epsilon greedy
    EPSILON_DECAY = 1 / 2000
    MIN_EPSILON = 0.1
    MAX_EPSILON = 1.0

    # categorical dqn
    V_MIN = 0.0
    V_MAX = 200.0
    ATOM_SIZE = 51

    GAMMA = 0.99

    MODEL_NAME = "categorical_dqn"
    # RESUME = None
    RESUME = "./work_dir/categorical_dqn/model_dict/categorical_dqn_checkpoint.pth.tar"

    SAVE_DIR = "./work_dir/categorical_dqn"

    LIB_AGENT = "algorithms.categorical_dqn.agent"
    LIB_MODEL = "algorithms.categorical_dqn.model"
    LIB_MEMORY = "algorithms.dqn.replay_buffer"