class Config:

    ENV_NAME = "CartPole-v0"

    STATE_DIM = 0
    ACTION_DIM = 0

    PHASE = "test"

    # train
    FRAME_NUM = 20000
    TARGET_UPDATE = 100

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

    GAMMA = 0.99

    MODEL_NAME = "noisy_net"
    #RESUME = None
    RESUME = "./work_dir/noisy_net/model_dict/noisy_net_checkpoint.pth.tar"

    SAVE_DIR = "./work_dir/noisy_net"

    LIB_AGENT = "algorithms.noisy_net.agent"
    LIB_MODEL = "algorithms.noisy_net.model"
    LIB_MEMORY = "algorithms.dqn.replay_buffer"