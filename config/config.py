class Config:

    ENV_NAME = "CartPole-v0"

    STATE_DIM = 0
    ACTION_DIM = 0

    PHASE = "train"

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

    # categorical dqn
    V_MIN = 0.0
    V_MAX = 200.0
    ATOM_SIZE = 51

    # n-step learning
    GAMMA = 0.99
    N_STEP = 2

    MODEL_NAME = "policy_gradient"
    RESUME = None
    # RESUME = "./work_dir/rainbow/model_dict/rainbow_checkpoint.pth.tar"

    SAVE_DIR = "./work_dir/policy_based_rl/policy_gradient"

    LIB_AGENT = "algorithms.policy_based_rl.policy_gradient.agent"
    LIB_MODEL = "algorithms.policy_based_rl.policy_gradient.model"
    LIB_MEMORY = "algorithms.policy_based_rl.policy_gradient.replay_buffer"
