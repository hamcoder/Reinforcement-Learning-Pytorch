class ReplayBuffer(object):
    def __init__(self):
        self.batch_states = []
        self.batch_rewards = []
        self.batch_actions = []

    def store(self, states, rewards, actions):
        self.batch_states.extend(states)
        self.batch_rewards.extend(rewards)
        self.batch_actions.extend(actions)

    def sample(self):
        return dict(state=self.batch_states,
                    reward=self.batch_rewards,
                    action=self.batch_actions)

    def reset(self):
        self.batch_states = []
        self.batch_rewards = []
        self.batch_actions = []