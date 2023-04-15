class BaseRewarder:
    def __init__(self, simple, n_actions, n_states):
        self.simple = simple
        self.n_actions = n_actions
        self.n_states = n_states

    def reset(self):
        raise NotImplementedError("Reset method should be implemented in the subclass")

    def infer(self, next_state, action, state, learn):
        raise NotImplementedError("Infer method should be implemented in the subclass")
