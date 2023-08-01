import numpy as np

class Environment():

    def __init__(self, state_dim: int, action_dim: int) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        # state is an np.array of size (state_dim,)
        self.state = np.zeros(self.state_dim)

    def reset(self) -> None:
        # resets state according to some distribution
        raise NotImplementedError()
    