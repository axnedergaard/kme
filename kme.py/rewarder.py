import numpy as np
from .rewarder.kmeans_rewarder import KMeansRewarder

class Rewarder:

    def __init__(
        self,
        n_actions: int,
        n_states: int,
        k: int,
        learning_rate: float,
        balancing_strength: float,
        fn_type: str,
        power_fn_exponent: float = 0.5,
    ):
        # private attributes
        self.n_actions = n_actions
        self.n_states = n_states
        self.k = k
        self.learning_rate = learning_rate
        self.balancing_strength = balancing_strength
        self.fn_type = fn_type
        self.power_fn_exponent = power_fn_exponent
        self.reset()

        # public attributes
        self.rewarder = KMeansRewarder(
            k=self.k,
            function_type=self.fn_type,
            n_states=self.n_states,
            learning_rate=self.learning_rate,
            balancing_strength=self.balancing_strength,
            power_fn_exponent=self.power_fn_exponent,
        )


    def reset(self):
        self.rewarder = None

    def infer(self, state: np.ndarray, learn: bool) -> float:
        state = state.astype(np.float64)
        reward = self.rewarder.infer(state, learn)
        pathological_updates = self.rewarder.pathological_updates
        return reward, pathological_updates
    