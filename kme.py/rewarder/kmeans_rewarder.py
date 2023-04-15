import numpy as np
from .base_rewarder import Rewarder
from encoder.kmeans_encoder import KMeansEncoder


class EntropicFunctionType:
    LOG = 0
    ENTROPY = 1
    EXPONENTIAL = 2
    POWER = 3
    IDENTITY = 4


class KMeansRewarder(Rewarder):
    def __init__(
        self,
        k,
        function_type,
        n_states,
        learning_rate,
        balancing_strength,
        eps=1e-9,
        differential=True,
        power_fn_exponent=0.5,
    ):
        super().__init__(simple=True, n_actions=0, n_states=n_states)
        self.k = k
        self.eps = eps
        self.differential = differential
        self.power_fn_exponent = power_fn_exponent
        self.function_type = function_type
        self.encoder = KMeansEncoder(
            n_states, k, learning_rate, balancing_strength, online=True
        )

    def entropic_function(self, x):
        if self.function_type == EntropicFunctionType.LOG:
            return np.log(x + self.eps)
        elif self.function_type == EntropicFunctionType.ENTROPY:
            return -x * np.log(x + self.eps)
        elif self.function_type == EntropicFunctionType.EXPONENTIAL:
            return -np.exp(-x)
        elif self.function_type == EntropicFunctionType.POWER:
            return np.power(x, self.power_fn_exponent)
        elif self.function_type == EntropicFunctionType.IDENTITY:
            return x
        else:
            print("Warning: Entropic function type not found.")
            return 0.0

    def estimate_entropy(self, _encoder):
        entropy = 0.0
        for i in range(self.k):
            entropy += self.entropic_function(_encoder.closest_distances[i])
        return entropy

    def infer(self, next_state, action, state, learn):
        if self.differential:
            entropy_before = self.estimate_entropy(self.encoder)
        if learn:
            self.encoder.embed(next_state, None)
            entropy_after = self.estimate_entropy(self.encoder)
        else:
            tmp_encoder = KMeansEncoder.copy(self.encoder)
            tmp_encoder.embed(next_state, None)
            entropy_after = self.estimate_entropy(tmp_encoder)

        if self.differential:
            entropy_change = entropy_after - entropy_before
            return entropy_change
        else:
            return entropy_after

    def reset(self):
        self.encoder.reset()
