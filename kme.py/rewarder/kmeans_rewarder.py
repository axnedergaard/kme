import numpy as np
from .base_rewarder import Rewarder
from encoder.kmeans_encoder import KMeansEncoder
from typing import Optional


class EntropicFunctionType:
    LOG = 0
    ENTROPY = 1
    EXPONENTIAL = 2
    POWER = 3
    IDENTITY = 4


class KMeansRewarder(Rewarder):

    def __init__(
        self,
        k: Optional[int] = None,
        function_type: Optional[EntropicFunctionType] = None,
        n_states: Optional[int] = None,
        learning_rate: Optional[float] = None,
        balancing_strength: Optional[float] = None,
        eps: Optional[float] = 1e-9,
        differential: Optional[bool] = True,
        power_fn_exponent: Optional[float] = 0.5,
    ):
        super().__init__(simple=True, n_actions=0, n_states=n_states)

        # private attributes
        self.k: int = k
        self.eps: float = eps
        self.differential: bool = differential
        self.power_fn_exponent: float = power_fn_exponent
        self.function_type: EntropicFunctionType = function_type

        # public attributes
        self.encoder: KMeansEncoder = KMeansEncoder(
            n_states, k, learning_rate, balancing_strength, online=True
        )

    
    # --- public interface methods ---

    def estimate_entropy(self, _encoder: KMeansEncoder) -> float:
        entropies = [self.entropic_function(x) for x in _encoder.closest_distances]
        return np.sum(entropies)
    

    def infer(self, next_state: np.ndarray, action: np.ndarray, state: np.ndarray, learn: bool) -> float:
        encoded_next_state: np.ndarray = next_state
        entropy_before, entropy_after = 0.0, 0.0
        
        if self.differential:
            entropy_before = self.estimate_entropy(self.encoder)
        
        if learn:
            self.encoder.embed(encoded_next_state, None)
            entropy_after = self.estimate_entropy(self.encoder)
        else:
            tmp_encoder: KMeansEncoder = KMeansEncoder.copy(self.encoder)
            tmp_encoder.embed(encoded_next_state, None)
            entropy_after = self.estimate_entropy(tmp_encoder)
        
        if self.differential:
            entropy_change: float = entropy_after - entropy_before
            return entropy_change
        else:
            return entropy_after
        
    
    def reset(self) -> None:
        self.encoder.reset()
    
    
    # --- private interface methods ---

    def entropic_function(self, x: float) -> float:
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
            raise ValueError("Entropic function type not found.")

