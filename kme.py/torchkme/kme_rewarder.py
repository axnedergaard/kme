from typing import Optional
from enum import Enum

import torch
from torch import Tensor

from .kmeans.kmeans_encoder import KMeansEncoder
from .constvars import device


class EntropicFunctionType(Enum):
    LOG = "log"
    ENTROPY = "entropy"
    EXPONENTIAL = "exponential"
    POWER = "power"
    IDENTITY = "identity"


class KMERewarder:
    
    def __init__(
        self,
        k: int,                                     # n_components of KMeandsEncoder
        dim_states: int,                            # dimension of environment states (R^n)
        dim_actions: int,                           # dimension of environment actions (R^m)
        learning_rate: float,                       # alpha - learning rate of KMeansEncoder
        balancing_strength: float,                  # kappa - balancing strength of KMeansEncoder
        function_type: str,                         # type of entropic fn to use for objective
        power_fn_exponent: Optional[float] = 0.5,   # exponent of entropic fn (if power is used)
        eps: Optional[float] = 1e-9,                # epsilon for numerical stability
        homeostasis: bool = True,                   # whether to use homeostasis in KMeansEncoder
        differential: bool = True                   # whether to use differential entropy reward
    ) -> None:
        
        if function_type not in [fn.value for fn in list(EntropicFunctionType)]:
            raise ValueError(f"Entropic function '{function_type}' not supported.")

        # entropic fn. calculation specs
        self.diffential: bool = differential
        self.fn_type: str = function_type
        self.power_fn_exponent: Tensor = torch.tensor(power_fn_exponent, device=device)
        self.eps: Tensor = torch.tensor(eps, device=device)

        # unused yet but maybe
        self.dim_states: int = dim_states
        self.dim_actions: int = dim_actions

        # kmeans encoder state
        self.k_encoder: KMeansEncoder = KMeansEncoder(
            k, dim_states, learning_rate, balancing_strength, homeostasis)


    # --- public interface methods ---
    
    def infer(self, next_state: Tensor, learn: bool = True) -> tuple:
        # Infer the reward and the number of pathological updates given the next state
        # .BAD NAMING CONVENTION FOR LEARN PARAMETER
        
        entropy_before: Tensor = self._estimate_entropy_lb(self.kmeans_enc) # .NOT OPTIMAL
        encoder = self.kmeans_enc.update(next_state, self.learning_rate) \
            if learn else self.kmeans_enc.sim_update(next_state, self.learning_rate)

        if self.differential:
            entropy_after: Tensor = self._estimate_entropy_lb(encoder)
            reward = entropy_before - entropy_after
        else:
            reward = self._estimate_entropy_lb(encoder)

        return reward, 0.0 # no pathological updates yet.


    # --- private interface methods ---

    def _estimate_entropy_lb(self, kmeans_enc: KMeansEncoder) -> Tensor:
        # Estimate the lower bound of the entropy of the kmeans encoding
        # according to eq. (3) in https://arxiv.org/pdf/2205.15623.pdf
        entropies = self._entropic_function(kmeans_enc.closest_distances)
        return torch.sum(entropies)


    def _entropic_function(self, x: Tensor) -> Tensor:
        if self.fn_type == EntropicFunctionType.LOG.value:
            return torch.log(x + self.eps)
        elif self.fn_type == EntropicFunctionType.ENTROPY.value:
            return -x * torch.log(x + self.eps)
        elif self.fn_type == EntropicFunctionType.EXPONENTIAL.value:
            return -torch.exp(-x)
        elif self.fn_type == EntropicFunctionType.POWER.value:
            return torch.pow(x, self.power_fn_exponent)
        elif self.fn_type == EntropicFunctionType.IDENTITY.value:
            return x
        else:
            raise ValueError("Entropic function type not found.")

