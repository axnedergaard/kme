from typing import Optional, Tuple
from enum import Enum

import torch
from torch import Tensor

from ..kmeans import KMeansEncoder
from ..constvars import device, dtype


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
        
        assert power_fn_exponent > 0, "power_fn_exponent must be greater than 0"
        assert 0 < eps < 1, "eps must be in the range (0, 1)"
        assert dim_actions > 0, "Dimension of environment actions must be greater than 0"

        # entropic fn. calculation specs
        self.differential: bool = differential
        self.fn_type: EntropicFunctionType = EntropicFunctionType(function_type)
        self.power_fn_exponent: Tensor = torch.tensor(power_fn_exponent, device=device)
        self.eps: Tensor = torch.tensor(eps, device=device)

        # unused yet but maybe in future
        self.dim_states: int = dim_states
        self.dim_actions: int = dim_actions

        # kmeans encoder state
        self.k_encoder: KMeansEncoder = KMeansEncoder(
            k, dim_states, learning_rate, balancing_strength, homeostasis)


    # --- public interface methods ---
    
    def infer(self, next_state: Tensor, update_encoder: bool = True) -> Tuple[Tensor, float, Tensor]:
        # Infer the reward and the number of pathological updates given the next state

        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=dtype, device=device)
        
        if self.differential:
            entropy_before: Tensor = self._estimate_entropy_lb(self.k_encoder)
        
        tmp_encoder, cluster_idx = self.k_encoder.update(next_state) \
            if update_encoder else self.k_encoder.sim_update_v1(next_state)

        if self.differential:
            entropy_after: Tensor = self._estimate_entropy_lb(tmp_encoder)
            reward = entropy_before - entropy_after
        else:
            reward = self._estimate_entropy_lb(tmp_encoder)

        return reward, 0.0, cluster_idx # no pathological updates


    # --- private interface methods ---

    def _estimate_entropy_lb(self, encoder: KMeansEncoder) -> Tensor:
        # Estimate the lower bound of the entropy of the kmeans encoding
        # according to eq. (3) in https://arxiv.org/pdf/2205.15623.pdf
        entropies = self._entropic_function(encoder.closest_distances)
        return torch.sum(entropies)


    def _entropic_function(self, x: Tensor) -> Tensor:
        if self.fn_type == EntropicFunctionType.LOG:
            return torch.log(x + self.eps)
        elif self.fn_type == EntropicFunctionType.ENTROPY:
            return -x * torch.log(x + self.eps)
        elif self.fn_type == EntropicFunctionType.EXPONENTIAL:
            return -torch.exp(-x)
        elif self.fn_type == EntropicFunctionType.POWER:
            return torch.pow(x, self.power_fn_exponent)
        elif self.fn_type == EntropicFunctionType.IDENTITY:
            return x
        else:
            raise ValueError("Entropic function type not found.")

