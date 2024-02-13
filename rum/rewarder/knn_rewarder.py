from rum.rewarder.rewarder import Rewarder
from rum.density import KNNDensityEstimator
from dataclasses import dataclass
from torch import Tensor, FloatTensor
from typing import Literal
import torch


@dataclass
class KNNDensityEstimatorParams:
    k: int
    dim_states: int  # state_dim
    constant_c: float = 1e-3


@dataclass
class KNNRewarderParams:
    differential: bool = True


@dataclass
class TorchEnvParams:
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32


class KNNRewarder(Rewarder):

    def __init__(
        self,
        knn_params: KNNDensityEstimatorParams,
        rewarder_params: KNNRewarderParams,
        torch_env: TorchEnvParams,
        concurrent: bool = True,
    ):
        super(KNNRewarder, self).__init__(concurrent)
        self.torch_env = torch_env
        self.rewarder_params = rewarder_params
        self.knn = KNNDensityEstimator(**knn_params.__dict__)

    def reward_function(self, states: Tensor) -> FloatTensor:
        if states.dim() == 2:
            return self._reward_function(states)
        else:
            num_steps, num_envs, num_dims = states.shape
            states = states.view(num_steps * num_envs, num_dims)
            rewards = self._reward_function(states)
            return rewards.view(num_steps, num_envs)
        
    def _reward_function(self, states: Tensor, form: Literal['entropy', 'information'] = 'entropy') -> FloatTensor:
        if not isinstance(states, Tensor):
            raise ValueError("States must be of shape (B, dim_states)")
        
        def reward_entropy(state: Tensor) -> FloatTensor:
            distances = self.knn.simulate_step(state)
            entropy_approx = self.knn.entropy_approx(distances)
            if self.rewarder_params.differential:
                entropy_approx_before = self.knn.entropy_approx()
                return entropy_approx - entropy_approx_before
            return entropy_approx            
        
        def reward_information(state: Tensor) -> FloatTensor:
            information = self.knn.information(state)
            return information

        rewards = torch.zeros(states.size(0)) # shape: (B,)

        for i, state in enumerate(states):
            if form == 'entropy':
                rewards[i] = reward_entropy(state)
            elif form == 'information':
                rewards[i] = reward_information(state)
            else:
                raise ValueError("form must be either 'entropy' or 'information'")
        
        return rewards # shape: (B,)        
    
    def learn(self, states: Tensor) -> FloatTensor:
        if not isinstance(states, Tensor) or states.dim() != 2:
            raise ValueError("States must be of shape (B, dim_states)")
        self.knn.learn(states)
