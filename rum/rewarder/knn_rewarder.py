from rum.rewarder.rewarder import Rewarder
from rum.density import KNNDensityEstimator
from dataclasses import dataclass

from stable_baselines3.common.buffers import RolloutBuffer
from torch import FloatTensor
import torch
import os


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
    ):
        super().__init__()
        self.torch_env = torch_env
        self.rewarder_params = rewarder_params
        self.knn = KNNDensityEstimator(**knn_params.__dict__)
        self.entropy_buff = 0.0  # store previous entropy

        if torch_env.device == torch.device("cuda"):
            self.num_cuda_cores_per_device = 1024  # varies by GPU
            self.num_threads = (
                torch.cuda.device_count() * self.num_cuda_cores_per_device
            )
        else:
            # Use the number of CPU cores if on CPU
            self.num_threads = os.cpu_count()

    def reward(self, buffer: RolloutBuffer) -> FloatTensor:
        if not isinstance(buffer, RolloutBuffer):
            raise TypeError("Buffer must be an instance of RolloutBuffer")
        n_steps, n_envs = buffer.buffer_size, buffer.num_envs
        states = buffer.observations.view(n_steps * n_envs, self.knn.dim_states)
        return self._reward(states).view(n_steps, n_envs)

    def _reward(self, states: torch.Tensor) -> torch.Tensor:
        if not isinstance(states, torch.Tensor) or states.dim() != 2:
            raise ValueError("States must be of shape (batch_size, state_dim)")
        entropy = self.knn.entropy(states)
        return self._compute_reward(entropy)

    def learn(self, states: torch.Tensor) -> None:
        self.knn.update_replay_buffer(states)
        return None

    def _compute_reward(self, entropy: torch.Tensor) -> torch.Tensor:
        r = entropy
        reward = r - self.entropy_buff if self.rewarder_params.differential else r
        if self.rewarder_params.differential:
            self.entropy_buff = r
        return reward
