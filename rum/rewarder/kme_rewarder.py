from .rewarder import Rewarder
from ..density import OnlineKMeansEstimator
from typing import Callable, Union, Optional, Tuple
from torch import Tensor, FloatTensor, LongTensor
import concurrent.futures
from enum import Enum
import numpy as np
import torch
import os


class KMERewarder(Rewarder):

    def __init__(
        self,
        # KM DENSITY ESTIMATOR
        k: int,
        dim_states: int,
        learning_rate: float,
        balancing_strength: float,
        distance_func: Callable = None,
        origin: Union[Tensor, np.ndarray] = None,
        init_method: str = 'uniform',
        homeostasis: bool = True,
        # KME REWARDER HYPERPARAMS
        entropic_func: str = 'exponential',
        power_fn_exponent: float = 0.5,
        differential: bool = True,
        eps: Optional[float] = 1e-9,
        # TORCH
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()

        self.device = device
        self.dtype = dtype

        if device == torch.device('cuda'):
            self.num_cuda_cores_per_device = 1024  # varies by GPU
            self.num_threads = torch.cuda.device_count() * self.num_cuda_cores_per_device
        else:
            # Use the number of CPU cores if on CPU
            self.num_threads = os.cpu_count()

        self.k = k
        self.differential: bool = differential
        self.entropy_buff = 0.0 # store previous entropy

        # underlying kmeans density estimator
        self.kmeans = OnlineKMeansEstimator(
            k, 
            dim_states, 
            homeostasis=homeostasis,
            learning_rate=learning_rate, 
            balancing_strength=balancing_strength,
            distance_func=distance_func, 
            origin=origin, 
            init_method=init_method, 
        )


    def reward_function(self, states: Tensor) -> FloatTensor:
        if states.dim() == 2:
          return self._reward_function(states)
        else:
          num_steps, num_envs, num_dims = states.shape
          states = states.view(num_steps * num_envs, num_dims)
          rewards = self._reward_function(states)
          return rewards.view(num_steps, num_envs)


    def _reward_function(self, states: Tensor) -> FloatTensor:
        if not isinstance(states, Tensor): #or states.dim() != 2:
            raise ValueError("States must be of shape (B, dim_states)")

        def f(state: Tensor) -> FloatTensor:
            #km, assign_idx, diameters = self.kmeans.simulate_step(state)
            diameters = self.kmeans.simulate_step(state)
            entropy_lb = self.kmeans.entropy_lb(diameters)
            return self._compute_reward(entropy_lb)
        

        rewards = torch.zeros(states.size(0)) # shape: (B,)
        #num_threads = min(states.size(0), self.num_threads)
        #with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        #    for i, rew in enumerate(executor.map(lambda s: f(s), states)):
       #         rewards[i] = rew
        for i, state in enumerate(states):
          rewards[i] = f(state)
        
        return rewards # shape: (B,)


    def _compute_reward(self, entropy_lb: Tensor) -> FloatTensor:
        r = entropy_lb
        reward = r - self.entropy_buff if self.differential else r
        if self.differential: self.entropy_buff = r
        return reward


    def learn(self, states: Tensor) -> FloatTensor:
        self._learn(states)


    def _learn(self, states: Tensor) -> None:
        if not isinstance(states, Tensor) or states.dim() != 2:
            raise ValueError("States must be of shape (B, dim_states)")
        self.kmeans.learn(states)
        # shuffle_idx = torch.randperm(states.size(0))
        # shuffled_states = states[shuffle_idx]

