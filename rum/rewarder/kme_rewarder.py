from .rewarder import Rewarder
from ..density import OnlineKMeansEstimator
from typing import Callable, Union, Optional, Tuple, Literal
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
        density: OnlineKMeansEstimator,
        # KME REWARDER HYPERPARAMS
        differential: bool = True,
        # TORCH
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
        # SB3
        concurrent: bool = True,
    ) -> None:
        super(KMERewarder, self).__init__(concurrent)

        # torch env settings
        self.device = device
        self.dtype = dtype

        self.differential: bool = differential
        self.entropy_buff = 0.0 # store previous entropy
        self.pdf_approx_buff = 0.0 # store previous pdf approx

        # Underlying kmeans density estimator
        self.kmeans = density 


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
            diameters = self.kmeans.simulate_step(state)
            entropy_lb = self.kmeans.entropy_lb(diameters)
            reward = entropy_lb - self.entropy_buff if self.differential else entropy_lb
            if self.differential: self.entropy_buff = entropy_lb
            return reward
        
        def reward_information(state: Tensor) -> FloatTensor:
            diameters = self.kmeans.diameters
            pdf_approx = self.kmeans.pdf_approx(state, diameters)
            reward = pdf_approx - self.pdf_approx_buff if self.differential else pdf_approx
            if self.differential: self.pdf_approx_buff = pdf_approx
            return reward

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
        self._learn(states)


    def _learn(self, states: Tensor) -> None:
        if not isinstance(states, Tensor) or states.dim() != 2:
            raise ValueError("States must be of shape (B, dim_states)")
        self.kmeans.learn(states)
