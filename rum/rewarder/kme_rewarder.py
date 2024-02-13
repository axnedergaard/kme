from rum.rewarder.rewarder import Rewarder
from rum.density import OnlineKMeansEstimator
from torch import Tensor, FloatTensor
from typing import Literal
import torch


class KMERewarder(Rewarder):

    def __init__(
        self,
        # KM DENSITY ESTIMATOR
        density: OnlineKMeansEstimator,
        # KME REWARDER HYPERPARAMS
        differential: bool = True,
        # TORCH
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        # SB3
        concurrent: bool = True,
    ) -> None:
        super(KMERewarder, self).__init__(concurrent)

        # torch env settings
        self.device = device
        self.dtype = dtype

        # Underlying kmeans density estimator
        self.kmeans = density
        self.differential: bool = differential

    def reward_function(self, states: Tensor) -> FloatTensor:
        if states.dim() == 2:
            return self._reward_function(states)
        else:
            num_steps, num_envs, num_dims = states.shape
            states = states.view(num_steps * num_envs, num_dims)
            rewards = self._reward_function(states)
            return rewards.view(num_steps, num_envs)

    def _reward_function(
        self, states: Tensor, form: Literal["entropy", "information"] = "entropy"
    ) -> FloatTensor:
        if not isinstance(states, Tensor):
            raise ValueError("States must be of shape (B, dim_states)")

        def reward_entropy(state: Tensor) -> FloatTensor:
            diameters = self.kmeans.simulate_step(state)
            entropy_lb = self.kmeans.entropy_lb(diameters)
            if self.differential:
                entropy_lb_before = self.kmeans.entropy_lb(self.kmeans.diameters)
                return entropy_lb - entropy_lb_before
            return entropy_lb

        def reward_information(state: Tensor) -> FloatTensor:
            diameters = self.kmeans.diameters
            information = self.kmeans.information(state, diameters)
            return information

        rewards = torch.zeros(states.size(0))  # shape: (B,)

        for i, state in enumerate(states):
            if form == "entropy":
                rewards[i] = reward_entropy(state)
            elif form == "information":
                rewards[i] = reward_information(state)
            else:
                raise ValueError("form must be either 'entropy' or 'information'")

        return rewards  # shape: (B,)

    def learn(self, states: Tensor) -> FloatTensor:
        if not isinstance(states, Tensor) or states.dim() != 2:
            raise ValueError("States must be of shape (B, dim_states)")
        self.kmeans.learn(states)
