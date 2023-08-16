"""Wrapper to use environments as gymnasium environments without render support."""

from collections import OrderedDict
import torch
import dm_env
import gymnasium as gym

def flatten_observation(observation):
    """Flatten dicts of tensors of observations into list of observations."""
    def _flatten_recursive(observation):
        flattened = []
        if isinstance(observation, OrderedDict):
            for key in observation:
                flattened += flatten_observation(observation[key])
        elif isinstance(observation, torch.Tensor):
            if observation.numel() == 1:
                return [observation.item()]
            else:
                for obs in observation:
                    flattened += flatten_observation(obs)
        else:
            return [observation]

        return flattened

    return _flatten_recursive(observation)

class GymnasiumWrapper(gym.Env):

    def __init__(self, env):
        self.env = env
        self.action_space = self._action_space_dm2gymnasium(env.action_spec())
        self.observation_space = self._observation_space_dm2gymnasium(env.observation_spec())
        self.metadata = {'render.modes': ['rgb_array']}

    def step(self, action):
        time_step = self.env.step(action)
        observation = self._observation_dm2gymnasium(time_step.observation)
        reward = time_step.reward
        terminated = (time_step.step_type == dm_env.StepType.LAST)
        truncated = False  # Assuming there's no truncation in dm_control
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        time_step = self.env.reset()
        observation = self._observation_dm2gymnasium(time_step.observation)
        return observation, {}  # Added an empty info dictionary as per new standards

    def render(self, mode='rgb_array', camera=0):
        raise NotImplementedError

    @staticmethod
    def _observation_space_dm2gymnasium(observation_spec):
        n_observations = 0

        for observation in observation_spec.values():
            shape = observation.shape
            size = torch.prod(torch.tensor(shape))
            n_observations += size

        low = torch.full((n_observations,), -float('inf'))
        high = torch.full((n_observations,), float('inf'))

        return gym.spaces.Box(low, high, dtype=torch.float64)

    @staticmethod
    def _action_space_dm2gymnasium(action_spec):
        low = torch.tensor(action_spec.minimum)
        high = torch.tensor(action_spec.maximum)

        return gym.spaces.Box(low, high, dtype=torch.float64)

    @staticmethod
    def _observation_dm2gymnasium(observation):
        flattened = flatten_observation(observation)
        return torch.tensor(flattened)
