"""Wrapper to use environments as gymnasium environments without render support."""

from collections import OrderedDict
import dm_env
import gymnasium as gym
import numpy as np

def flatten_observation(observation):
    """Flatten dicts of numpy arrays of observations into list of observations."""
    def _flatten_recursive(observation):
        flattened = []
        if isinstance(observation, OrderedDict):
            for key in observation:
                flattened += flatten_observation(observation[key])
        elif isinstance(observation, np.ndarray):
            if observation.size == 1:
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
        self.action_space = self._action_space_dmc2gym(env.action_spec())
        self.observation_space = self._observation_space_dmc2gym(env.observation_spec())
        self.metadata = {'render.modes': ['rgb_array']}

    def step(self, action):
        time_step = self.env.step(action)
        observation = self._observation_dmc2gym(time_step.observation)
        reward = time_step.reward
        terminated = (time_step.step_type == dm_env.StepType.LAST)
        truncated = False  # Assuming there's no truncation in dm_control
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None):
        time_step = self.env.reset()
        observation = self._observation_dmc2gym(time_step.observation)
        return observation, {}  # Added an empty info dictionary as per new standards

    def render(self, mode='rgb_array', camera=0):
        raise NotImplementedError

    @staticmethod
    def _observation_space_dmc2gym(observation_spec):
        n_observations = 0

        for observation in observation_spec.values():
            shape = observation.shape
            size = int(np.prod(shape))
            n_observations += size

        low = np.array([-np.inf] * n_observations)
        high = np.array([np.inf] * n_observations)

        return gym.spaces.Box(low, high, dtype=np.float64)

    @staticmethod
    def _action_space_dmc2gym(action_spec):
        low = action_spec.minimum
        high = action_spec.maximum

        return gym.spaces.Box(low, high, dtype=np.float64)

    @staticmethod
    def _observation_dmc2gym(observation):
        flattened = flatten_observation(observation)
        return np.array(flattened)
