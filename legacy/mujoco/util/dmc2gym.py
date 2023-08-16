"""Wrapper to use environments as gym environments without render support."""

from collections import OrderedDict
import numpy as np
import dm_env
from gym import core, spaces

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

class GymWrapper(core.Env):

    def __init__(self, env):
        self.env = env
        self.action_space = self._action_space_dm2gym(env.action_spec())
        self.observation_space = self._observation_space_dm2gym(
            env.observation_spec())
        self.metadata = {'render.modes': ['rgb_array']}

    def step(self, action):
        time_step = self.env.step(action)

        observation = self._observation_dm2gym(time_step.observation)
        reward = time_step.reward
        done = (time_step.step_type == dm_env.StepType.LAST)
        info = {}
        #info = self.env._task.info

        return observation, reward, done, info

    def reset(self):
        time_step = self.env.reset()
        observation = self._observation_dm2gym(time_step.observation)
        return observation

    def render(self, mode='rgb_array', camera=0):
        raise NotImplementedError
        #return self.env._physics.render(240, 320, camera)

    @staticmethod
    def _observation_space_dm2gym(observation_spec):
        n_observations = 0

        for observation in observation_spec.values():
            shape = observation.shape
            size = 1
            for dimension in shape:
                size *= dimension
            n_observations += size

        low = np.array([-np.inf] * n_observations)
        high = np.array([np.inf] * n_observations)
        shape = None

        return spaces.Box(low, high, shape, dtype=np.float64)

    @staticmethod
    def _action_space_dm2gym(action_spec):
        low = action_spec.minimum
        high = action_spec.maximum
        shape = None

        return spaces.Box(low, high, shape, dtype=np.float64)

    @staticmethod
    def _observation_dm2gym(observation):
        flattened = flatten_observation(observation)
        return np.array(flattened)
