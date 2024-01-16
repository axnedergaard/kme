import numpy as np
import itertools
import scipy
import gymnasium
from ..density import Density
from ..geometry import Geometry
from ..geometry import EuclideanGeometry
from .util import sphere_sample_uniform

# TODO. Sampling multiple points in one call for Euclidean and Spherical.
# TODO. Proper implementation of hypersphere and torus (support d!=2 and use multiple charts to avoid singularities.
# TODO. Fix sampling in torus.

class Chart():
  def __init__(self, map_, inverse_map, differential_map):
    self.domain = None
    self.map = map_
    self.inverse_map = inverse_map
    self.differential_map = differential_map

class Manifold(gymnasium.Env, Density, Geometry):
  def __init__(self, dim, ambient_dim):
    self.dim = dim
    self.ambient_dim = ambient_dim 
    #self.max_step_size = 0.01 # Propertion of diameter of space.
    self.max_step_size = 0.1 # Propertion of diameter of space.
    self.observation_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=[self.ambient_dim]) 
    self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=[self.dim])
    self.diameter = np.sqrt(8) # Movement in ambient space.

  def reset(self, seed=None):
    self.state = self.starting_state()
    info = {}
    return self.state.copy(), info
  
  def step(self, action):
    # Warning: Undefined behavior if reset not called before this.
    self.state = self._manifold_step(self.state, action)
    self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)
    reward = 0.0
    terminated = False
    truncated = False
    info = {}
    return self.state.copy(), reward, terminated, truncated, info

  def random_walk(self, n, starting_state=None, step_size=None):
    state = starting_state if starting_state is not None else self.starting_state() 
    samples = []
    for i in range(n):
      prob_state = self.pdf(state)
      accepted = False
      while not accepted:
        change_state = sphere_sample_uniform(self.dim - 1) 
        new_state = self._manifold_step(state, change_state, step_size if step_size is not None else self.max_step_size)
        prob_new_state = self.pdf(new_state)
        if np.random.uniform() < prob_new_state / prob_state:
          state = new_state 
          accepted = True
      samples.append(state)
    return np.array(samples)

  def _manifold_step(self, state, action):
    # The manifold can define the retraction through charts or directly (e.g. when a hack is easier).
    action_scaling = self.diameter * self.max_step_size # TODO. Can optimize by storing this.
    chart_index = self.get_chart_index(state)
    if chart_index != -1: 
      chart = self.charts[chart_index]
      local = chart.map(state)
      local += chart.differential_map(state, action_scaling * action)
      state = chart.inverse_map(local)
      return state
    else:
      return self.retraction(state, action_scaling * action)

  def get_chart_index(self, state):
    return -1

  def retraction(self, state, action):
    return state + action # Movement in ambient space.

  def _metric_size(self, state, action):
    return np.matmul(action, np.matmul(self.metric_tensor(state), action.T)) 

  def starting_state(self):
    raise NotImplementedError

  def pdf(self, x):
    raise NotImplementedError

  def sample(self):
    raise NotImplementedError

  def distance_function(self, x, y):
    raise NotImplementedError

  def metric_tensor(self):
    raise NotImplementedError

  def implicit_function(self, c):
    raise NotImplementedError

  def learn(self, _):
    pass
