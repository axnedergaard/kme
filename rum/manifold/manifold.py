import numpy as np
import itertools
import scipy
import gymnasium
from ..density import Density
from ..geometry import Geometry
from ..geometry import EuclideanGeometry
from .util import sphere_sample_uniform

# TODO. Sampling multiple points in one call for Euclidean and Spherical.
# TODO. Fix sampling in torus.

class Chart():
  def __init__(self, map_, inverse_map, norm):
    self.map = map_
    self.inverse_map = inverse_map
    self.norm = norm

class Atlas():
  def get_chart(self, p):
    raise NotImplementedError

class Manifold(gymnasium.Env, Density, Geometry):
  def __init__(self, dim, ambient_dim):
    self.dim = dim
    self.ambient_dim = ambient_dim 
    self.max_step_size = 0.2 
    self.observation_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=[self.ambient_dim]) 
    self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=[self.dim])
    self.atlas = None

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
        updated_state = self._manifold_step(state, change_state, step_size if step_size is not None else self.max_step_size)
        prob_updated_state = self.pdf(updated_state)
        if np.random.uniform() < prob_updated_state / prob_state:
          state = updated_state 
          accepted = True
      samples.append(state)
    return np.array(samples)

  def _manifold_step(self, state, action):
    # The manifold can define the retraction through charts or directly (e.g. when a hack is easier).
    scaled_action = self.max_step_size * action
    if self.atlas is not None:
      chart = self.atlas.get_chart(state) 
      local = chart.map(state)
      local += self.normalize(state, scaled_action, chart.norm) 
      state = chart.inverse_map(local)
      return state
    else:
      return self.retraction(state, scaled_action)

  def norm(self, p, v): # Riemannian norm.
    # Warning: Relies on self.metric_tensor() being implemented. Can be overridden for efficiency.
    return np.sqrt(np.matmul(v, np.matmul(self.metric_tensor(p), v.T)))

  def normalize(self, p, v, norm=None):
    # Normalize according to direction and position using Riemannian norm.
    norm = self.norm if norm is None else norm # Use norm from metric if not provided.
    if np.any(v != 0.0):
      return v * np.linalg.norm(v) / norm(p, v) 
    else:
      return v 

  def step_within_ball(self, p, v):
    updated_p = p + v
    norm = np.linalg.norm(updated_p)
    if norm > 1.0: # Take maximal step such that we remain within boundary.
      #updated_p = (p + v) / norm
      step_size = np.roots([np.dot(v, v), 2 * np.dot(p, v), np.dot(p, p) - 1]) # Roots of equation ||p + step_size * v|| = 1.
      step_size = step_size[np.argmin(np.abs(step_size))] # One root will be close to 1, the other will have large magnitude.
      updated_p = p + step_size * v
    return updated_p 

  def retraction(self, p, v):
    raise NotImplementedError

  def starting_state(self):
    raise NotImplementedError

  def pdf(self, p):
    raise NotImplementedError

  def sample(self):
    raise NotImplementedError

  def distance_function(self, p, q):
    raise NotImplementedError

  def metric_tensor(self, p):
    raise NotImplementedError

  def implicit_function(self, p):
    raise NotImplementedError

  def learn(self, _):
    pass
