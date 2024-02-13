import numpy as np
import itertools
import scipy
import gymnasium
from ..density import Density
from ..geometry import Geometry
from ..geometry import EuclideanGeometry
from .util import sphere_sample_uniform

class Chart():
  def __init__(self, map_, inverse_map, norm, differential_map=None, differential_inverse_map=None):
    self.map = map_
    self.inverse_map = inverse_map
    self.norm = norm
    self.differential_map = differential_map
    self.differential_inverse_map = differential_inverse_map

class Atlas():
  def get_chart(self, p):
    raise NotImplementedError

class GlobalChartAtlas(Atlas):
  def __init__(self, _map, inverse_map, norm, differential_map=None, differential_inverse_map=None):
    self.chart = Chart(_map, inverse_map, norm, differential_map, differential_inverse_map)

  def get_chart(self, p):
    return self.chart

class GeodesicManifold():
  # Wrapper for manifold object.
  def __init__(self, base_object, *args, **kwargs):
    self.__dict__['base_object'] = base_object # Avoid infinite recursion on __setattr__().
    # Action space is rotation and thrust.
    max_angle = np.pi / 4
    self.action_space = gymnasium.spaces.Box(low=np.array([-1.0] + [-max_angle] * (self.dim - 1)), high=np.array([1.0] + [max_angle] * (self.dim - 1))) #, shape=[self.dim])
    #self.velocity = np.array([1.0] + [0.0] * (self.dim - 1))
    self.velocity = sphere_sample_uniform(self.dim - 1)[0]

  def parallel_transport(self, previous_state, state, vector):
    # If the chart is unchanged, parallel transport in the coordinate bases is represented by the identity matrix. For changed charts, we must perform the change of basis at a given point.
    if np.all(vector == 0): # No transport needed.
      return vector

    previous_chart = self.atlas.get_chart(previous_state)
    chart = self.atlas.get_chart(state)

    # Coordinate transformation is not necessary if the charts are the same.
    if previous_chart == chart: # No transport needed.
      return vector

    state_local = chart.map(state)
    matrix_1 = previous_chart.differential_inverse_map(state_local)
    matrix_2 = chart.differential_map(state)
    transform = np.matmul(matrix_2, matrix_1)

    transported_vector = np.matmul(transform, vector)
    
    # Renormalization is only necessary due to approximation errors.
    #if np.any(transported_vector != 0):
    #  transported_vector /= np.linalg.norm(transported_vector)

    return transported_vector

  def rotate(self, vector, angle):
    rotation = np.array([
      [np.cos(angle), -np.sin(angle)],
      [np.sin(angle), np.cos(angle)]
    ])
    return np.matmul(rotation, vector)

  def step(self, action):
    previous_state = self.state.copy()
    self.velocity = self.rotate(self.velocity, action[1])
    gym_return_values = self.base_object.step(self.velocity * action[0])
    self.velocity = self.parallel_transport(previous_state, self.state, self.velocity)
    return gym_return_values

  def __getattr__(self, name):
    return getattr(self.base_object, name)

  def __setattr__(self, name, value):
    return setattr(self.base_object, name, value)

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
    self.state = self.manifold_step(self.state, action, self.max_step_size)
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
        change_state = sphere_sample_uniform(self.dim - 1)[0]
        updated_state = self.manifold_step(state, change_state, step_size if step_size is not None else self.max_step_size)
        prob_updated_state = self.pdf(updated_state)
        if np.random.uniform() < prob_updated_state / prob_state:
          state = updated_state 
          accepted = True
      samples.append(state)
    return np.array(samples)

  def manifold_step(self, state, action, step_size):
    return self.retraction(state, step_size * action)

  def retraction(self, p, v):
    # Warning: Relies on self.atlas being defined.
    # Can be overriden for efficiency.
    if self.atlas is not None:
      chart = self.atlas.get_chart(p) 
      local = chart.map(p)
      local += self.normalize(p, v, chart.norm) 
      p = chart.inverse_map(local)
      return p
    else:
      raise NotImplementedError("Must define retraction or atlas.")

  def norm(self, p, v): # Riemannian norm.
    # Warning: Relies on self.metric_tensor() being implemented. 
    # Can be overridden for efficiency.
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

  def starting_state(self):
    raise NotImplementedError

  def pdf(self, p):
    raise NotImplementedError

  def sample(self):
    raise NotImplementedError

  def grid(self, n):
    raise NotImplementedError

  def implicit_function(self, p):
    raise NotImplementedError

  def distance_function(self, p, q):
    # Terence Tao does not know how to do this for surfaces with non-constant curvature: https://mathoverflow.net/questions/37651/riemannian-surfaces-with-an-explicit-distance-function
    raise NotImplementedError

  def interpolate(self, p, q, alpha):
    # See comment in distance_function.
    raise NotImplementedError

  def learn(self, _):
    pass
