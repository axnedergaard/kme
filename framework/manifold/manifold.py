import numpy as np
import scipy

# TODO. Sampling multiple points in one call for Euclidean and Spherical.
# TODO. Proper definition of state and action space.
# TODO. Proper implementation of hypersphere (support d!=2 and use multiple charts to avoid singularities.
# TODO. More manifolds.

class Chart():
  # We assume chart domains and images are balls.
  def __init__(self, domain_center, domain_radius, image_radius, map_, inverse_map, distance_function):
    self.distance_function = distance_function 
    self.domain_center = domain_center 
    self.image_center = map_(domain_center)
    self.domain_radius = domain_radius
    self.image_radius = image_radius
    self.map = map_
    self.inverse_map = inverse_map
    self.distance_function = distance_function

  def domain_contains(self, x):
    return self.distance_function(x, self.domain_center) < self.domain_radius

  def image_contains(self, x):
    return EuclideanManifold.distance(x, self.image_center) < self.image_radius

#class Manifold(Environment, Density, Distance):
class Manifold():
  def __init__(self):
    self.max_step_size = 0.01 # TODO.
  
  def step(self, action):
    # Warning: Undefined behavior if reset not called before this.
    self.state = self._manifold_step(self.state, action, self.max_step_size)
    return self.state.copy()

  def random_walk(self, n_samples, starting_state=None, step_size=None):
    state = starting_state if starting_state is not None else np.array([1.0] + [0.0] * (self.ambient_dim - 1))
    samples = []
    for i in range(n_samples):
      prob_state = self.pdf(state)
      accepted = False
      while not accepted:
        change_state = SphericalManifold._sample_uniform(self.dim - 1)
        new_state = self._manifold_step(state, change_state, step_size if step_size is not None else self.max_step_size)
        prob_new_state = self.pdf(new_state)
        if np.random.uniform() < prob_new_state / prob_state:
          state = new_state 
          accepted = True
      samples.append(state)
    return np.array(samples)
  
  def _manifold_step(self, state, action, max_step_size):
    action_euclidean_size = np.linalg.norm(action)
    action_metric_size = self._metric_size(state, action) # This is a crude approximation.
    change_local_state = max_step_size * (action_euclidean_size / action_metric_size) * action
    # Search for a chart compatible with the step.
    for chart in self.charts:
      if chart.domain_contains(state):
        local_state = chart.map(state) 
        cand_local_state = local_state + change_local_state 
        if chart.image_contains(cand_local_state):
          return chart.inverse_map(cand_local_state)
    raise Exception("No compatible chart found.")

  def _metric_size(self, state, action):
    return np.matmul(action, np.matmul(self.metric_tensor(state), action.T)) 

class EuclideanManifold(Manifold):
  def __init__(self, dim, low, high, sampler):
    super(EuclideanManifold, self).__init__()

    self.dim = dim
    self.ambient_dim = dim
    self.low = low
    self.high = high
    self.state_dim = dim
    self.action_dim = dim
    self.sampler = sampler 

    self.charts = [
      Chart(
        domain_center=np.zeros(self.dim),
        domain_radius=np.inf,
        image_radius=np.inf,
        map_=lambda x: x,
        inverse_map=lambda x: x,
        distance_function=self.distance
      )
    ]

    if self.sampler['type'] == 'uniform':
      assert self.low <= self.sampler['low'] and self.sampler['high'] <= self.high

  def reset(self):
    self.state = np.zeros(self.dim)
    return self.state

  def sample(self, n_samples):
    if n_samples > 1: # TODO.
      return np.array([self.sample(1) for _ in range(n_samples)])
    if self.sampler['type'] == 'uniform':
      return np.random.uniform(self.sampler['low'], self.sampler['high'], self.dim)
    elif self.sampler['type'] == 'gaussian':
      return np.random.normal(self.sampler['mean'], self.sampler['std'], self.dim)

  def pdf(self, x):
    if self.sampler['type'] == 'uniform':
      if np.all(x >= self.sampler['low']) and np.all(x <= self.sampler['high']):
        return 1 / np.prod(self.sampler['high'] - self.sampler['low'])
      else:
        return 0.0
    elif self.sampler['type'] == 'gaussian':
      return np.exp(-np.sum((x - self.sampler['mean'])**2) / (2 * self.sampler['std']**2)) / (np.sqrt((2 * np.pi)**self.dim) * self.sampler['std'])

  @staticmethod
  def distance(x, y):
    return np.linalg.norm(x - y)

  def metric_tensor(self, x):
    return np.eye(self.dim)

class SphericalManifold(Manifold):
  # We assume unit radius.
  def __init__(self, dim, sampler):
    assert dim == 2 # TODO.
    super(SphericalManifold, self).__init__()
    self.dim = dim
    self.ambient_dim = dim + 1
    self.state_dim = self.ambient_dim
    self.action_dim = self.dim
    self.sampler = sampler

    # TODO. Right now this only works for dim=2 and has a singular point.
    self.charts = [
      Chart(
        domain_center=np.array([0.0, 1.0, 0.0]),
        domain_radius=np.inf, # TODO
        image_radius=np.inf, # TODO
        map_=self._cartesian_to_spherical,
        inverse_map=self._spherical_to_cartesian,
        distance_function=self.distance
      )
    ]

  def reset(self):
    self.state = np.zeros(self.dim + 1)
    self.state[0] = 1.0
    return self.state

  def sample(self, n_samples):
    if n_samples > 1: # TODO.
      return np.array([self.sample(1) for _ in range(n_samples)])
    if self.sampler['type'] == 'uniform':
      return self._sample_uniform(self.dim)
    elif self.sampler['type'] == 'vonmises_fisher':
      return scipy.stats.vonmises_fisher.rvs(self.sampler['mu'], self.sampler['kappa'], size=1)[0]
      
  def pdf(self, x):
    if self.sampler['type'] == 'uniform':
      if np.linalg.norm(x) == 1: 
        return 1 / (2 * np.pi)**(self.dim / 2) 
      else:
        return 0.0
    elif self.sampler['type'] == 'vonmises_fisher':
      return scipy.stats.vonmises_fisher.pdf(x, self.sampler['mu'], self.sampler['kappa'])

  def distance(self, x, y):
    return np.arccos(np.dot(x, y))

  def metric_tensor(self, x):
    return np.array([
        [1.0, 0.0],
        [0.0, np.sin(x[0])**2]
      ])

  def _cartesian_to_spherical(self, x): 
    theta = np.arccos(x[0])
    phi = np.arctan2(x[1], x[2])
    return np.array([theta, phi])

  def _spherical_to_cartesian(self, _x): 
    x = np.sin(_x[0]) * np.cos(_x[1])
    y = np.sin(_x[0]) * np.sin(_x[1])
    z = np.cos(_x[0])
    return np.array([x, y, z])

  @staticmethod
  def _sample_uniform(dim):
    x = np.random.normal(0, 1, dim + 1)
    return x / np.linalg.norm(x)
