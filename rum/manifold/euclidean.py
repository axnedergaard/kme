import numpy as np
import itertools
from .manifold import Manifold

class EuclideanManifold(Manifold):
  def __init__(self, dim, sampler):
    super(EuclideanManifold, self).__init__(dim, dim)
    self.low = -1.0 
    self.high = 1.0 
    self.sampler = sampler 

    if self.sampler['type'] == 'uniform':
      assert self.low <= self.sampler['low'] and self.sampler['high'] <= self.high

  def retraction(self, p, v):
    return self.step_within_ball(p, v)

  def starting_state(self):
    return np.zeros(self.dim)

  def pdf(self, x):
    if self.sampler['type'] == 'uniform':
      if np.all(x >= self.sampler['low']) and np.all(x <= self.sampler['high']):
        return 1 / np.prod(self.sampler['high'] - self.sampler['low'])
      else:
        return 0.0
    elif self.sampler['type'] == 'gaussian':
      return np.exp(-np.sum((x - self.sampler['mean'])**2) / (2 * self.sampler['std']**2)) / (np.sqrt((2 * np.pi)**self.dim) * self.sampler['std'])

  def sample(self, n):
    if n > 1: # TODO.
      return np.array([self.sample(1) for _ in range(n)])
    if self.sampler['type'] == 'uniform':
      return np.random.uniform(self.sampler['low'], self.sampler['high'], self.dim)
    elif self.sampler['type'] == 'gaussian':
      return np.random.normal(self.sampler['mean'], self.sampler['std'], self.dim)

  def grid(self, n):
    # TODO: Could be more efficient.
    n_per_dim = int(np.power(n, 1.0 / self.dim))
    points = [] #= np.zeros([self.dim, n_per_dim])
    for point in itertools.product(np.linspace(-1.0, 1.0, n_per_dim), repeat=self.dim):
      norm = np.linalg.norm(point)
      if norm <= 1.0:
        points.append(point)
    points = np.array(points)
    return points

  @staticmethod
  def distance_function(x, y):
    return np.linalg.norm(x - y)

  def metric_tensor(self, x):
    return np.eye(self.dim)

  def implicit_function(self, c):
    if self.dim >= 3:
      raise ValueError
    else:
      return 0.0
