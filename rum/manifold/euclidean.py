import numpy as np
import itertools
from .manifold import Manifold, GlobalChartAtlas

class EuclideanManifold(Manifold):
  def __init__(self, dim, sampler):
    super(EuclideanManifold, self).__init__(dim, dim)
    self.sampler = sampler 
    identity = lambda x: x
    self.atlas = GlobalChartAtlas(identity, identity, self.norm, identity, identity)

  def retraction(self, p, v):
    return self.step_within_ball(p, v)

  def norm(self, p, v):
    return np.linalg.norm(v)

  def starting_state(self):
    return np.zeros(self.dim)

  def pdf(self, p):
    if self.sampler['type'] == 'uniform':
      if np.all(p >= self.sampler['low']) and np.all(p <= self.sampler['high']):
        return 1 / np.prod(self.sampler['high'] - self.sampler['low'])
      else:
        return 0.0
    elif self.sampler['type'] == 'gaussian':
      return np.exp(-np.sum((p - self.sampler['mean'])**2) / (2 * self.sampler['std']**2)) / (np.sqrt((2 * np.pi)**self.dim) * self.sampler['std'])

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

  def implicit_function(self, p):
    if self.dim >= 3:
      raise ValueError
    else:
      return 0.0

  @staticmethod
  def distance_function(p, q):
    return np.linalg.norm(p - q)
