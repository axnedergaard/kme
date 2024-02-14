import numpy as np
import itertools
from .manifold import Manifold, GlobalChartAtlas
from rum.geometry import EuclideanGeometry 


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
    if self.sampler['name'] == 'uniform':
      if np.linalg.norm(p) <= 1.0:
        return 1.0 / np.prod(self.sampler['high'] - self.sampler['low'])
      else:
        return 0.0
    elif self.sampler['name'] == 'gaussian': # Isotropic Gaussian.
      return np.exp(-np.sum((p - self.sampler['mean']) ** 2) / (2 * self.sampler['std'] ** 2)) / ((2 * np.pi) ** (self.dim / 2.0) * self.sampler['std'])
    else:
      raise ValueError(f'Unknown sampler: {self.sampler["name"]}')

  def sample(self, n):
    if self.sampler['name'] == 'uniform': # TODO. This not correct as it is uniform on unit cube, not ball.
      return np.random.uniform(self.sampler['low'], self.sampler['high'], (n, self.dim))
    elif self.sampler['name'] == 'gaussian':
      return np.random.normal(self.sampler['mean'], self.sampler['std'], (n, self.dim))
    else:
      raise ValueError(f'Unknown sampler: {self.sampler["name"]}')

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

  def distance_function(self, p, q):
    return EuclideanGeometry.distance_function(self, p, q)

  def interpolate(self, p, q, alpha):
    return EuclideanGeometry.interpolate(self, p, q, alpha)
