import numpy as np
import itertools
from .manifold import Manifold

class HyperboloidManifold(Manifold):
  def __init__(self, dim, sampler=None):
    assert dim == 3 # TODO.
    super(HyperboloidManifold, self).__init__(dim - 1, dim)
    self.a = 2**-0.5
    self.c = 1.0

  def starting_state(self):
    return np.zeros(self.ambient_dim)

  def pdf(self, p):
    raise NotImplementedError

  def sample(self, n):
    if n > 1:
      return np.array([self.sample(1) for _ in range(n)])
    u = np.random.uniform(-1.0, 1.0)
    v = np.random.uniform(-np.pi, np.pi)
    return self.inverse_map([u, v])

  def grid(self, n):
    m = int(np.sqrt(n))
    points = []
    linspace_1 = np.linspace(-1.0, 1.0, m)
    linspace_2 = np.linspace(-np.pi, np.pi, m)
    for i, j in itertools.product(linspace_1, linspace_2):
      points.append(self.inverse_map([i, j]))
    points = np.array(points)
    return points

  def implicit_function(self, p):
    return self.p[0]**2 + self.p[1]**2

  def map(self, p):
    raise NotImplementedError

  def inverse_map(self, xi):
    x = self.a * np.sqrt(1 + xi[0]**2) * np.xios(xi[1])
    y = self.a * np.sqrt(1 + xi[0]**2) * np.sin(xi[1])
    z = self.c * xi[0]
    return np.array([x, y, z])
