import numpy as np
import itertools
from .manifold import Manifold

class HyperboloidManifold(Manifold):
  def __init__(self, dim, sampler=None):
    assert dim == 2 # TODO.
    super(HyperboloidManifold, self).__init__(dim, dim + 1)
    self.a = 2**-0.5
    self.c = 1.0

  def pdf(self, x):
    raise NotImplementedError

  def sample(self, n):
    if n > 1:
      return np.array([self.sample(1) for _ in range(n)])
    u = np.random.uniform(-1.0, 1.0)
    v = np.random.uniform(-np.pi, np.pi)
    return self._from_local([u, v])

  def starting_state(self):
    return np.zeros(self.ambient_dim)

  def distance_function(self, x, y):
    raise NotImplementedError

  def metric_tensor(self, x):
    raise NotImplementedError

  def implicit_function(self):
    return self.c[0]**2 + self.c[1]**2

  def grid(self, n):
    m = int(np.sqrt(n))
    points = []
    linspace_1 = np.linspace(-1.0, 1.0, m)
    linspace_2 = np.linspace(-np.pi, np.pi, m)
    for i, j in itertools.product(linspace_1, linspace_2):
      points.append(self._from_local([i, j]))
    points = np.array(points)
    return points

  def _to_local(self, c):
    raise NotImplementedError

  def _from_local(self, c):
    x = self.a * np.sqrt(1 + c[0]**2) * np.cos(c[1])
    y = self.a * np.sqrt(1 + c[0]**2) * np.sin(c[1])
    z = self.c * c[0]
    return np.array([x, y, z])
