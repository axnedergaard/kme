import numpy as np
from .manifold import Manifold

class HyperboloidManifold(Manifold):
  def __init__(self, dim):
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

  def _to_local(self, c):
    raise NotImplementedError

  def _from_local(self, c):
    x = self.a * np.sqrt(1 + c[0]**2) * np.cos(c[1])
    y = self.a * np.sqrt(1 + c[0]**2) * np.sin(c[1])
    z = self.c * c[0]
    return np.array([x, y, z])
