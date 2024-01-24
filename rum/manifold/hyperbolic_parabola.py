import numpy as np
import itertools
from .manifold import Manifold

class HyperbolicParabolaManifold(Manifold):
  def __init__(self, dim, sampler=None):
    assert dim == 2 
    super(HyperbolicParabolaManifold, self).__init__(dim, dim + 1)
    self.low = -1.0 
    self.high = 1.0 

  def retraction(self, p, v):
    v = self.normalize(p, v)
    xi = self._to_local(p)
    xi = self.step_within_ball(xi, v)
    return self._from_local(xi)

  def starting_state(self):
    return self._from_local([0.0, 0.0])

  def pdf(self, x):
    raise NotImplementedError

  def sample(self, n):
    if n > 1:
      return np.array([self.sample(1) for _ in range(n)])
    u, v = np.random.uniform(self.low, self.high, 2)
    return self._from_local([u, v])

  def metric_tensor(self, p):
    return np.array([
      [1.0 + 4.0 * p[0] ** 2, - 4.0 * p[0] * p[1]],
      [- 4.0 * p[0] * p[1], 1.0 + 4.0 * p[1] ** 2]
    ])

  def impicit_function(self, p):
    return p[0]**2 - p[1]**2

  def grid(self, n):
    m = int(np.sqrt(n))
    points = []
    linspace = np.linspace(self.low, self.high, m)
    for i, j in itertools.product(linspace, repeat=2):
      if np.linalg.norm([i, j]) < 1.0:
        points.append(self._from_local([i, j]))
      #points.append(self._from_local([i, j]))
    points = np.array(points)
    return points

  def _to_local(self, p):
    u = p[0]
    v = p[1]
    return np.array([u, v])

  def _from_local(self, xi):
    x = xi[0]
    y = xi[1]
    z = xi[0] ** 2 - xi[1] ** 2
    return np.array([x, y, z])
