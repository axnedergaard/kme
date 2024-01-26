import numpy as np
import itertools
from .manifold import Manifold, GlobalChartAtlas

class HyperbolicParabolaManifold(Manifold):
  def __init__(self, dim, sampler=None):
    assert dim == 2 
    super(HyperbolicParabolaManifold, self).__init__(dim, dim + 1)

    self.atlas = GlobalChartAtlas(
      self.map,
      self.inverse_map,
      self.norm,
      None, # TODO.
      None # TODO
    )

  def retraction(self, p, v):
    v = self.normalize(p, v)
    xi = self.map(p)
    xi = self.step_within_ball(xi, v)
    return self.inverse_map(xi)

  def metric_tensor(self, p):
    return np.array([
      [1.0 + 4.0 * p[0] ** 2, - 4.0 * p[0] * p[1]],
      [- 4.0 * p[0] * p[1], 1.0 + 4.0 * p[1] ** 2]
    ])

  def starting_state(self):
    return self.inverse_map([0.0, 0.0])

  def pdf(self, p):
    raise NotImplementedError

  def sample(self, n):
    if n > 1:
      return np.array([self.sample(1) for _ in range(n)])
    u, v = np.random.uniform(-1.0, 1.0, 2)
    return self.inverse_map([u, v])

  def grid(self, n):
    m = int(np.sqrt(n))
    points = []
    linspace = np.linspace(-1.0, 1.0, m)
    for i, j in itertools.product(linspace, repeat=2):
      if np.linalg.norm([i, j]) < 1.0:
        points.append(self.inverse_map([i, j]))
      #points.append(self.inverse_map([i, j]))
    points = np.array(points)
    return points

  def implicit_function(self, p):
    return p[0]**2 - p[1]**2

  def map(self, p):
    u = p[0]
    v = p[1]
    return np.array([u, v])

  def inverse_map(self, xi):
    x = xi[0]
    y = xi[1]
    z = xi[0] ** 2 - xi[1] ** 2
    return np.array([x, y, z])
