import numpy as np
from .manifold import Manifold

class TorusManifold(Manifold):
  def __init__(self, dim):
    assert dim == 2 # TODO.
    super(TorusManifold, self).__init__(dim, dim + 1)

    self.R = 2.0 / 3.0 # Radius of "inside" circle around 1-d hole. 
    self.r = 1.0 / 3.0 # Radius of "outside" circle around 2-d hole.

    self.diameter = (2 * self.r + (self.R - self.r)) * np.pi # Upper bound based on traversing two "outside" and one "inside" half-circles.

  def retraction(self, p, v):
    xi = self._to_local(p)
    v_1_scaling = self.r / np.sqrt(p[0] ** 2 + p[1] ** 2)
    xi[0] += v[0]
    xi[1] += v_1_scaling * v[1] 
    standardize = lambda x : (x + 2 * np.pi) % (2 * np.pi)
    xi = standardize(xi) 
    return self._from_local(xi)

  def starting_state(self):
    local = np.zeros(self.dim) 
    return self._from_local(local)

  def pdf(self, x):
    # Uniform.
    if True: # TODO. Check if on surface.
      return 1 / ((2 * np.pi * self.r)**2 * self.R) # TODO. This is only valid for dim=2.

  def sample(self, n):
    if n > 1:
      return np.array([self.sample(1) for _ in range(n)])
    # TODO. This is wrong.
    local = np.random.uniform(-np.pi, np.pi, 2)
    return self._from_local(local)

  def grid(self, n):
    assert self.dim == 2
    assert self._from_local is not None
    n_per_dim = int(np.power(n, 1.0 / self.dim))
    local_points = np.zeros([self.dim, n_per_dim])
    local_points[0] = np.linspace(-np.pi, np.pi, n_per_dim)
    local_points[1] = np.linspace(-np.pi, np.pi, n_per_dim)
    local_mesh = np.meshgrid(*local_points)
    local_mesh = np.reshape(local_mesh, [self.dim, -1]).T
    mesh = np.stack([self._from_local(local_point) for local_point in local_mesh])
    return mesh 

  def distance_function(self, x, y):
    # Terence Tao does not know how to do this for surfaces with non-constant curvature (I give up): https://mathoverflow.net/questions/37651/riemannian-surfaces-with-an-explicit-distance-function
    raise NotImplementedError

  def metric_tensor(self, p):
    return np.array([
      [(self.R + self.r * np.cos(p[1])) ** 2, 0],
      [0, self.r ** 2]
    ])

  def implicit_function(self, p):
    return np.sqrt(self.r**2 - (np.sqrt(p[0]**2 + p[1]**2) - self.R)**2)

  def _to_local(self, p):
    xi_0 = np.arctan2(p[2], np.sqrt(p[0]**2 + p[1]**2) - self.R)
    xi_1 = np.arctan2(p[1], p[0])
    return np.array([xi_0, xi_1])

  def _from_local(self, xi):
    p_0 = (self.R + self.r * np.cos(xi[0])) * np.cos(xi[1])
    p_1 = (self.R + self.r * np.cos(xi[0])) * np.sin(xi[1])
    p_2 = self.r * np.sin(xi[0])
    return np.array([p_0, p_1, p_2])
