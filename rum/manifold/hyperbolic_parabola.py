from .manifold import Manifold

class HyperbolicParabolaManifold(Manifold):
  def __init__(self, dim):
    assert dim == 2 # TODO.
    super(HyperbolicParabolaManifold, self).__init__(dim, dim + 1)
    self.low = -1.0 
    self.high = 1.0 

  def pdf(self, x):
    raise NotImplementedError

  def sample(self, n):
    if n > 1:
      return np.array([self.sample(1) for _ in range(n)])
    u, v = np.random.uniform(self.low, self.high, 2)
    return self._from_local([u, v])

  def starting_state(self):
    return np.zeros(self.ambient_dim)

  def distance_function(self, x, y):
    raise NotImplementedError

  def metric_tensor(self, x):
    raise NotImplementedError

  def impicit_function(self, c):
    return c[0]**2 - c[1]**2

  def _to_local(self, c):
    u = c[0]
    v = c[1]
    return np.array([u, v])

  def _from_local(self, c):
    x = c[0]
    y = c[1]
    z = c[0] * c[1]
    return np.array([x, y, z])
