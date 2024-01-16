import numpy as np
import itertools
from .manifold import Manifold, Chart
from .util import sphere_sample_uniform

def chart_0(p):
  return np.array([
    p[0] / (1 + p[2]),
    p[1] / (1 + p[2])
  ])

def chart_1(p):
  return np.array([
    p[0] / (1 - p[2]),
    p[1] / (1 - p[2])
  ])

def inverse_chart_0(xi):
  a = xi[0]**2 + xi[1]**2
  return (1 / (a + 1)) * np.array([
    2 * xi[0],
    2 * xi[1],
    1 - a 
  ])

def inverse_chart_1(xi):
  a = xi[0]**2 + xi[1]**2
  return (1 / (a + 1)) * np.array([
    2 * xi[0],
    2 * xi[1],
    a - 1
  ])

def differential_chart_0(p, v):
  return (1 / (1 + p[2]) ** 1) * v

def differential_chart_1(p, v):
  #return (1 / (1 - p[2]) ** 1) * v
  return - (1 / (1 - p[2]) ** 1) * v # The minus sign is less systematic but more intuitive.

class SphereManifold(Manifold):
  # We assume unit radius.
  def __init__(self, dim, sampler):
    assert dim == 2 
    super(SphereManifold, self).__init__(dim, dim + 1)
    self.sampler = sampler

    self.charts = [
      Chart(
        chart_0,
        inverse_chart_0,
        differential_chart_0
      ),
      Chart(
        chart_1,
        inverse_chart_1,
        differential_chart_1
      )
    ]

    self.diameter = np.pi # Unit sphere. Note that diameter here means distance between furthest points.

  def get_chart_index(self, p):
    if p[2] >= 0:
      return 0
    else:
      return 1

  def starting_state(self):
    return np.array([0.0, 0.0, 1.0])

  def pdf(self, x):
    if self.sampler['type'] == 'uniform':
      if np.linalg.norm(x) == 1: 
        return 1 / (2 * np.pi)**(self.dim / 2) 
      else:
        return 0.0
    elif self.sampler['type'] == 'vonmises_fisher':
      return scipy.stats.vonmises_fisher.pdf(x, self.sampler['mu'], self.sampler['kappa'])

  def sample(self, n):
    if n > 1: # TODO.
      return np.array([self.sample(1) for _ in range(n)])
    if self.sampler['type'] == 'uniform':
      return sample_uniform(self.dim)
    elif self.sampler['type'] == 'vonmises_fisher':
      return scipy.stats.vonmises_fisher.rvs(self.sampler['mu'], self.sampler['kappa'], size=1)[0]

  def grid(self, n):
    assert self.dim == 2

    # Using global coordinates.
    m = int(np.power(n, 1.0 / 3.0))
    points = []
    linspace = np.linspace(-1, 1, m)
    for i, j, k in itertools.product(*([linspace] * 3)):
      points.append(np.array([i,j,k]))
      if np.linalg.norm(points[-1]) > 1:
        points[-1] /= np.linalg.norm(points[-1])
    points = np.array(points)
    return points

    # Using local coordinates.
    #assert self._from_local is not None
    #n_per_dim = int(np.power(n, 1.0 / self.dim))
    #local_points = np.zeros([self.dim, n_per_dim])
    #local_points[0] = np.linspace(-np.pi, np.pi, n_per_dim)
    #local_points[1] = np.linspace(-np.pi / 2, np.pi / 2, n_per_dim)
    #local_mesh = np.meshgrid(*local_points)
    #local_mesh = np.reshape(local_mesh, [self.dim, -1]).T
    #mesh = np.stack([self._from_local(local_point) for local_point in local_mesh])
    #return mesh 

  @staticmethod
  def distance_function(x, y):
    return np.arccos(np.dot(x, y))

  def metric_tensor(self, p):
    return np.array([
        [1.0, 0.0],
        [0.0, np.sin(p[0])**2]
      ])

  def implicit_function(self, p):
    return 1.0 - p[0]**2 - p[1]**2

  def _to_local(self, p): 
    return np.array([
      np.arctan2(np.sqrt(p[0]**2 + p[1]**2), p[2]),
      np.arctan2(p[1], p[0])
    ])

  def _from_local(self, xi): 
    return np.array([
      np.sin(xi[0]) * np.cos(xi[1]),
      np.sin(xi[0]) * np.sin(xi[1]),
      np.cos(xi[0])
    ])
