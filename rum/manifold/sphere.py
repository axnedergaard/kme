import numpy as np
import itertools
from scipy.spatial import geometric_slerp
from scipy.stats import vonmises_fisher
import torch
from .manifold import Manifold, Atlas, Chart
from .util import sphere_sample_uniform

class SphereAtlas(Atlas):
  # Atlas for n-sphere using stereographic projections.
  def map_0(self, p):
    return np.array([p[i] / (1.0 + p[self.n]) for i in range(self.n)])

  def map_1(self, p):
    return np.array([p[i] / (1.0 - p[self.n]) for i in range(self.n)])

  def inverse_map_0(self, xi):
    a = np.sum(xi ** 2)
    return (1.0 / (a + 1.0)) * np.concatenate((2 * xi, [1.0 - a]))

  def inverse_map_1(self, xi):
    a = np.sum(xi ** 2)
    return (1.0 / (a + 1.0)) * np.concatenate((2 * xi, [a - 1.0]))

  def norm_0(self, p, v):
    # TODO. Can optimize by not computing Euclidean norm.
    return (1.0 + p[self.n]) * np.linalg.norm(v)

  def norm_1(self, p, v):
    # TODO. Can optimize by not computing Euclidean norm.
    return (1.0 - p[self.n]) * np.linalg.norm(v)

  def differential_map_0(self, p):
    raise NotImplementedError
    #return (1 / (1 + p[2])) * np.array([
    #  [1, 0, -p[0] / (1 + p[2])],
    #  [0, 1, -p[1] / (1 + p[2])]
    #])

  def differential_map_1(self, p):
    raise NotImplementedError
    #return (1 / (1 - p[2])) * np.array([
    #  [1, 0, p[0] / (1 - p[2])],
    #  [0, 1, p[1] / (1 - p[2])]
    #])

  def differential_inverse_map_0(self, xi):
    raise NotImplementedError
    #a = xi[0]**2 + xi[1]**2
    #m_1 = np.array([
    #  [1, 0],
    #  [0, 1],
    #  [0, 0]
    #])
    #m_2 = np.array([
    #  [xi[0] ** 2, xi[0] * xi[1]],
    #  [xi[0] * xi[1], xi[1] ** 2],
    #  [xi[0], xi[1]]
    #])
    #return (2 / (1 + a)) * m_1 - (2 / (1 + a)) ** 2 * m_2

  def differential_inverse_map_1(self, xi):
    raise NotImplementedError
    #a = xi[0]**2 + xi[1]**2
    #m_1 = np.array([
    #  [1, 0],
    #  [0, 1],
    #  [0, 0]
    #])
    #m_2 = np.array([
    #  [xi[0] ** 2, xi[0] * xi[1]],
    #  [xi[0] * xi[1], xi[1] ** 2],
    #  [-xi[0], -xi[1]]
    #])
    #return (2 / (1 + a)) * m_1 - (2 / (1 + a)) ** 2 * m_2

  def __init__(self, n):
    super(SphereAtlas, self).__init__()
    self.n = n # Dimension of sphere (ambient dimension is n + 1).
    self.charts = [
      Chart(
        self.map_0, 
        self.inverse_map_0, 
        self.norm_0,
        self.differential_map_0,
        self.differential_inverse_map_0
      ),
      Chart(
        self.map_1, 
        self.inverse_map_1, 
        self.norm_1,
        self.differential_map_1,
        self.differential_inverse_map_1
      ),
    ]

  def get_chart(self, p):
    if p[self.n] >= 0:
      return self.charts[0]
    else:
      return self.charts[1]

class SphereManifold(Manifold):
  # We assume unit radius.
  def __init__(self, dim, sampler):
    super(SphereManifold, self).__init__(dim - 1, dim)
    self.sampler = sampler
    self.atlas = SphereAtlas(dim - 1)

  def starting_state(self):
    return sphere_sample_uniform(self.manifold_dim)[0]

  def pdf(self, p):
    if self.sampler['name'] == 'uniform':
      if np.linalg.norm(p) == 1: 
        return 1.0 / (2.0 * np.pi) ** (self.manifold_dim / 2.0) 
      else:
        return 0.0
    elif self.sampler['name'] == 'vonmises_fisher':
      return vonmises_fisher.pdf(p, self.sampler['mu'], self.sampler['kappa'])
    else:
      raise ValueError(f'Unknown sampler: {self.sampler["name"]}')

  def sample(self, n):
    if self.sampler['name'] == 'uniform':
      return sphere_sample_uniform(self.manifold_dim, n)
    elif self.sampler['name'] == 'vonmises_fisher':
      return vonmises_fisher.rvs(self.sampler['mu'], self.sampler['kappa'], size=n)
    else:
      raise ValueError(f'Unknown sampler: {self.sampler["name"]}')

  def grid(self, n):
    m = int(np.power(n, 1.0 / 3.0))
    points = []
    linspace = np.linspace(-1, 1, m)
    for i, j, k in itertools.product(*([linspace] * 3)):
      points.append(np.array([i,j,k]))
      if np.linalg.norm(points[-1]) > 1:
        points[-1] /= np.linalg.norm(points[-1])
    points = np.array(points)
    return points

  def implicit_function(self, p):
    return 1.0 - p[0] ** 2 - p[1] ** 2

  def distance_function(self, p, q):
    return torch.acos(torch.inner(p, q).squeeze(0))

  def interpolate(self, p, q, alpha):
    # Computer graphics people already solved this to interpolate rotatations.
    # https://en.wikipedia.org/wiki/Slerp#Geometric_Slerp
    return torch.tensor(geometric_slerp(p, q, alpha), dtype=torch.float32)
