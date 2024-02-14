import numpy as np
import itertools
from scipy.spatial import geometric_slerp
import torch
from .manifold import Manifold, Atlas, Chart
from .util import sphere_sample_uniform

class SphereAtlas(Atlas):
  @staticmethod
  def map_0(p):
    return np.array([
      p[0] / (1 + p[2]),
      p[1] / (1 + p[2])
    ])
  @staticmethod
  def map_1(p):
    return np.array([
      p[0] / (1 - p[2]),
      p[1] / (1 - p[2])
    ])

  @staticmethod
  def inverse_map_0(xi):
    a = xi[0]**2 + xi[1]**2
    return (1 / (a + 1)) * np.array([
      2 * xi[0],
      2 * xi[1],
      1 - a 
    ])

  @staticmethod
  def inverse_map_1(xi):
    a = xi[0]**2 + xi[1]**2
    return (1 / (a + 1)) * np.array([
      2 * xi[0],
      2 * xi[1],
      a - 1
    ])

  @staticmethod
  def differential_map_0(p):
    return (1 / (1 + p[2])) * np.array([
      [1, 0, -p[0] / (1 + p[2])],
      [0, 1, -p[1] / (1 + p[2])]
    ])

  @staticmethod
  def differential_map_1(p):
    return (1 / (1 - p[2])) * np.array([
      [1, 0, p[0] / (1 - p[2])],
      [0, 1, p[1] / (1 - p[2])]
    ])

  @staticmethod
  def differential_inverse_map_0(xi):
    a = xi[0]**2 + xi[1]**2
    m_1 = np.array([
      [1, 0],
      [0, 1],
      [0, 0]
    ])
    m_2 = np.array([
      [xi[0] ** 2, xi[0] * xi[1]],
      [xi[0] * xi[1], xi[1] ** 2],
      [xi[0], xi[1]]
    ])
    return (2 / (1 + a)) * m_1 - (2 / (1 + a)) ** 2 * m_2

  @staticmethod
  def differential_inverse_map_1(xi):
    a = xi[0]**2 + xi[1]**2
    m_1 = np.array([
      [1, 0],
      [0, 1],
      [0, 0]
    ])
    m_2 = np.array([
      [xi[0] ** 2, xi[0] * xi[1]],
      [xi[0] * xi[1], xi[1] ** 2],
      [-xi[0], -xi[1]]
    ])
    return (2 / (1 + a)) * m_1 - (2 / (1 + a)) ** 2 * m_2

  @staticmethod
  def norm_0(p, v):
    # TODO. Can optimize by not computing Euclidean norm.
    return (1 + p[2]) * np.linalg.norm(v)

  @staticmethod
  def norm_1(p, v):
    # TODO. Can optimize by not computing Euclidean norm.
    return (1 - p[2]) * np.linalg.norm(v)

  def __init__(self):
    super(SphereAtlas, self).__init__()
    self.charts = [
      Chart(
        SphereAtlas.map_0, 
        SphereAtlas.inverse_map_0, 
        SphereAtlas.norm_0,
        SphereAtlas.differential_map_0,
        SphereAtlas.differential_inverse_map_0
      ),
      Chart(
        SphereAtlas.map_1, 
        SphereAtlas.inverse_map_1, 
        SphereAtlas.norm_1,
        SphereAtlas.differential_map_1,
        SphereAtlas.differential_inverse_map_1
      ),
    ]

  def get_chart(self, p):
    if p[2] >= 0:
      return self.charts[0]
    else:
      return self.charts[1]

class SphereManifold(Manifold):
  # We assume unit radius.
  def __init__(self, dim, sampler):
    assert dim == 3
    super(SphereManifold, self).__init__(dim - 1, dim)
    self.sampler = sampler
    self.atlas = SphereAtlas()

  def starting_state(self):
    return sphere_sample_uniform(self.manifold_dim)[0]

  def pdf(self, p):
    if self.sampler['name'] == 'uniform':
      if np.linalg.norm(p) == 1: 
        return 1.0 / (2.0 * np.pi) ** (self.manifold_dim / 2.0) 
      else:
        return 0.0
    elif self.sampler['name'] == 'vonmises_fisher':
      return scipy.stats.vonmises_fisher.pdf(p, self.sampler['mu'], self.sampler['kappa'])
    else:
      raise ValueError(f'Unknown sampler: {self.sampler["name"]}')

  def sample(self, n):
    if self.sampler['name'] == 'uniform':
      return sphere_sample_uniform(self.manifold_dim, n)
    elif self.sampler['name'] == 'vonmises_fisher':
      return scipy.stats.vonmises_fisher.rvs(self.sampler['mu'], self.sampler['kappa'], size=n)
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
