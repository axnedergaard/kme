import numpy as np
from scipy.stats import vonmises
from scipy.stats.sampling import SimpleRatioUniforms
from .manifold import Manifold, GlobalChartAtlas

def torus_uniform_outer_angle_pdf(x, R, r):
  return (R + r * (1.0 + np.cos(x))) / (2.0 * np.pi * (R + r))

class TorusUniformOuterAngleDist():
  # Required for efficient rejection sampling using scipy.stats.sampling.SimpleRatioUniforms.
  def __init__(self, R, r):
    self.R = R
    self.r = r

  def pdf(self, x):
    return torus_uniform_outer_angle_pdf(x, self.R, self.r)

class TorusManifold(Manifold):
  # TODO. Methods pdf and sample can be optimized by making samplers during initialization.
  def __init__(self, dim, sampler):
    assert dim == 3 # TODO.
    super(TorusManifold, self).__init__(dim - 1, dim)

    self.sampler = sampler

    self.R = 2.0 / 3.0 # Radius of "inside" circle around 1-d hole. 
    self.r = 1.0 / 3.0 # Radius of "outside" circle around 2-d hole.

    self.atlas = GlobalChartAtlas(
      self.map,
      self.inverse_map,
      self.norm,
      None, # TODO.
      None # TODO.
    )

  def retraction(self, p, v):
    v = self.normalize(p, v)
    xi = self.map(p)
    xi += v
    standardize = lambda x : (x + 2 * np.pi) % (2 * np.pi)
    xi = standardize(xi) 
    return self.inverse_map(xi)

  def starting_state(self):
    #local = np.zeros(self.manifold_dim) 
    local = [np.random.uniform(-np.pi, np.pi), 0]
    return self.inverse_map(local)

  def pdf(self, p):
    # Uniform.
    if self.sampler['name'] == 'uniform':
      point_is_on_manifold = True # TODO. Implement.
      if point_is_on_manifold:
        # Points on "inside" of circle around 1-d hole are less likely.
        xi = self.map(p)         
        pdf_0 = 1.0 / (2.0 * np.pi)
        pdf_1 = torus_uniform_outer_angle_pdf(xi[1], self.R, self.r)
        return pdf_0 * pdf_1
      else:
        return 0.0
    elif self.sampler['name'] == 'bivariate_vonmises':
      # We use the cosine variant with no correlation between the two angles.
      xi = self.map(p)
      pdf_0 = vonmises.pdf(loc=self.sampler['mu'][0], kappa=self.sampler['kappa'][0], x=xi[0])
      pdf_1 = vonmises.pdf(loc=self.sampler['mu'][1], kappa=self.sampler['kappa'][1], x=xi[1])
      return pdf_0 * pdf_1
    else:
      raise ValueError(f'Unknown sampler: {self.sampler["name"]}')

  def sample(self, n):
    if self.sampler['name'] == 'uniform':
      xi = np.zeros([n, self.manifold_dim])
      xi[:, 0] = np.random.uniform(-np.pi, np.pi, n)
      dist = TorusUniformOuterAngleDist(self.R, self.r) 
      uniform = SimpleRatioUniforms(dist, mode=0.0, domain=[-np.pi, np.pi])
      xi[:, 1] = uniform.rvs(n)
      return np.apply_along_axis(self.inverse_map, 1, xi)
    elif self.sampler['name'] == 'bivariate_vonmises':
      xi = np.zeros([n, self.manifold_dim])
      xi[:, 0] = vonmises(loc=self.sampler['mu'][0], kappa=self.sampler['kappa'][0]).rvs(n)
      xi[:, 1] = vonmises(loc=self.sampler['mu'][1], kappa=self.sampler['kappa'][1]).rvs(n)
      return np.apply_along_axis(self.inverse_map, 1, xi)
    else: 
      raise ValueError(f'Unknown sampler: {self.sampler["name"]}')

  def grid(self, n):
    assert self.manifold_dim == 2
    n_per_dim = int(np.power(n, 1.0 / self.manifold_dim))
    local_points = np.zeros([self.manifold_dim, n_per_dim])
    local_points[0] = np.linspace(-np.pi, np.pi, n_per_dim)
    local_points[1] = np.linspace(-np.pi, np.pi, n_per_dim)
    local_mesh = np.meshgrid(*local_points)
    local_mesh = np.reshape(local_mesh, [self.manifold_dim, -1]).T
    mesh = np.stack([self.inverse_map(local_point) for local_point in local_mesh])
    return mesh 

  def metric_tensor(self, p):
    xi = self.map(p)
    return np.array([
      [(self.R + self.r * np.cos(xi[1])) ** 2, 0.0],
      [0.0, self.r ** 2]
    ])

  def implicit_function(self, p):
    return np.sqrt(self.r ** 2 - (np.sqrt(p[0] ** 2 + p[1] ** 2) - self.R) ** 2)

  def map(self, p):
    xi_0 = np.arctan2(p[1], p[0])
    xi_1 = np.arctan2(p[2], np.sqrt(p[0] ** 2 + p[1] ** 2) - self.R)
    return np.array([xi_0, xi_1])

  def inverse_map(self, xi):
    p_0 = (self.R + self.r * np.cos(xi[1])) * np.cos(xi[0])
    p_1 = (self.R + self.r * np.cos(xi[1])) * np.sin(xi[0])
    p_2 = self.r * np.sin(xi[1])
    return np.array([p_0, p_1, p_2])
