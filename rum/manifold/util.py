import numpy as np

def sphere_sample_uniform(dim, n=1):
  x = np.random.normal(0, 1, (n, dim + 1))
  norm = np.linalg.norm(x, axis=1)
  x /= norm[:, None]
  return x
