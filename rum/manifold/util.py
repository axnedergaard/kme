import numpy as np

def sphere_sample_uniform(dim):
  x = np.random.normal(0, 1, dim + 1)
  return x / np.linalg.norm(x)
