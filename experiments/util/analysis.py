import torch
import numpy as np

def scale_independent_loss(x, y):
  return np.mean(np.abs(x - y)) # TODO. Implement properly.

def mean_intrinsic_reward(rollouts, **kwargs):
  return torch.mean(rollouts['intrinsic_rewards'])

def mean_extrinsic_reward(rollouts, **kwargs):
  return torch.mean(rollouts['extrinsic_rewards'])

def pathological_updates(density, **kwargs):
  return density.n_pathological

def entropy(density, **kwargs):
  return density.entropy().item()

def kmeans_loss(density, **kwargs):
  return density.kmeans_loss().item()

def kmeans_count_variance(density, **kwargs):
  cluster_sizes = density.cluster_sizes.item()
  return np.var(cluster_sizes)

def pdf_loss(manifold, density, n_points=1000, **kwargs):
  samples = manifold.sample(n_points)
  pdf_true = manifold.pdf(samples)
  pdf_est = density.pdf(samples)
  return scale_independent_loss(pdf_true, pdf_est)

def distance_loss(manifold, geometry, n_points=1000, **kwargs):
  x, y = manifold.sample(n_points), manifold.sample(n_points)
  xt, yt = torch.tensor(x), torch.tensor(y)
  distances_true = geometry.distance_function(xt, yt)
  distances_est = manifold.distance_function(x, y)
  return scale_independent_loss(distances_true, distances_est)

def state(samples, **kwargs):
  return samples

def test(success=True, **kwargs):
  if success:
    print('Test succeeded.')
  else:
    print('Test failed (but succeeeded).')
