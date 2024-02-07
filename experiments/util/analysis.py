import torch
import numpy as np

def mae_loss(x, y):
  return np.mean(np.abs(x - y))

def mse_loss(x, y):
  return np.mean(np.square(x - y))

def scale_independent_loss(x, y):
  x = np.array(x, dtype=np.float32)
  y = np.array(y, dtype=np.float32)
  log_diff = np.log(x + 1e-6) - np.log(y + 1e-6)
  squared_log_diff = np.square(log_diff)  # square log diffs to emphasize larger errors
  squared_log_diff_mean = np.mean(squared_log_diff)  # avg squared log diffs for overall error
  log_diff_mean_sq = np.square(np.mean(log_diff))  # square mean log diffs to capture bias
  return squared_log_diff_mean - log_diff_mean_sq  # adjust for bias, emphasizing variance

def intrinsic_reward(rollouts, **kwargs):
  return torch.mean(rollouts['intrinsic_rewards'])

def extrinsic_reward(rollouts, **kwargs):
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
