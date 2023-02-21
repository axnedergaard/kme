# Adapted from https://github.com/younggyoseo/RE3

import numpy as np
import torch
from torch import nn

def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta + batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class TorchRunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, axis=0)
            batch_var = torch.var(x, axis=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    @property
    def std(self):
        return torch.sqrt(self.var)

class RE3Rewarder:
  def __init__(self, n_states, k, hidden_dim, rep_dim, reward_scaling, batch_size, n_envs, device, version=1, granularity=5000):
    self.reward_scaling = reward_scaling
    self.n_states = n_states
    self.hidden_dim = hidden_dim
    self.rep_dim = rep_dim
    self.batch_size = batch_size
    self.n_envs = n_envs
    self.k = k
    self.version = version
    self.device = device
    self.granularity = granularity

    self.s_ent_stats = TorchRunningMeanStd(shape=[1], device=self.device)

    self.random_encoder = nn.Sequential(nn.Linear(n_states, hidden_dim), nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, rep_dim))
    self.random_encoder.to(self.device)
    self.random_encoder.train()

  def _compute_state_entropy(self, features):
    if self.version == 1:
      distances = torch.matmul(features, features.transpose(0, 1))
      knn_distances = torch.kthvalue(distances, self.k + 1, dim=1).values
      return knn_distances
    else:
      with torch.no_grad():
        dists = []
        for idx in range(len(features) // self.granularity + 1):
          start = idx * self.granularity
          end = (idx + 1) * self.granularity
          dist = torch.norm(
            features[:, None, :] - features[None, start:end, :], dim=-1, p=2
          )
          dists.append(dist)

        dists = torch.cat(dists, dim=1)
        knn_dists = 0.0
        average_entropy=True
        if average_entropy:
          for k in range(5):
              knn_dists += torch.kthvalue(dists, k + 1, dim=1).values
          knn_dists /= 5
        else:
          knn_dists = torch.kthvalue(dists, k=self.k + 1, dim=1).values
        state_entropy = knn_dists
      return state_entropy.unsqueeze(1)

  def compute_rewards_and_learn(self, rollout_buffer, device=False):
    # Learn
    states = torch.as_tensor(rollout_buffer.observations, device=self.device)
    shuffled_indices = torch.randperm(rollout_buffer.buffer_size)
    states = states[shuffled_indices]
    states = states.view([self.batch_size * self.n_envs, self.n_states])
    features = self.random_encoder(states)
    s_ent = self._compute_state_entropy(features)
    self.s_ent_stats.update(s_ent)
    norm_state_entropy = s_ent / self.s_ent_stats.std
    s_ent = norm_state_entropy
    s_ent = torch.log(s_ent + 1.0)
    int_rews = self.reward_scaling * s_ent

    # Add rewards to rollout buffer.
    int_rews = int_rews.squeeze().reshape([self.batch_size, self.n_envs]).cpu().detach().numpy()
    shuffled_indices = shuffled_indices.detach().cpu().numpy()
    for idx, shuffled_idx in enumerate(shuffled_indices):
      rollout_buffer.rewards[idx] += int_rews[shuffled_idx]

    metrics = {'reward/mean_int_rew': int_rews.mean()}

    return metrics


