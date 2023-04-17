# Adapted from https://github.com/rll-research/url_benchmark

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure, Logger
from envs import load
from util.dmc2gym import GymWrapper
from util.callbacks import CheckpointCallback
from kme.rewarder import Rewarder
import numpy as np
import torch
from torch import nn

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class RMS(object):
    """running mean and std """
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs +
                 torch.square(delta) * self.n * bs /
                 (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S


class RND(nn.Module):
    def __init__(self,
                 n_states,
                 hidden_dim,
                 rep_dim,
                 clip_val=5.):
        super().__init__()
        self.clip_val = clip_val

        self.normalize_state = nn.BatchNorm1d(n_states, affine=False)

        self.predictor = nn.Sequential(nn.Linear(n_states, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, rep_dim))
        self.target = nn.Sequential(nn.Linear(n_states, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, rep_dim))

        for param in self.target.parameters():
            param.requires_grad = False

        self.apply(weight_init)

    def forward(self, state):
        state = self.normalize_state(state)
        state = torch.clamp(state, -self.clip_val, self.clip_val)
        prediction, target = self.predictor(state), self.target(state)
        prediction_error = torch.square(target.detach() - prediction).mean(dim=-1, keepdim=True)
        return prediction_error


class RNDRewarder:
  def __init__(self, n_states, hidden_dim, rep_dim, reward_scaling, learning_rate, batch_size, n_envs, device):
    self.reward_scaling = reward_scaling
    self.device = device
    self.rnd = RND(n_states, hidden_dim, rep_dim).to(self.device)
    self.intrinsic_reward_rms = RMS(device=self.device)
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.n_states = n_states
    self.n_envs = n_envs
    self.rnd_opt = torch.optim.Adam(self.rnd.parameters(), lr=self.learning_rate)
    self.rnd.train()

  def compute_rewards_and_learn(self, rollout_buffer, device=False):
      # Learn
      states = torch.as_tensor(rollout_buffer.observations, device=self.device)
      shuffled_indices = torch.randperm(rollout_buffer.buffer_size)
      states = states[shuffled_indices]
      states = states.view([self.batch_size * self.n_envs, self.n_states])
      prediction_error = self.rnd(states)
      loss = prediction_error.mean()
      self.rnd_opt.zero_grad(set_to_none=True)
      loss.backward()
      self.rnd_opt.step()

      # Compute reward
      _, intr_reward_var = self.intrinsic_reward_rms(prediction_error)
      int_rews = self.reward_scaling * prediction_error / (torch.sqrt(intr_reward_var) + 1e-8)

      # Add reward to rollout buffer.
      int_rews = int_rews.squeeze().reshape([self.batch_size, self.n_envs]).cpu().detach().numpy()
      shuffled_indices = shuffled_indices.detach().cpu().numpy() 
      for idx, shuffled_idx in enumerate(shuffled_indices):
        rollout_buffer.rewards[idx] += int_rews[shuffled_idx]

      metrics = {'reward/mean_int_rew': int_rews.mean()}

      return metrics
