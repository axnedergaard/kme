from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure, Logger
from envs import load
from util.dmc2gym import GymWrapper
from util.callbacks import CheckpointCallback
from kme import Rewarder
import numpy as np

class KMERewarder:
  def __init__(self, n_envs, n_actions, n_states, k, learning_rate, balancing_strength, fn_type, power_fn_exponent, reward_scaling, coupled):
    self.n_envs = n_envs
    self.coupled = coupled
    self.reward_scaling = reward_scaling
    self.warmup = True
    self._rewarder = Rewarder(n_actions, n_states, k, learning_rate, balancing_strength, fn_type, power_fn_exponent)
    self._rewarder.reset()

  def _compute_reward(self, state, learn):
    reward, pathological_updates = self._rewarder.infer(state, learn)
    return reward * self.reward_scaling, {'pathological_updates': pathological_updates}

  def compute_rewards_and_learn(self, rollout_buffer, device=None):
    buffer_size = rollout_buffer.buffer_size
    int_rews = np.zeros([buffer_size, self.n_envs], dtype=np.float32)
    pat_upds = np.zeros([buffer_size, self.n_envs], dtype=np.int32)
    shuffled_indices = list(range(buffer_size))
    np.random.shuffle(shuffled_indices)
    for shuffled_idx in shuffled_indices:
      if not self.coupled and self.warmup: # Skip computing rewards on first update.
        self.warmup = False
      else:
        for env in range(self.n_envs):
          state = rollout_buffer.observations[shuffled_idx, env]
          int_rew, info = self._compute_reward(state, learn=self.coupled)
          int_rews[shuffled_idx, env] = int_rew
          pat_upds[shuffled_idx, env] = info['pathological_updates']
    if not self.coupled: # Pass over buffer again, this time learning instead of computing rewards.
      for shuffled_idx in shuffled_indices:
        for env in range(self.n_envs):
          state = rollout_buffer.observations[shuffled_idx, env]
          self._compute_reward(state, learn=True)
    rollout_buffer.rewards += int_rews

    metrics = {'reward/mean_int_rew': int_rews.mean(),
               'reward/mean_pat_upd': pat_upds.mean()}

    return metrics
