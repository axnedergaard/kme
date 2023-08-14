from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure, Logger
from envs import load
from util.dmc2gym import GymWrapper
from util.callbacks import CheckpointCallback
from pathlib import Path
import hydra
import wandb
import omegaconf
import types
from agents.kme import KMERewarder
from agents.rnd import RNDRewarder
from agents.re3 import RE3Rewarder
import torch
import numpy as np
import random

def wrap_logger(logger, wandb):
  logger._record = logger.record

  def record(self, key, value, exclude=None):
    wandb.log({key: value})
    self._record(key, value)

  logger.record = types.MethodType(record, logger)
  return logger

def get_space_dims(_make_env):
  env = _make_env()
  n_states = env.observation_space.shape[0]
  n_actions = env.action_space.shape[0]
  return n_states, n_actions

def make_env(domain, task):
  def _init():
    env = load(domain, task)
    env = GymWrapper(env)
    return env
  return _init

def set_all_seeds(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

@hydra.main(config_path='.', config_name='train')
def main(cfg):
  print('Starting experiment at {}'.format(Path.cwd()))

  set_all_seeds(cfg.seed)

  wandb.config = omegaconf.OmegaConf.to_container(
      cfg, resolve=True, throw_on_missing=True
  )
  group_name = '_'.join([cfg.exp, cfg.domain, cfg.task, cfg.agent])
  exp_name = group_name + '_' + str(cfg.seed)
  wandb.init(project='sm', name=exp_name, group=group_name)

  env = SubprocVecEnv([make_env(cfg.domain, cfg.task) for i in range(cfg.n_envs)])
  checkpoints = list(dict.fromkeys(cfg.checkpoints + [cfg.n_timesteps])) # Last state is always checkpoint.
  callback = CheckpointCallback(checkpoints, 'models', name_prefix='{}_{}_{}'.format(cfg.domain, cfg.task, cfg.agent))

  n_states, n_actions = get_space_dims(make_env(cfg.domain, cfg.task))
  n_steps = cfg.rollout_size // cfg.n_envs
  if cfg.rollout_size % cfg.n_envs != 0:
    print('Warning: rollout_size should be divisible by n_env')

  if cfg.agent == 'kme':
    rewarder = KMERewarder(cfg.n_envs, n_actions, n_states, cfg.k, cfg.learning_rate, cfg.balancing_strength, cfg.fn_type, cfg.power_fn_exponent, cfg.reward_scaling, cfg.coupled)
  elif cfg.agent == 'rnd':
    rewarder = RNDRewarder(n_states, cfg.hidden_dim, cfg.rep_dim, cfg.reward_scaling, cfg.learning_rate, n_steps, cfg.n_envs, cfg.device)
  elif cfg.agent == 're3':
    rewarder = RE3Rewarder(n_states, cfg.k, cfg.hidden_dim, cfg.rep_dim, cfg.reward_scaling, n_steps, cfg.n_envs, cfg.device, version=cfg.re3_version, granularity=cfg.re3_granularity)
  else:
    rewarder = None
  model = PPO('MlpPolicy', env, rewarder=rewarder, n_steps=n_steps, device=cfg.device, seed=cfg.seed, verbose=cfg.verbose)

  logger = configure('.', ['stdout', 'csv'])
  logger = wrap_logger(logger, wandb)
  model.set_logger(logger)

  model.learn(total_timesteps=cfg.n_timesteps, callback=callback)

if __name__ == '__main__':
  main()
