import os
import torch 
import hydra
import wandb
from omegaconf import OmegaConf
from util.logger import Logger
from util.make import make
from util.resolver import init_resolver
import stable_baselines3 as sb3

init_resolver()

@hydra.main(config_path='config', config_name='run', version_base='1.3')
def main(cfg):
  # Init config and logging.
  wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
  wandb.init(project='test', config=wandb_cfg, dir=os.getcwd(), id=cfg.name, name=cfg.name)

  # Make learning objects.
  manifold = make(cfg, 'manifold')
  if cfg.rewarder != {}:
    rewarder = make(cfg, 'rewarder') 
    density = rewarder.kmeans
  else:
    rewarder = None
    if 'density' in cfg and cfg.density is not None:
      density = make(cfg, 'density')
    else:
      density = manifold
  geometry = make(cfg, 'geometry')
  if geometry is None: # Use natural geometry.
    geometry = manifold

  # Make reinforcement learning objects. # TODO.
  env = manifold  # TODO. Support MuJoCo.
  agent = None
  if cfg.sampling_method == 'reinforcement_learning':
    assert agent is None or cfg.n_envs % cfg.samples_per_iter == 0
    env = sb3.common.vec_env.SubprocVecEnv([lambda: env for _ in range(cfg.n_envs)])
    agent = sb3.PPO('MlpPolicy', env, n_steps=cfg.samples_per_iter // cfg.n_envs, rewarder=rewarder) 

  # Make logger.
  logger = Logger(cfg, manifold, geometry, density, rewarder, agent, verbose=cfg.verbose)
  if agent is not None:
    sb3_logger = sb3.common.logger.configure('.', ['stdout'])
    sb3_logger = logger.wrap_sb3_logger(sb3_logger)
    agent.set_logger(sb3_logger) # Necessary to inject scripts.

  # Sanity checks.
  assert cfg.sampling_method in ['reinforcement_learning', 'random_walk', 'sample']

  # Run experiment main loop.
  if cfg.sampling_method == 'reinforcement_learning':
    agent.learn(total_timesteps=cfg.max_samples)
  else:
    n_iter = 0
    while n_iter * cfg.samples_per_iter < cfg.max_samples:
      if cfg.sampling_method == 'random_walk':
        samples = manifold.random_walk(cfg.samples_per_iter)
      elif cfg.sampling_method == 'sample':
        samples = manifold.sample(cfg.samples_per_iter)
      else:
        raise ValueError('Unknown sampling method {}.'.format(cfg.sampling_method))
      samples_tensor = torch.Tensor(samples)

      if rewarder is not None:
        rewarder.learn(samples_tensor)
      else:
        geometry.learn(samples_tensor)
        density.learn(samples_tensor)

      logger.run_scripts(rollouts={'states': samples_tensor})
      n_iter += 1

  print('Ran experiment {}. It can be found at\n{}'.format(cfg.name, os.getcwd()))

if __name__ == '__main__':
  main()
