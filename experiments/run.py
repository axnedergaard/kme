import os
import torch 
import hydra
import wandb
from omegaconf import OmegaConf
from util.logger import Logger
from util.make import make, make_environment
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
  if cfg.geometry != {}:
    geometry = make(cfg, 'geometry')
  else: # Default is using the natural geometry (i.e. geometry of the manifold).
    geometry = manifold
  if cfg.density != {}:
    density = make(cfg, 'density', geometry=geometry)
  else:
    density = None # Default is not doing density estimation.
  if cfg.rewarder != {}:
    rewarder = make(cfg, 'rewarder', density=density) 
  else:
    rewarder = None # Default is not using intrinsic rewards. 
  if cfg.environment != {}:
    environment = make_environment(cfg)
  else:
    environment = manifold # Default is using the manifold as the environment.
  if cfg.sampling_method == 'reinforcement_learning':
    # assert agent is None or cfg.n_envs % cfg.samples_per_iter == 0
    vec_env = sb3.common.vec_env.SubprocVecEnv([lambda: environment for _ in range(cfg.n_envs)])
    agent = sb3.PPO('MlpPolicy', vec_env, n_steps=cfg.samples_per_iter // cfg.n_envs, rewarder=rewarder) 
  else:
    agent = None

  # Make logger.
  logger = Logger(cfg, manifold, geometry, density, rewarder, environment, agent, verbose=cfg.verbose)
  if agent is not None:
    sb3_logger = sb3.common.logger.configure('.', [])
    sb3_logger = logger.wrap_sb3_logger(sb3_logger)
    agent.set_logger(sb3_logger) # Necessary to inject scripts.

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
        if geometry is not None:
          geometry.learn(samples_tensor)
        if density is not None:
          density.learn(samples_tensor)

      logger.run_scripts(rollouts={'states': samples_tensor})
      n_iter += 1

  print('Ran experiment {}. It can be found at\n{}'.format(cfg.name, os.getcwd()))

if __name__ == '__main__':
  main()
