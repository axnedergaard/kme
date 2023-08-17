import os
import torch 
import hydra
import wandb
from omegaconf import OmegaConf
from util.logger import Logger
from util.make import make
from util.resolver import init_resolver

init_resolver()

@hydra.main(config_path='config', config_name='run', version_base='1.3')
def main(cfg):
  # Init config and logging.
  wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
  wandb.init(project='test', config=wandb_cfg, dir=os.getcwd(), id=cfg.name, name=cfg.name)

  # Make non-reinforcement learning objects.
  manifold = make(cfg, 'manifold')
  geometry = make(cfg, 'geometry')
  if geometry is None: # Use natural geometry.
    geometry = manifold
  density = make(cfg, 'density')
  if density is None: 
    density = manifold

  # Make reinforcement learning objects. # TODO.
  env = None
  rewarder = None
  agent = None
  if cfg.sampling_method == 'reinforcement_learning':
    assert agent is None or cfg.n_envs % cfg.samples_per_iter == 0
    env = None
    rewarder = None
    agent = sb3.PPO('MlpPolicy', env, n_steps=cfg.samples_per_iter // cfg.n_envs) 

  # Make logger.
  logger = Logger(cfg, manifold, geometry, density, rewarder, agent, verbose=cfg.verbose)

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
      else: # cfg.sampling_method == 'sample'
        samples = manifold.sample(cfg.samples_per_iter)
      samples_tensor = torch.Tensor(samples)

      geometry.learn(samples_tensor)
      density.learn(samples_tensor)

      logger.run_scripts(samples_tensor)
      n_iter += 1

  print('Ran experiment {}. It can be found at\n{}'.format(cfg.name, os.getcwd()))

if __name__ == '__main__':
  main()
