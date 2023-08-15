import torch 
import hydra
import wandb
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)
from util.logger import Logger
from util.make import make

@hydra.main(config_path="config", config_name="sample", version_base='1.3')
def main(cfg):
  wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
  wandb.init(project='test', config=wandb_cfg)

  manifold = make(cfg, 'manifold')
  geometry = make(cfg, 'geometry')
  if geometry is None: # Use natural geometry.
    geometry = manifold
  density = make(cfg, 'density')
  if density is None: 
    density = manifold
  logger = Logger(cfg, manifold, geometry, density)
  import pdb; pdb.set_trace()

  n_iter = 0
  while n_iter * cfg.samples_per_iter < cfg.max_samples:
    if cfg.sampling_method == 'random_walk':
      samples = manifold.random_walk(cfg.samples_per_iter)
    else:
      samples = manifold.sample(cfg.samples_per_iter)
    samples_tensor = torch.Tensor(samples)
    n_iter += 1

    geometry.learn(samples_tensor)
    density.learn(samples_tensor)

    logger.run_scripts(n_iter, samples_tensor)

if __name__ == '__main__':
  main()
