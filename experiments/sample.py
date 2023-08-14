import torch 
import hydra
import wandb
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)
from rum.manifold import EuclideanManifold
from rum.density import OnlineKMeansEstimator
from rum.geometry import NeuralGeometry

class Logger:
  def __init__(self, cfg, manifold, geometry_estimator, density_estimator):
    self.scripts = cfg.scripts if 'scripts' in cfg else {}
    self.manifold = manifold
    self.geometry_estimator = geometry_estimator
    self.density_estimator = density_estimator

  def run_scripts(self, n_iter, samples):
    for name, freq in self.scripts.items():
      if freq % n_iter == 0: # Run script.
        if name == 'states':
          print('Saving states...')
          self.log({'states': samples}, use_wandb=False)
        elif name == 'entropy':
          print('Computing entropy...')
          self.log({'entropy': 666}) 

  def log(self, data, use_wandb=True):
    if use_wandb:
      wandb.log(data)
    else:
      print(data)

@hydra.main(config_path="config", config_name="sample")
def main(cfg):
  wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
  wandb.init(project='test', config=wandb_cfg)

  dim = 2
  manifold = EuclideanManifold(**cfg.manifold)
  #cfg_geom = cfg.geometry.copy() # Hack. TODO. 
  #cfg_geom = OmegaConf.to_container(cfg.geometry, resolve=True) # Hack. TODO.
  #geometry_estimator = NeuralGeometry(**cfg_geom)
  geometry_estimator = None
  density_estimator = OnlineKMeansEstimator(**cfg.density)
  logger = Logger(cfg, manifold, geometry_estimator, density_estimator)
  n_iter = 0
  while n_iter * cfg.samples_per_iter < cfg.max_samples:
    if cfg.sampling_method == 'random_walk':
      samples = manifold.random_walk(cfg.samples_per_iter)
    else:
      samples = manifold.sample(cfg.samples_per_iter)
    samples_tensor = torch.Tensor(samples)
    n_iter += 1

    if geometry_estimator is not None:
      geometry_estimator.learn(samples_tensor)
    if density_estimator is not None:
      density_estimator.learn(samples_tensor)

    logger.run_scripts(n_iter, samples)

if __name__ == '__main__':
  main()
