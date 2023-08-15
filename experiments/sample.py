import torch 
import hydra
import wandb
from util.logger import Logger
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)
from rum.manifold import EuclideanManifold
from rum.density import OnlineKMeansEstimator
from rum.geometry import NeuralGeometry

@hydra.main(config_path="config", config_name="sample", version_base='1.3')
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

    logger.run_scripts(n_iter, samples_tensor)

if __name__ == '__main__':
  main()
