import time
import sys
import numpy as np
import hydra
import torch
from omegaconf import DictConfig, ListConfig
from util.visualizer import Visualizer
from util.make import make
from util.resolver import init_resolver
from util.xtouch_interface import get_xtouch_interface 

init_resolver()

def kmeans_centers(density, **kwargs):
  return density.centroids

def samples(samples, **kwargs):
  return samples

def grid(manifold, n=1000, **kwargs):
  return manifold.grid(n)

def _get_color(color):
  if type(color) == ListConfig and len(color) == 3 and all([type(c) == int for c in color]):
    return [float(c) for c in color] 
  elif color == 'black':
    return [0.0, 0.0, 0.0]    
  elif color == 'white':
    return [1.0, 1.0, 1.0]
  elif color == 'red':
    return [1.0, 0.0, 0.0]
  elif color == 'green':
    return [0.0, 1.0, 0.0]
  elif color == 'blue':
    return [0.0, 0.0, 1.0]
  else:
    raise ValueError('Invalid color {}'.format(color))

class XTouchPolicy():
  def __init__(self, action_space):
    self.action_dim = action_space.shape[0]
    parameters = [['a_{}'.format(i), action_space.low[i], action_space.high[i]] for i in range(self.action_dim)]
    self.interface = get_xtouch_interface(parameters)

  def __call__(self, state):
    values = self.interface.get_values()
    return np.array([values['a_{}'.format(i)] for i in range(self.action_dim)])

@hydra.main(config_path='config', config_name='visualize', version_base='1.3')
def main(cfg):
  # Prepare scripts.
  scripts = {}
  for name, spec in cfg.script.items():
    if type(spec) == str:
      scripts[name] = {
        'color': _get_color(spec),
        'freq': 1,
        'persist': False,
        'kwargs': {},
        'runs': 0
      }
    elif type(spec) == DictConfig:
      scripts[name] = {
        'color': _get_color(spec.color),
        'freq': spec.freq if 'freq' in spec else 1,
        'persist': spec.persist if 'persist' in spec else False,
        'kwargs': {key: value for key, value in spec.items() if key not in ['color', 'freq', 'persist', 'runs']},
        'runs': 0
      }

  # Make objects.
  manifold = make(cfg, 'manifold')
  geometry = make(cfg, 'geometry')
  density = make(cfg, 'density')
  rewarder = None
  visualizer = Visualizer(
    interface=cfg.interface, 
    manifold=manifold, 
    distance_function=manifold.distance_function, 
    cursor_target=cfg.cursor_target if 'cursor_target' in cfg else None,
    cursor_color=_get_color(cfg.cursor_color) if 'cursor_color' in cfg else None,
  )
  if cfg.policy == 'xtouch':
    policy = XTouchPolicy(manifold.action_space)
  else:
    policy = None

  state, _ = manifold.reset()
  while True:
      time_start = time.time()

      # Sample states.
      if cfg.sampling_method == 'reinforcement_learning':
        samples = np.zeros((cfg.samples_per_iter, manifold.observation_space.shape[0]))
        samples[0] = state
        for i in range(cfg.samples_per_iter):
          samples[i] = state
          action = policy(state)
          state, _, _, _, _ = manifold.step(action)
      elif cfg.sampling_method == 'random_walk':
        samples = manifold.random_walk(cfg.samples_per_iter)
      else: # cfg.sampling_method == 'sample'
        samples = manifold.sample(cfg.samples_per_iter)
      samples_tensor = torch.Tensor(samples)

      # Learn.
      if density is not None:
        density.learn(samples_tensor)
      # TODO. Learn policy if required.

      # Add points to render according to scripts. 
      for name, spec in scripts.items():
        spec['runs'] += 1
        if spec['runs'] % spec['freq'] == 0:
          script = getattr(sys.modules[__name__], name) 
          points = script(  
            manifold = manifold,
            geometry = geometry, 
            density = density, 
            rewarder = rewarder,
            policy = policy,
            samples = samples, 
            **spec['kwargs']
          )
          if not spec['persist']:
            visualizer.remove(name)
          data = {'name': name, 'points': points, 'color': spec['color']}
          visualizer.add(data)

      # Render points.
      visualizer.render()

      time_elapsed = time.time() - time_start
      time.sleep(max(0, cfg.time_per_iter - time_elapsed))

if __name__ == '__main__':
  main()
