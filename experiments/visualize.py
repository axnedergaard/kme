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
from util import analysis

init_resolver()

def kmeans_centers(density, **kwargs):
  return density.centroids

def samples(samples, **kwargs):
  return samples

def grid(manifold, n=1000, **kwargs):
  return manifold.grid(n)

def upper_grid(manifold, n=1000, **kwargs):
  points = manifold.grid(n)
  points = np.array([p for p in points if p[2] > 0])
  return points
  
def lower_grid(manifold, n=1000, **kwargs):
  points = manifold.grid(n)
  points = np.array([p for p in points if p[2] <= 0])
  return points

def surround(manifold, policy, samples, n=25, radius=0.1, **kwargs):
  state = np.copy(manifold.state) 
  angles = np.linspace(0, 2*np.pi, n)
  directions = np.array([radius * np.array([np.cos(a), np.sin(a)]) for a in angles])
  surrounding_states = []
  for direction in directions:
    manifold.step(direction)
    surrounding_states.append(np.copy(manifold.state))
    manifold.state = state 
  return np.array(surrounding_states)

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
  elif color == 'purple':
    return [1.0, 0.0, 1.0]
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

class RandomPolicy:
  def __init__(self, action_space, max_repeats=1):
    self.action_space = action_space
    self.num_repeats = 0 
    self.max_repeats = max_repeats
    self.action = action_space.sample()

  def __call__(self, state):
    if self.num_repeats >= self.max_repeats:
      self.action = self.action_space.sample()
      self.num_repeats = 0 
    else:
      self.num_repeats += 1 
    return self.action

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
  if 'policy' in cfg and cfg.policy == 'xtouch':
    policy = XTouchPolicy(manifold.action_space)
  else:
    policy = RandomPolicy(manifold.action_space)

  state, _ = manifold.reset()

  if cfg.time_per_iter == 'xtouch':
    xtouch_interface = get_xtouch_interface([['delay', 0.01, 10, 0.01]])
    get_time_per_iter = lambda: xtouch_interface.get_values()['delay']
  elif type(cfg.time_per_iter) == float:
    get_time_per_iter = lambda: cfg.time_per_iter
  else:
    raise ValueError('Invalid time_per_iter {}'.format(cfg.time_per_iter))
  
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
      #print(samples_tensor)

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
      time.sleep(max(0, get_time_per_iter() - time_elapsed))

if __name__ == '__main__':
  main()
