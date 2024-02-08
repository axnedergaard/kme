import os
import types
import wandb
import omegaconf
import torch
import h5py
from . import analysis
from .data import * 

LOCAL_LOG_SCRIPTS = [
  'state'
]

class Logger:
  def __init__(self, cfg, manifold, geometry, density, rewarder, environment, agent, verbose=0):
    # Convert script specification to proper format.
    self.script = cfg.script if 'script' in cfg else {}
    for name, spec in self.script.items():
      if type(spec) is int:
        spec = {
          'freq': spec,
          'local': name in LOCAL_LOG_SCRIPTS,
          'kwargs': {},
          'runs': 0,
        }
      elif type(spec) is omegaconf.dictconfig.DictConfig:
        spec = omegaconf.OmegaConf.to_container(spec, resolve=True)
        if 'freq' not in spec:
          raise Exception('Script frequency not specified.')
        if 'local' not in spec:
          spec['local'] = name in LOCAL_LOG_SCRIPTS
        spec['kwargs'] = {key: value for key, value in spec.items() if key not in ['freq', 'local', 'runs']}
        spec['runs'] = 0
      else:
        raise Exception('Script specification must be int or omegaconf.dictconfig.DictConfig.')
      if spec['local']: # If local logging used, make sure data directory exists.
        self._make_data_dir(name)
      self.script[name] = spec

    self.manifold = manifold
    self.geometry = geometry
    self.density = density
    self.rewarder = rewarder
    self.agent = agent
    self.verbose = verbose
    self.runs = 0

  def run_scripts(self, rollouts):
    self.runs += 1
    for name, spec in self.script.items():
      if self.runs % spec['freq'] == 0: # Run script.
        self.script[name]['runs'] += 1
        script = getattr(analysis, name)
        data = script(
            manifold = self.manifold,
            geometry = self.geometry, 
            density = self.density, 
            rewarder = self.rewarder,
            agent = self.agent,
            rollouts = rollouts, 
            **spec['kwargs'])
        # TODO. We currently do not do any data conversion, which could be inefficient.
        self.log({name: data}, use_wandb=not spec['local'], runs=spec['runs'])

  def _get_data_path(self, name):
    return os.path.join(os.getcwd(), 'data', name)

  def _make_data_dir(self, name):
    path = self._get_data_path(name)
    os.makedirs(path, exist_ok=True)

  def _get_path(self, name, runs):
    return os.path.join(self._get_data_path(name), '{}-{}.h5'.format(name, runs))

  def log(self, data, use_wandb=True, runs=None):
    if self.verbose > 0:
      print(data)
    if use_wandb:
      wandb.log(data)
    else:
      for name, value in data.items():
        path = self._get_path(name, runs)
        save_data(value, path)
        #torch.save(data, self._get_path(name, runs))

  def wrap_sb3_logger(self, sb3_logger):
    # Wrap SB3 logger to also log using our logger.
    sb3_logger._record = sb3_logger.record

    def record(obj, key, value, exclude=None):
      self.log({key: value})
      obj._record(key, value, exclude)

    sb3_logger.record = types.MethodType(record, sb3_logger)
    sb3_logger.run_scripts = lambda rollouts : self.run_scripts(rollouts)
    return sb3_logger
