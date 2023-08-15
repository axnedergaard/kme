import os
import wandb
import omegaconf
import torch
import h5py
from . import analysis

LOCAL_LOG_SCRIPTS = [
  'state'
]

class Logger:
  def __init__(self, cfg, manifold, geometry, density, verbose=0):
    # Convert script specification to proper format.
    self.script = cfg.script if 'script' in cfg else {}
    for name, spec in self.script.items():
      if type(spec) is int:
        spec = {
          'freq': spec,
          'local': name in LOCAL_LOG_SCRIPTS,
          'params': {},
          'runs': 0,
        }
      elif type(spec) is omegaconf.dictconfig.DictConfig:
        spec = omegaconf.OmegaConf.to_container(spec, resolve=True)
        if 'freq' not in spec:
          raise Exception('Script frequency not specified.')
        if 'local' not in spec:
          spec['local'] = name in LOCAL_LOG_SCRIPTS
        spec['params'] = spec['params'] if 'params' in spec else {}
        spec['runs'] = 0
      else:
        raise Exception('Script specification must be int or omegaconf.dictconfig.DictConfig.')
      if spec['local']: # If local logging used, make sure data directory exists.
        self._make_data_dir(name)
      self.script[name] = spec

    self.manifold = manifold
    self.geometry = geometry
    self.density = density
    self.verbose = verbose

  def run_scripts(self, n_iter, samples):
    for name, spec in self.script.items():
      if n_iter % spec['freq'] == 0: # Run script.
        self.script[name]['runs'] += 1
        script = getattr(analysis, name)
        data = script(
            manifold = self.manifold,
            geometry = self.geometry, 
            density = self.density, 
            samples = samples, 
            **spec['params'])
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
        with h5py.File(path, 'a') as f: 
          f.create_dataset(name, data=value) 
        #torch.save(data, self._get_path(name, runs))

