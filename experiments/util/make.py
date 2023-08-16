import rum
from omegaconf import OmegaConf

def make(cfg, _type):
  if _type not in cfg or 'name' not in cfg[_type]:
    return None
  cfg = OmegaConf.to_container(cfg[_type], resolve=True)
  name = cfg.pop('name')
  module = getattr(rum, _type)
  cls = getattr(module, name)
  return cls(**cfg)
