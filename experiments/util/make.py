import rum
from omegaconf import OmegaConf

def make(cfg, _type, **kwargs):
  if _type not in cfg or 'name' not in cfg[_type]:
    print(f'No {_type} created.')
    return None
  cfg = OmegaConf.to_container(cfg[_type], resolve=True)
  name = cfg.pop('name')
  cfg.update(kwargs)
  module = getattr(rum, _type)
  cls = getattr(module, name)
  return cls(**cfg)

def make_environment(cfg, **kwargs):
  if 'domain_name' not in cfg['environment'] and 'task_name' not in cfg['environment']:
    print('No environment created.')
    return None
  cfg = OmegaConf.to_container(cfg['environment'], resolve=True)
  cfg.update(kwargs)
  domain_name = cfg.pop('domain_name')
  task_name = cfg.pop('task_name')
  env = rum.environment.load(domain_name, task_name, **cfg)
  env = rum.environment.GymnasiumWrapper(env)
  return env

  #if 'name' not in cfg['env']:
  #  print('No environment created.')
  #  return None
  #cfg = OmegaConf.to_container(cfg['env'], resolve=True)
  #if 'sparse' in cfg:
  #  sparse = cfg.pop('sparse')
  #else:
  #  sparse = False
  #domain_name = cfg.pop('name')

  #if domain_name in ['humanoid', 'cheetah', 'walker', 'quadruped']:
  #  task_name = 'run' 
  #elif domain_name in ['cartpole', 'acrobot']:
  #  task_name = 'swingup'
  #else:
  #  raise ValueError(f'Unknown environment domain name: {domain_name}')
  #if sparse:
  #  task_name += '_sparse'

  #env = rum.environment.load(domain_name, task_name, **cfg)
  #env = rum.dm2gym.GymnasiumWrapper(env)
  #return env
