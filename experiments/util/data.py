import h5py
from stable_baselines3 import PPO

def load_data(path):
  raise NotImplementedError

def save_data(data, path):
  with h5py.File(path, 'a') as f: 
    f.create_dataset(name, data=value) 

def load_policy(path, env):
  return PPO.load(path, env)

def save_policy(policy, path):
  policy.save(path)
