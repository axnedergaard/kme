import h5py
import matplotlib.pyplot as plt
import wandb
from stable_baselines3 import PPO

class Data:
  def __init__(self, run):
    self.config = run.config
    self.df = run.history()

  def __getitem__(self, key):
    values = self.df[key]
    non_nan_values = values[values.notna()]
    return non_nan_values.to_numpy()

def get_plot_path(fn):
  return f"outputs/plots/{fn}.png"

def load_experiment_data(experiment_name, local=False):
  if local:
    raise NotImplementedError
  else:
    api = wandb.Api()
    run = api.run(f"test/{experiment_name}")
    data = Data(run)
    return data

def save_plot(fig, fn):
  path = get_plot_path(fn)
  fig.savefig(path)

def load_data(path):
  raise NotImplementedError

def save_data(data, path):
  with h5py.File(path, 'a') as f: 
    f.create_dataset(name, data=value) 

def load_policy(path, env):
  return PPO.load(path, env)

def save_policy(policy, path):
  policy.save(path)
