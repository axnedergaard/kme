from util import plotting
from util.data import load_experiment_data, get_plot_path, save_plot 
from matplotlib import pyplot as plt
import omegaconf
import hydra
import os

def get_script_fns(cfg):
  script_fns = {}
  for script, spec in cfg.script.items():
    spec = cfg.script[script]
    if isinstance(spec, omegaconf.dictconfig.DictConfig):
      spec = omegaconf.OmegaConf.to_container(spec, resolve=True)
    else:
      spec = {}
    script_fn = getattr(plotting, script)
    script_fns[script] = lambda data: script_fn(data, **spec)
  return script_fns

@hydra.main(config_path='config', config_name='plot', version_base='1.3')
def main(cfg):
  # Load the data.
  data = []
  for exp_name in cfg.exp_names: 
    exp_data = load_experiment_data(exp_name, local=cfg.local)
    data.append(exp_data)
  print('Loaded data.')
  # Make the plot.
  figures = {}
  for key, script_fn in get_script_fns(cfg).items():
    fig = script_fn(data)
    figures[key] = fig
  print('Made plots.')
  # Show or save the plot.
  if cfg.show:
    #for key, fig in figures.items():
    #  fig.show()
    plt.show()
  if cfg.name is not None:
    print(f'Saving plots to')
    for key, fig in figures.items():
      plot_name = f'{cfg.name}-{key}'
      save_plot(fig, plot_name)
      plot_path = os.path.join(os.getcwd(), get_plot_path(plot_name))
      print(f'{plot_path}')

if __name__ == '__main__':
  main()
