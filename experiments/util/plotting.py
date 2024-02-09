from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np

COLORS = ['#1f77b4', '#ff7f0e', '#d62728',  '#2ca02c', '#9467bd']
CONFIDENCE_COLORS = ['#aec7e8', '#ffbb78', '#ff9896', '#98df8a', '#c5b0d5']

def _get_environment_name(config):
  if config['environment'] == {}: # Manifold.
    return config['manifold']['name']
  else: # MuJoCo.
    return config['environment']['domain_name'] 

def _get_rewarder_name(config):
  if config['rewarder'] == {}:
    return 'None'
  else:
    return config['rewarder']['name']

def dummy(data, **kwargs):
  entropy = data[0]['entropy']
  fig, ax = plt.subplots()
  ax.plot(entropy)
  return fig

def plot_with_confidence_interval(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str = None,
    label: str = None,
    low_perc: int = 25,
    high_perc: int = 75,
):
    """
    Plot the mean of the given data with a confidence interval
    :params: x (np.ndarray): The x-axis group data shape (T,)
    :params: y (np.ndarray): The y-axis group data shape (B, T)
    """
    assert len(x.shape) == 1 and len(y.shape) == 2 and x.shape[0] == y.shape[1]
    mean = np.mean(y, axis=0)
    percentiles = np.percentile(y, [low_perc, high_perc], axis=0)
    color = color if color else "black"
    ax.plot(x, mean, label=label, color=color)
    ax.fill_between(x, percentiles[0], percentiles[1], color=color, alpha=0.3)


def extrinsic_rewards_vs_steps_single_env(data, ax: Axes = None, **kwargs):
    """
    Plot the extrinsic rewards vs steps with confidence interval
    :params: data (undefined) : The data to plot
    """

    def extract_and_format_data(data, **kwargs):
        
        groups = {}

        for exp_data in data:
            group_key = _get_rewarder_name(exp_data.config)
            rewards = exp_data["extrinsic_reward"]
            if group_key not in groups:
                groups[group_key] = {"rewards": []}
            groups[group_key]["rewards"].append(rewards)

        for idx, (group_key, _) in enumerate(groups.items()):
            groups[group_key]["color"] = COLORS[idx]

        timesteps = data[0]["time/total_timesteps"]

        return [
            (timesteps, np.array(group["rewards"]), group["color"], group_key)
            for group_key, group in groups.items()
        ]

    if ax is None:
        fig, ax = plt.subplots()

    config = data[0].config
    env_name = _get_environment_name(config)

    for timesteps, rewards, color, label in extract_and_format_data(data, **kwargs):
        plot_with_confidence_interval(ax, timesteps, rewards, color, label)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Extrinsic Reward")
    ax.set_title(f"{env_name.capitalize()} (Run Sparse)")

    # By default the grid is not shown.
    if 'grid' in kwargs: 
        ax.grid(visible=kwargs['grid'])
    else:
        ax.grid(visible=False)

    # By default the legend is shown.
    if 'legend' not in kwargs or kwargs['legend']:
        ax.legend(loc="upper right")
    
    return fig


def extinsic_rewards_vs_steps_multiple_envs(data, **kwargs):
    """
    Plot the extrinsic rewards vs steps with confidence interval multiple environments
    :params: data (undefined) : The data to plot
    """

    def extract_and_format_data(data, **kwargs):
        raise NotImplementedError()

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()

    envs, data = extract_and_format_data(data, **kwargs)
    for i, (env, data) in enumerate(zip(envs, data)):
        extrinsic_rewards_vs_steps_single_env(
            data, env, ax=axs[i], legend=(i == 0), **kwargs
        )

    plt.tight_layout()

    return fig


def kmeans_loss_vs_k(data, **kwargs):
    """
    Plot the kmeans loss vs k with confidence interval
    :params: data (undefined) : The data to plot
    """

    def extract_and_format_data(data, **kwargs):

        groups = {} # Each group is a different value of K

        for exp_data in data:
            # Grab the value of K for that experiment
            k = exp_data.config["density"]["k"]
            # Grab the final kmeans loss for that experiment
            kmeans_final_loss = exp_data["kmeans_loss"][-1]
            if k not in groups:
                groups[k] = []
            # Append the final kmeans loss to the group
            # Those values will get averaged and plotted
            groups[k].append(kmeans_final_loss)

        # Sort the groups by K
        groups = dict(sorted(groups.items()))

        # Assert all the groups have the same length
        assert len(set([len(v) for v in groups.values()])) == 1, "All groups must have the same length"

        # Grab the data we need
        k = np.array(list(groups.keys()))
        kmeans_loss = np.array(list(groups.values())).T
        label = "KMeans loss"

        return (k, kmeans_loss, COLORS[0], label)

    fig, ax = plt.subplots()
    k, kmeans_loss, color, label = extract_and_format_data(data, **kwargs)
    plot_with_confidence_interval(ax, k, kmeans_loss, color, label)

    ax.set_xlabel("K")
    ax.set_ylabel("Kmeans objective")
    ax.set_title("Kmeans Loss vs K")

    if kwargs.get("grid", True):
        # by default the grid is shown
        ax.grid(visible=True)

    if kwargs.get("legend", True):
        # by default the legend is shown
        ax.legend(loc="upper left")
    
    return fig


def count_variance_vs_beta(data, **kwargs):
    """
    Plot the count variance vs beta with confidence interval
    :params: data (undefined) : The data to plot
    """

    def extract_and_format_data(data, **kwargs):
        
        groups = {} # Each group is a different value of beta

        for exp_data in data:
            # Grab the value of beta for that experiment
            beta = exp_data.config["density"]["balancing_strength"]
            # Grab the final count variance for that experiment
            count_variance = exp_data["kmeans_count_variance"][-1]
            if beta not in groups:
                groups[beta] = []
            # Append the final count variance to the group
            # Those values will get averaged and plotted
            groups[beta].append(count_variance)

        # Sort the groups by K
        groups = dict(sorted(groups.items()))

        # Assert all the groups have the same length
        assert len(set([len(v) for v in groups.values()])) == 1, "All groups must have the same length"

        # Grab the data we need
        beta = np.array(list(groups.keys()))
        count_variance = np.array(list(groups.values())).T
        label = "Count Variance"

        return (beta, count_variance, COLORS[0], label)

    fig, ax = plt.subplots()
    beta, count_variance, color, label = extract_and_format_data(data, **kwargs)
    plot_with_confidence_interval(ax, beta, count_variance, color, label)

    ax.set_xlabel("Beta")
    ax.set_ylabel("Count Variance")
    ax.set_title("Count Variance vs Beta")

    if kwargs.get("grid", True):
        # by default the grid is shown
        ax.grid(visible=True)

    if kwargs.get("legend", True):
        # by default the legend is shown
        ax.legend(loc="upper left")
    
    return fig


def scale_independent_loss_pdf_vs_steps(data, **kwargs):
    """
    Plot the scale independant loss pdf vs steps with confidence interval
    :params: data (undefined) : The data to plot
    """

    def extract_and_format_data(data, **kwargs):
        
        experiments = []

        for exp_data in data:
            # Grab the steps range for that experiment
            steps = exp_data["_step"]
            # Grab the scale independant loss pdf for that experiment
            scale_independant_loss_pdf = exp_data["scale_independent_loss"]
            experiments.append((steps, scale_independant_loss_pdf))

        # Assert all the sublists have the same length
        unique_lengths = {len(l) for exp in experiments for l in exp}
        assert len(unique_lengths) == 1 and all(len(tup[0]) == len(tup[1]) for tup in experiments)

        # Grab the data we need
        steps = experiments[0][0]
        scale_independant_loss_pdf = np.array([tup[1] for tup in experiments])
        label = "Scale independent loss pdf"

        return (steps, scale_independant_loss_pdf, COLORS[0], label)

    fig, ax = plt.subplots()
    steps, scale_independant_loss_pdf, color, label = extract_and_format_data(data, **kwargs)
    plot_with_confidence_interval(ax, steps, scale_independant_loss_pdf, color, label)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Scale Independant Loss PDF")
    ax.set_title("Scale Independant Loss PDF vs Steps")

    if kwargs.get("grid", True):
        # by default the grid is shown
        ax.grid(visible=True)

    if kwargs.get("legend", True):
        # by default the legend is shown
        ax.legend(loc="upper left")
    
    return fig


def scale_independent_loss_distance_vs_steps(data, **kwargs):
    """
    Plot the scale independant loss distance vs steps with confidence interval
    :params: data (undefined) : The data to plot
    """

    def extract_and_format_data(data, **kwargs):
                
        experiments = []

        for exp_data in data:
            # Grab the steps range for that experiment
            steps = exp_data["_step"]
            # Grab the scale independant distance_loss for that experiment
            scale_independant_loss_pdf = exp_data["distance_loss"]
            experiments.append((steps, scale_independant_loss_pdf))

        # Assert all the sublists have the same length
        unique_lengths = {len(l) for exp in experiments for l in exp}
        assert len(unique_lengths) == 1 and all(len(tup[0]) == len(tup[1]) for tup in experiments)

        # Grab the data we need
        steps = experiments[0][0]
        scale_independant_loss_pdf = np.array([tup[1] for tup in experiments])
        label = "Scale independent distance loss"

        return (steps, scale_independant_loss_pdf, COLORS[0], label)

    fig, ax = plt.subplots()
    steps, scale_independant_loss_distance, color, label = extract_and_format_data(data, **kwargs)
    plot_with_confidence_interval(ax, steps, scale_independant_loss_distance, color, label)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Scale Independant Loss Distance")
    ax.set_title("Scale Independant Loss Distance vs Steps")

    if kwargs.get("grid", True):
        # by default the grid is shown
        ax.grid(visible=True)

    if kwargs.get("legend", True):
        # by default the legend is shown
        ax.legend(loc="upper left")
    
    return fig

def entropy_ranking(data, **kwargs):
  def get_label(config):
    # Return sampler label.
    # For now, samplers are always on manifolds.
    manifold_name = config['manifold']['name']
    sampler_name = config['manifold']['sampler']['type'].capitalize()
    return f"{manifold_name}-{sampler_name}"

  def extract_and_format_data(data):
    entropies = {}
    labels = []
    # Add each entropy and sampler label to list at density estimatior key.
    for exp_data in data:
      label = get_label(exp_data.config)
      # If there is an entropy but no density, the manifold (ground truth) was used to compute entropy.
      density_name = exp_data.config['density']['name'] if exp_data.config['density'] != {} else "True" 
      entropies.setdefault(density_name, []).append([exp_data["entropy"], label])
      if label not in labels:
        labels.append(label)
    # Sort each list according to sampler labels. # TODO. Somehow need to have ordering based on actual entropies, not string ordering.
    for key in entropies:
      entropies[key].sort(key=lambda x: x[1])
      for i in range(len(entropies[key])):
        # Remove sampler label tags used for sorting.
        values = entropies[key][i][0]
        # Average entropies where density estimator and labels are the same.
        entropies[key][i] = np.mean(values)
    return entropies, labels

  fig, ax = plt.subplots()
  entropies, labels = extract_and_format_data(data)

  for i, (key, value) in enumerate(entropies.items()):
    assert len(value) == len(labels)
    ax.bar(range(len(value)), value, label=key, color=COLORS[i])

  ax.set_xticks(range(len(labels)), labels)
  ax.set_xlabel("Density")
  ax.set_ylabel("Entropy")
  ax.set_title("Entropy ranking")

  return fig
