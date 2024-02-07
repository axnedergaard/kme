from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np

COLORS = ['#1f77b4', '#ff7f0e', '#d62728',  '#2ca02c', '#9467bd']
CONFIDENCE_COLORS = ['#aec7e8', '#ffbb78', '#ff9896', '#98df8a', '#c5b0d5']


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
            group_key = exp_data.config["rewarder"]["name"]
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
    env = config["manifold"]["name"]

    for timesteps, rewards, color, label in extract_and_format_data(data, **kwargs):
        plot_with_confidence_interval(ax, timesteps, rewards, color, label)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Extrinsic Reward")
    ax.set_title(f"{env} (Run Sparse)")

    # if "grid" not in kwargs and not kwargs["grid"]:
    #     # by default the grid is shown
    #     ax.set_grid(visible=True)

    # if "legend" not in kwargs and not kwargs["legend"]:
    #     # by default the legend is shown
    #     ax.set_legend(loc="upper left")

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
        raise NotImplementedError()

    fig, ax = plt.subplots()
    k, kmeans_loss, color, label = extract_and_format_data(data, **kwargs)
    plot_with_confidence_interval(ax, k, kmeans_loss, color, label)

    ax.set_xlabel("K")
    ax.set_ylabel("Kmeans objective")
    ax.title("Kmeans Loss vs K")

    if "grid" not in kwargs and not kwargs["grid"]:
        # by default the grid is shown
        ax.grid(visible=True)

    if "legend" not in kwargs and not kwargs["legend"]:
        # by default the legend is shown
        ax.legend(loc="upper left")

    return fig


def count_variance_vs_beta(data, **kwargs):
    """
    Plot the count variance vs beta with confidence interval
    :params: data (undefined) : The data to plot
    """

    def extract_and_format_data(data, **kwargs):
        raise NotImplementedError()

    fig, ax = plt.subplots()
    beta, count_variance, color, label = extract_and_format_data(data, **kwargs)
    plot_with_confidence_interval(ax, beta, count_variance, color, label)

    ax.set_xlabel("Beta")
    ax.set_ylabel("Count Variance")
    ax.title("Count Variance vs Beta")

    if "grid" not in kwargs and not kwargs["grid"]:
        # by default the grid is shown
        ax.grid(visible=True)

    if "legend" not in kwargs and not kwargs["legend"]:
        # by default the legend is shown
        ax.legend(loc="upper left")

    return fig


def scale_independent_loss_pdf_vs_steps(data, **kwargs):
    """
    Plot the scale independant loss pdf vs steps with confidence interval
    :params: data (undefined) : The data to plot
    """

    def extract_and_format_data(data, **kwargs):
        raise NotImplementedError()

    fig, ax = plt.subplots()
    steps, scale_independant_loss_pdf, color, label = extract_and_format_data(data, **kwargs)
    plot_with_confidence_interval(ax, steps, scale_independant_loss_pdf, color, label)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Scale Independant Loss PDF")
    ax.title("Scale Independant Loss PDF vs Steps")

    if "grid" not in kwargs and not kwargs["grid"]:
        # by default the grid is shown
        ax.grid(visible=True)

    if "legend" not in kwargs and not kwargs["legend"]:
        # by default the legend is shown
        ax.legend(loc="upper left")

    return fig


def scale_independent_loss_distance_vs_steps(data, **kwargs):
    """
    Plot the scale independant loss distance vs steps with confidence interval
    :params: data (undefined) : The data to plot
    """

    def extract_and_format_data(data, **kwargs):
        raise NotImplementedError()

    fig, ax = plt.subplots()
    steps, scale_independant_loss_distance, color, label = extract_and_format_data(data, **kwargs)
    plot_with_confidence_interval(ax, steps, scale_independant_loss_distance, color, label)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Scale Independant Loss Distance")
    ax.title("Scale Independant Loss Distance vs Steps")

    if "grid" not in kwargs and not kwargs["grid"]:
        # by default the grid is shown
        ax.grid(visible=True)

    if "legend" not in kwargs and not kwargs["legend"]:
        # by default the legend is shown
        ax.legend(loc="upper left")

    return fig
