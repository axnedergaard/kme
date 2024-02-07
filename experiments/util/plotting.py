from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np


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
    :params: x (np.ndarray): The y-axis group data shape (T,)
    :params: y (np.ndarray): The x-axis group data shape (B, T)
    """
    assert y.dim() == 2 and x.dim() == 1 and x.shape[0] == y.shape[1]
    mean = np.mean(y, axis=0)
    percentiles = np.percentile(y, [low_perc, high_perc], axis=0)
    color = color if color else "black"
    ax.plot(x, mean, label=label, color=color)
    ax.fill_between(x, percentiles[0], percentiles[1], color=color, alpha=0.3)


def extinsic_rewards_vs_steps_single_env(data, env: str, ax: Axes = None, **kwargs):
    """
    Plot the extrinsic rewards vs steps with confidence interval
    :params: data (undefined) : The data to plot
    :params: env (str) : The environment to plot
    """

    def extract_and_format_data(data, **kwargs):
        out = []
        for group in data:
            timesteps = np.ndarray([])  # (T,)
            rewards = np.ndarray([[]])  # (B, T)
            color = "color for the group"
            label = "label for the group"
            out.append((timesteps, rewards, color, label))

        # return out
        raise NotImplementedError()

    if ax is None:
        fig, ax = plt.subplots()

    for timesteps, rewards, color, label in extract_and_format_data(data, **kwargs):
        plot_with_confidence_interval(ax, timesteps, rewards, color, label)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Extrinsic Reward")
    ax.title(f"{env} (Run Sparse)")

    if "grid" not in kwargs and not kwargs["grid"]:
        # by default the grid is shown
        ax.grid(visible=True)

    if "legend" not in kwargs and not kwargs["legend"]:
        # by default the legend is shown
        ax.legend(loc="upper left")

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
        extinsic_rewards_vs_steps_single_env(
            data, env, ax=axs[i], legend=(i == 0), **kwargs
        )

    plt.tight_layout()

    return fig
