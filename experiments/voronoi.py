from matplotlib import pyplot as plt
from scipy.spatial import voronoi_plot_2d, Voronoi
from scipy.stats import kstest, uniform
from scipy.spatial import ConvexHull
from scipy.special import gamma

from rum.density.kmeans_estimator import OnlineKMeansEstimator
from rum.manifold.euclidean import EuclideanManifold

import numpy as np
import torch
import math


def voroplot(ax, centers, radiuses, vorpoints):
    voronoi_plot_2d(vorpoints, ax=ax, point_size=3, show_vertices=False)
    for i, center in enumerate(centers):
        circle = plt.Circle(center, radiuses[i], color="orange", fill=False)
        ax.add_artist(circle)


def trajplot(ax, states, idx, sampling_type, buffer=50):
    lb = max(0, idx - buffer)
    states = states.numpy()
    if sampling_type == 'sample':
        ax.scatter(states[lb:idx, 0], states[lb:idx, 1], s=1, color="red", label="samples")
    elif sampling_type == 'rw':
        ax.plot(states[lb:idx, 0], states[lb:idx, 1], label="rw path", color="red", linewidth=1)
        ax.scatter(states[lb:idx, 0], states[lb:idx, 1], s=1, color="red")
    else:
        raise ValueError("Unknown sampling type.")

def voronoi_volumes(vorpoints):
    vol = np.zeros(vorpoints.npoints)
    for i, reg_num in enumerate(vorpoints.point_region):
        indices = vorpoints.regions[reg_num]
        if -1 in indices:  # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(vorpoints.vertices[indices]).volume
    return vol


def volume_nsphere(radius, n):
    volume = (math.pi ** (n / 2)) / gamma((n / 2) + 1) * (radius**n)
    return volume


def compute_ratio(radiuses, vorpoints):
    clusters_vols = [volume_nsphere(r, dim) for r in radiuses]
    voronoi_vols = voronoi_volumes(vorpoints)
    ratio = clusters_vols / voronoi_vols
    ratio = ratio[ratio != 0]
    avg_ratio = ratio.mean()
    std_ratio = ratio.std()

    # Normalize ratios to have a range [0, 1] for the K-S test
    # Perform the K-S test against a uniform distribution
    ratio_normalized = (ratio - ratio.min()) / (ratio.max() - ratio.min())
    ks_stat, p_value = kstest(ratio_normalized, 'uniform')

    print(f"Average Ratio: {avg_ratio:.2f} ± {std_ratio:.2f} Uniform p-value: {p_value:.3f}", end="\r")

k, dim = 50, 2
x_low, x_high = -1, 1
states_per_round = 5000
states_per_iter = 20
idx = states_per_round

sampling_type = 'sample' # 'rw' or 'sample'
m = EuclideanManifold(dim=dim, sampler={"type": "uniform", "low": x_low, "high": x_high})
kmeans = OnlineKMeansEstimator(k=k, dim_states=dim)

if dim == 2:
    fig, ax = plt.subplots(figsize=(6, 6))

try:
    while True:
        if dim == 2:
            ax.clear()

        # Generate some states from a walk.
        if idx == states_per_round:
            if sampling_type == 'sample':
                states = m.sample(states_per_round)
            elif sampling_type == 'rw':
                states = m.random_walk(states_per_round)
            else:
                raise ValueError("Unknown sampling type.")
            states = torch.Tensor(states)
            idx = 0
        
        s = states[idx:idx+states_per_iter].view(-1, dim)
        
        # Estimate the density with Kmeans.
        kmeans.learn(s)
        
        # Fetch and compute information.
        centers = kmeans.centroids.numpy()
        radiuses = kmeans.diameters.numpy() / 2
        vorpoints = Voronoi(centers)

        # Information about ratio.
        compute_ratio(radiuses, vorpoints)

        if dim == 2:
            # Plot everything we need to plot.
            voroplot(ax, centers, radiuses, vorpoints)
            trajplot(ax, states, idx, sampling_type, buffer=max(20, states_per_iter))

            # Set plot limits and title.
            ax.set_xlim(x_low, x_high)
            ax.set_ylim(x_low, x_high)
            ax.set_title(f"kmeans density estimate live k={k}")
            ax.legend(loc="upper left")
        
        # Move the clock forward.
        idx += states_per_iter
        plt.pause(1e-6)

        if dim == 2 and not plt.fignum_exists(fig.number):
            # Exit the loop if the window is closed
            print("\nPlot window closed.")
            break

except KeyboardInterrupt:
    # Exit the program if the user stops it.
    print("\nStopped by user.")
    plt.close(fig)
