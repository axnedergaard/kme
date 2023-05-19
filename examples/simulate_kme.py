from torchkme import sample_from
from torchkme import KMERewarder
import matplotlib.pyplot as plt
from time import sleep
import numpy as np

n_points = 500 # number of states to sample
dim = 2 # roughly a mujoco env. state (R^n)

distribution = "random-walk-euc"
means, std_devs = [0., 0., 0.], [10., 20., 50.]

def plot(states, encoder, assignments):
    # states is a np.ndarray of shape (n_points, dim)
    # encoder is a KMeansEncoder object with access to centroids
    # centroids is a torch.tensor of shape (k, dim)
    # assignments is a np.ndarray of shape (n_points,)
    # that index to which centroids each state belongs to
    # write a function that plots the states and centroids

    if not isinstance(states, np.ndarray):
        states = states.cpu().numpy()

    # get centroids and convert to numpy
    centroids = encoder.centroids.cpu().detach().numpy()

    # create a color map with as many colors as there are centroids
    colors = plt.cm.rainbow(np.linspace(0, 1, len(centroids)))

    if states.shape[1] == 1:
        # 1D data
        for i, centroid in enumerate(centroids):
            mask = (assignments == i)
            plt.scatter(states[mask], np.zeros_like(states[mask]), color=colors[i], label=f'Cluster {i}')
        plt.scatter(centroids, np.zeros_like(centroids), color='black', label='Centroids', marker='x')
        plt.yticks([])  # hide the y-axis as it's not meaningful here
    elif states.shape[1] == 2:
        # 2D data
        for i, centroid in enumerate(centroids):
            mask = (assignments == i)
            plt.scatter(states[mask, 0], states[mask, 1], color=colors[i], label=f'Cluster {i}')
        plt.scatter(centroids[:, 0], centroids[:, 1], color='black', label='Centroids', marker='x')
    else:
        raise ValueError('Data dimensionality is too high to plot directly')

    plt.legend()
    plt.show()


def simulate():
    rewarder = KMERewarder(k=5, dim_states=dim, dim_actions=-1, learning_rate=0.05, balancing_strength=1e-3, function_type="power")
    samples = sample_from(distribution, n_points, dim, means=means, std_devs=std_devs)
    assignments = np.zeros(n_points)

    for i, sample in enumerate(samples):
        reward, pathological, cluster_idx = rewarder.infer(sample)
        assignments[i] = cluster_idx

        # print(f"\nenv. state {i}: {sample}")
        # print(f"reward: {rewarder.infer(sample)}")
        # print(f"cluster sizes: {rewarder.k_encoder.cluster_sizes}")
        # print(f"centroids: {rewarder.k_encoder.centroids}")
        # sleep(0.5)

    plot(samples, rewarder.k_encoder, assignments)

if __name__ == "__main__":
    simulate()
