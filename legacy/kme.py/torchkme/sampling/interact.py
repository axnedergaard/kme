from .sampler import sample_from, supported_distributions
from time import sleep
from sys import argv

# --- variables ---

n_points = 1000     # number of points to sample
dimensionality = 45 # roughly a mujoco env. state

# uniform hp
low = -19.
high = 12.

# gaussian hp
mean = 0.
std_dev = 1.

# gaussian-mixture hp
# n_components = 3
means = [0., 5., 10.]
std_devs = [1., 1., 1.]

# random-walk hp
step_size = 1.


# --- script ---

def interact(distribution, x, d, **kwargs):
    # sample x d-dimensional points from a given distribution
    samples = sample_from(distribution, n_points, dimensionality, **kwargs)
    settings = {k:v for (k,v) in kwargs.items()}
    print(f"sampling setting: {distribution}. {x} points of dim {d}")
    print(f"distribution settings: {settings}")
    for i, sample in enumerate(samples):
        print(f"\nenv. state {i}: {sample}")
        sleep(2)


if __name__ == "__main__":
    distribution = argv[1] if len(argv) > 1 else 'gaussian-mixture'

    if distribution not in supported_distributions:
        raise ValueError(f"Unsupported distribution: {distribution} not in {supported_distributions}")
    
    if distribution == 'uniform':
        interact(distribution, n_points, dimensionality, low=low, high=high)

    elif distribution == 'gaussian':
        interact(distribution, n_points, dimensionality, mean=mean, std_dev=std_dev)

    elif distribution == 'gaussian-mixture':
        interact(distribution, n_points, dimensionality, means=means, std_devs=std_devs)

    elif distribution == 'random-walk-man':
        interact(distribution, n_points, dimensionality, step_size=step_size)

    elif distribution == 'random-walk-euc':
        interact(distribution, n_points, dimensionality, step_size=step_size)
    
