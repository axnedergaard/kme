from sampler import sample_from
from time import sleep

# --- variables ---

n_points = 1000     # number of points to sample
dimensionality = 45 # roughly a mujoco env. state

distribution = "gaussian-mixture"
# n_components = 3
means = [0., 5., 10.]
std_devs = [1., 1., 1.]

# --- script ---

def interact(distribution, x, d, **kwargs):
    # sample x d-dimensional points from a given distribution
    samples = sample_from(distribution, n_points, dimensionality, means=means, std_devs=std_devs)
    print(f"sampling setting: {distribution}. {x} points of dim {d}")
    for i, sample in enumerate(samples):
        print(f"\nenv. state {i}: {sample}")
        sleep(2)


if __name__ == "__main__":
    interact(distribution, n_points, dimensionality, means=means, std_devs=std_devs)
