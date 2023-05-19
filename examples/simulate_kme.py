from torchkme import sample_from
from torchkme import KMERewarder
from time import sleep

n_points = 1000 # number of states to sample
dim = 2 # roughly a mujoco env. state (R^n)

distribution = "gaussian-mixture"
means, std_devs = [0., 5., 20.], [1., 2., 5.]

def simulate():
    rewarder = KMERewarder(k=5, dim_states=dim, dim_actions=-1, learning_rate=1e-3, balancing_strength=1e-3, function_type="power")
    samples = sample_from(distribution, n_points, dim, means=means, std_devs=std_devs)
    for i, sample in enumerate(samples):
        print(f"\nenv. state {i}: {sample}")
        print(f"reward: {rewarder.infer(sample)}")
        sleep(2)

if __name__ == "__main__":
    simulate()
