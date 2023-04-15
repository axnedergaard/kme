import numpy as np
from scipy.stats import norm


class Sampler:
    def __init__(self, seed=None):
        if seed is None:
            seed = np.random.default_rng()
        self.rng = np.random.default_rng(seed)

    def sample_normal_matrix(self, shape, mean=0.0, variance=1.0):
        return self.rng.normal(loc=mean, scale=np.sqrt(variance), size=shape)

    def pdf_normal(self, mean, std, x):
        return norm.pdf(x, loc=mean, scale=std)

    def sample_normal(self, mean, std):
        return self.rng.normal(loc=mean, scale=std)

    def sample_uniform(self, low, high):
        return self.rng.uniform(low, high)

    def sample_uniform_integer(self, low, high):
        return self.rng.integers(low, high + 1)
