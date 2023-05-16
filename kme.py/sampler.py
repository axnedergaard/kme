import numpy as np

# --- public interface ---

__all__ = ['sample_from', 'supported_distributions']


# --- private interface ---

def _uniform(x, dim, low, high):
    return np.random.uniform(low=low, high=high, size=(x, dim))


def _gaussian(x, dim, mean, std_dev):
    return np.random.normal(loc=mean, scale=std_dev, size=(x, dim))


def _gaussian_mixture(x, dim, means, std_devs):
    assert len(means) == len(std_devs)
    n_components = len(means)
    indices = np.random.choice(n_components, size=x)
    samples = [_gaussian(1, dim, means[indices[i]], std_devs[indices[i]]) for i in range(x)]
    return np.stack([s.reshape(-1) for s in samples])


def _random_walk(x, dim):
    steps = np.random.choice([-1, 1], size=(x, dim))
    return np.cumsum(steps, axis=0)


# --- public interface ---

supported_distributions = ['uniform', 'gaussian', 'gaussian-mixture', 'random-walk']


def sample_from(distribution, x, d, **kwargs):
    # sample x d-dimensional points from a given distribution
    # supports: uniform, gaussian, gaussian-mixture, random-walk

    if distribution not in supported_distributions:
        raise ValueError(f"Unsupported distribution: {distribution} not in {supported_distributions}")
    
    if distribution == 'uniform':
        low = kwargs.get('low')
        high = kwargs.get('high')
        return _uniform(x, d, low, high)

    elif distribution == 'gaussian':
        mean = kwargs.get('mean')
        std_dev = kwargs.get('std_dev')
        return _gaussian(x, d, mean, std_dev)

    elif distribution == 'gaussian-mixture':
        means = kwargs.get('means')
        std_devs = kwargs.get('std_devs')
        return _gaussian_mixture(x, d, means, std_devs)

    elif distribution == 'random_walk':
        return _random_walk(x, d)
    
