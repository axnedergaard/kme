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


def _random_walk_man(x, dim, step_size=1.):
    # random walk in manhattan space (orthogonal steps only)
    steps = np.random.choice([-step_size, 0, step_size], size=(x, dim))
    return np.cumsum(steps, axis=0)


def _random_walk_euc(x, dim, step_size=1.):
    # random walk in euclidean space
    # choose directions uniformly from the unit sphere 
    # by normalizing a vector of normal random variables
    random_directions = np.random.normal(size=(x, dim))
    random_directions /= np.linalg.norm(random_directions, axis=1)[:, np.newaxis]
    steps = step_size * random_directions
    return np.cumsum(steps, axis=0)


# --- public interface ---

supported_distributions = ['uniform', 'gaussian', 'gaussian-mixture', 'random-walk-man', 'random-walk-euc']


def sample_from(distribution, x, d, **kwargs):
    # sample x d-dimensional points from a given distribution
    # supports: uniform, gaussian, gaussian-mixture, random-walk
    print(x, d, kwargs)

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

    elif distribution == 'random-walk-man':
        step_size = kwargs.get('step_size', 1.0)
        return _random_walk_man(x, d, step_size=step_size)
    
    elif distribution == 'random-walk-euc':
        step_size = kwargs.get('step_size', 1.0)
        return _random_walk_euc(x, d, step_size=step_size)
    
