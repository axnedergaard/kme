import numpy as np


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def random_color(sampler):
    color = np.array(
        [
            sampler.sample_uniform(0, 1),
            sampler.sample_uniform(0, 1),
            sampler.sample_uniform(0, 1),
        ]
    )
    if np.all(color < 0.2):
        return random_color(sampler)
    return color


def distance_to_similarity(distance):
    return np.exp(-distance)


def similarity_to_distance(similarity):
    return -np.log(similarity)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def tanhd(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2


def rescale_tanh(x):
    return (x + 1.0) / 2.0


def clip(x, min_value, max_value):
    return np.clip(x, min_value, max_value)


def clip_unit(x):
    return clip(x, -1.0, 1.0)


def clip_gradients(x):
    return clip(x, -0.1, 0.1)
