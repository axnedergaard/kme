import torch
from .base_encoder import BaseEncoder


class KMeansEncoder(BaseEncoder):
    def __init__(
        self, n_states, n_components, learning_rate, balancing_strength, differential
    ):
        super().__init__(n_states, n_components)
        self.learning_rate = learning_rate
        self.balancing_strength = balancing_strength
        self.differential = differential
        self.centroids = torch.randn((n_components, n_states)) * 0.1
        self.closest_distances = torch.zeros(n_components)

    def embed(self, state, learn=True):
        distances = torch.norm(self.centroids - state, dim=1)
        closest_cluster = torch.argmin(distances)
        closest_distance = distances[closest_cluster]

        if learn:
            update = self.learning_rate * (state - self.centroids[closest_cluster])
            self.centroids[closest_cluster] += update

            if self.differential:
                self.closest_distances[closest_cluster] = closest_distance

        return closest_cluster

    def reset(self):
        self.centroids = torch.randn((self.n_components, self.n_states)) * 0.1
        self.closest_distances = torch.zeros(self.n_components)
