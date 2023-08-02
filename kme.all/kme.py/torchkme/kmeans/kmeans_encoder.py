import copy
from typing import Callable, Tuple

import torch
from torch import Tensor

from ..constvars import device, dtype


class KMeansEncoder:

    def __init__(
        self,
        k: int,                     # n_components of (online) kmeans(++)
        dim_states: int,            # dimension of environment states (R^n)
        learning_rate: float,       # alpha - learning rate of kmeans
        balancing_strength: float,  # kappa - balancing strength of kmeans
        homeostasis: bool = True,   # homeostasis - whether to use homeostasis
    ) -> None:

        # kmeans specs
        self.k: int = k
        self.dim_states: int = dim_states

        # tunable hyperparameters
        self.hp_learning_rate: Tensor = torch.tensor(learning_rate, dtype=dtype, device=device)
        self.hp_balancing_strength: Tensor = torch.tensor(balancing_strength, dtype=dtype, device=device)
        self.hp_homeostasis: bool = homeostasis

        # internal kmeans encoder state
        self.centroids: Tensor = self._init_centroids(self.k, self.dim_states) # mu_i
        self.cluster_sizes: Tensor = torch.zeros((self.k,), dtype=dtype, device=device) # n_i
        self.closest_distances: Tensor = torch.zeros((self.k,), dtype=dtype, device=device) # M_i


    # --- public interface methods ---

    def clone(self) -> 'KMeansEncoder':
        return copy.deepcopy(self)

    def update(self, state: Tensor) -> Tuple['KMeansEncoder', int]:
        # Updates the internal state of the KMeansEncoder with a new state.
        # according to algorithm (1) in https://arxiv.org/pdf/2205.15623.pdf
        closest_cluster_idx = self._find_closest_cluster(state)
        self._online_update_clusters(state, closest_cluster_idx)
        self.closest_distances = self._dist_to_clusters(state, self._euclidean_dist)
        # CHECK. we dont have anymore access to pathological updates count here.
        return self, closest_cluster_idx


    def sim_update_v1(self, state: Tensor) -> Tuple['KMeansEncoder']:
        # Simulates a KMeansEncoder update with a new state.
        return self.clone().update(state)


    # --- private interface methods ---

    def _init_centroids_zeros(self, k: int, dim_states: int) -> Tensor:
        # Initializes centroids at zero
        return torch.zeros((k, dim_states), dtype=dtype, device=device)
    
    def _init_centroids(self, k: int, dim_states: int) -> Tensor:
        # Initialize the first centroid randomly from a standard normal distribution
        centroids = torch.randn((1, dim_states), dtype=dtype, device=device)
        
        for i in range(1, k):
            # fit centroids to mixture of normal distributions
            sample_points = torch.randn((1000, dim_states), dtype=dtype, device=device) # adjust the sample size as needed
            distances = torch.min(self._euclidean_dist(sample_points.unsqueeze(1), centroids.unsqueeze(0), dim=-1), dim=1)[0]
            probabilities = distances / torch.sum(distances)
            new_centroid_idx = torch.multinomial(probabilities, 1)
            new_centroid = sample_points[new_centroid_idx].view(1, dim_states)
            centroids = torch.cat((centroids, new_centroid), dim=0)
        
        return centroids

    def _euclidean_dist(self, t1: Tensor, t2: Tensor, p: float = 2) -> Tensor:
        # Computes Euclidean distance between two torch.Tensors objects.
        return torch.norm(t1 - t2, p=p, dim=-1)


    def _dist_to_clusters(self, state: Tensor, dist_fn: Callable) -> Tensor:
        # Computes objective distances between a given state and all centroids.
        distances = dist_fn(state.unsqueeze(0), self.centroids.unsqueeze(1)).view(-1)

        if self.hp_homeostasis:
            mean = torch.mean(self.cluster_sizes).item()
            distances += self.hp_balancing_strength * (self.cluster_sizes - mean)
        
        return distances


    def _find_closest_cluster(self, state: Tensor) -> int:
        # Finds the closest cluster and distance to a given state.
        distances: Tensor = self._dist_to_clusters(state, self._euclidean_dist)
        closest_cluster_idx = torch.argmin(distances)
        return closest_cluster_idx.item() # .CHECK for GPU


    def _online_update_clusters(self, state: Tensor, closest_cluster_idx: int) -> None:
        # Online update of closest cluster centroid and size with new state.
        self._update_cluster_centroid(state, closest_cluster_idx)
        self._update_cluster_size(closest_cluster_idx)


    def _update_cluster_centroid(self, state: Tensor, closest_cluster_idx: int) -> None:
        # Online update of closest cluster centroid with new state.
        state_contribution = self.hp_learning_rate * state
        centroid_contribution = (1 - self.hp_learning_rate) * self.centroids[closest_cluster_idx]
        self.centroids[closest_cluster_idx] = state_contribution + centroid_contribution


    def _update_cluster_size(self, closest_cluster_idx: int) -> None:
        # Online update of closest cluster size with new state.
        self.cluster_sizes[closest_cluster_idx] += 1

    