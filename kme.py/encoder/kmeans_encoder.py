import math
from abc import abstractmethod
from .base_encoder import BaseEncoder
from torch import Tensor
import torch
from typing import Optional, Tuple

import numpy as np


class KMeansEncoder(BaseEncoder):

    EPS: int = 1
    
    def __init__(
        self,
        n_states: Optional[int] = None,
        n_dims: Optional[int] = None,
        k: Optional[int] = None,
        learning_rate: Optional[float] = None,
        balancing_strength: Optional[float] = None,
        homeostasis: Optional[bool] = None,
    ):
        super().__init__(n_states, n_dims)

        # private attributes
        self.learning_rate: float = 0.0
        self.balancing_strength: float = 0.0
        self.homeostasis: int = 0
        self.distances: list[float] = []
        self.n_points_std: int = 0

        # public attributes
        self.closest_clusters: list[int] = []
        self.cluster_centers: list[Tensor] = []
        self.closest_distances: list[float] = []
        self.n_points: list[float] = []
        self.pathological_updates: int = 0 # For investigating practical time complexity.


    @abstractmethod
    def embed(self, state: Tensor, repr: Tensor) -> Tensor:
        return super().embed(state, repr)
    
    @abstractmethod
    def reset(self) -> None:
        return super().reset()
    
    @abstractmethod
    def _find_closest(index: int, max_distance_index: list[int]) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def _compute_distance(array_1: list[float], array_2: list[Tensor]) -> float:
        raise NotImplementedError

    @abstractmethod
    def _update_point_statistics() -> None:
        raise NotImplementedError
    

    # --- public implemented methods ---

    def embed(self, state: Tensor, repr: Tensor) -> Tensor:
        # Embeds the input state into a lower-dimensional representation.
        #  Find the closest neuron.
        for i in range(self.n_dims):
            distance: float = self._compute_distance(state, self.cluster_centers + i * self.n_states)
            self.distances[self.n_dims * (self.n_dims + 1) + i] = distance
            self.distances[i * (self.n_dims + 1) + self.n_dims] = distance

        self.distances[self.n_dims * (self.n_dims + 1) + self.n_dims] = 0.0
        closest_index: int = self._find_closest(self.n_dims)

        # Merge new snapshot with existing neurons (k-means)
        for i in range(self.n_states):
            self.cluster_centers[closest_index * self.n_states + i] += self.learning_rate * state[i] 
            + (1 - self.learning_rate) * self.cluster_centers[closest_index * self.n_states + i]

        if self.homeostasis:
            # Increase count for closest neuron.
            self.n_points[closest_index] += 1.0

        # Recompute distances
        updated_closest_distance_homestatic: float = math.inf
        updated_closest_cluste: float = math.inf
        num_pathological_updates: int = 0

        for i in range(self.n_dims):
            if i == closest_index:
                self.distances[closest_index * (self.n_dims + 1) + i] = 0.0
                self.distances[i * (self.n_dims + 1) + closest_index] = 0.0
        
            distance: float = self._compute_distance(
                self.cluster_centers + closest_index * self.n_states,
                self.cluster_centers + i * self.n_states
            )

            self.distances[closest_index * (self.n_dims + 1) + i] = distance
            self.distances[i * (self.n_dims + 1) + closest_index] = distance

            distance_homeostatic: float = distance + self.balancing_strength * \
                (self.n_points[i] - self.n_points[closest_index]) if self.homeostasis else distance
            
            distance_homeostatic_dual: float = distance + self.balancing_strength * \
                (self.n_points[closest_index] - self.n_points[i]) if self.homeostasis else distance
            
            if distance_homeostatic < 0: distance_homeostatic = 0.0
            if distance_homeostatic_dual < 0: distance_homeostatic_dual = 0.0

            if distance_homeostatic < updated_closest_distance_homestatic:
                updated_closest_distance_homestatic = distance_homeostatic
                updated_closest_cluster = i
            
            # Update clostest distance if distance to new cluster is shorter
            if distance_homeostatic_dual <= self.closest_distances[i]:
                self.closest_distances[i] = distance_homeostatic_dual
                self.closest_clusters[i] = closest_index

            elif self.closest_clusters[i] == closest_index:
                num_pathological_updates += 1
                # If updated cluster was previously closest cluster and moved further away, we must recompute.
                # This part messes up the otherwise O(kd) time complexity of the function, but should not
                # happen that much practice (expected number of neighbors depends on d according to Poisson
                # Voronoi tesselation theory).
                closest_distance_homeostatic: float = math.inf
                closest_cluster: int = -1
                for j in range(self.n_dims):
                    if i == j: continue
                    
                    _distance: float = self._compute_distance(
                        self.cluster_centers + i * self.n_states,
                        self.cluster_centers + j * self.n_states
                    )
                    
                    _distance_homeostatic: float = _distance + self.balancing_strength * \
                        (self.n_points[j] - self.n_points[i]) if self.homeostasis else _distance
                    
                    if _distance_homeostatic < 0: _distance_homeostatic = 0.0
                    if _distance_homeostatic < closest_distance_homeostatic:
                        closest_distance_homeostatic = _distance_homeostatic
                        closest_cluster = j

                self.closest_distances[i] = closest_distance_homeostatic
                self.closest_clusters[i] = closest_cluster

            self.closest_distances[i] = updated_closest_distance_homestatic
            self.closest_clusters[i] =  updated_closest_cluster

        # Set embedding
        if repr is not None:
            repr = torch.zeros(self.n_dims)
            repr[closest_index] = 1.0

        return repr
    

    def reset(self) -> None:
        # Resets the state of the embedding.
        self.reward = 0.0
        self.distances = np.zeros((self.n_dims + 1) ** 2)
        self.cluster_centers = np.zeros(self.n_dims * self.n_states)
        self.closest_distances = np.zeros(self.n_dims)
        self.closest_clusters = np.zeros(self.n_dims)
        self.n_points = np.ones(self.n_dims)


    # --- private implemented methods ---

    def _compute_distance(self, array_1: np.ndarray, array_2: np.ndarray) -> float:
        # Computes Euclidean distance between two np.ndarray objects.
        squared_diff: np.ndarray = (array_1 - array_2) ** 2
        return math.sqrt(np.sum(squared_diff))
    
    def _find_closest(self, index: int) -> Tuple(float, int):
        # Finds the closest cluster to a given point (@index)
        closest_distance: float = math.inf
        closest_distance_index: int = -1
        mean: float = 0.0 # only used if homeostasis = 1

        if self.homeostasis:
            for i in range(len(self.n_dims)):
                mean += self.n_points[i]
            mean /= len(self.n_dims)
    
        for i in range(len(self.n_dims)):
            if i == index: continue
            distance: float = self.distances[index * (self.n_dims + 1) + i]
            if self.homeostasis: 
                distance -= self.balancing_strength * (mean - self.n_points[i])
            if distance < closest_distance:
                closest_distance = distance
                closest_distance_index = i

        return closest_distance, closest_distance_index
    