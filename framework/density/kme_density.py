from .density import Density
from torchkme import KMeansEncoder
from typing import Optional
from scipy.stats import multivariate_normal, quad
import numpy as np
import torch

class KMeansDensity(Density):
    def __init__(
        self,
        k: int,
        dim_states: int,
        learning_rate: float,
        balancing_strength: float,
        homeostasis: float,
        init_method: str = 'kmeans++'
    ) -> None:
        
        super().__init__(ambient_dim=dim_states)

        # init kme encoder from torchkme module
        self.k_encoder = KMeansEncoder(
            k=k,
            dim_states=dim_states,
            learning_rate=learning_rate,
            balancing_strength=balancing_strength,
            homeostasis=homeostasis,
            init_method=init_method
        )
    

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Generates random samples from the distribution.
        This function randomly selects one of the k-means clusters and samples from the 
        corresponding Gaussian distribution, thus creating samples from the mixture of 
        Gaussians that represent the underlying density.

        :param n_samples: Number of samples to generate.
        :return: Tensor containing the generated samples.
        """
        samples = []
        for _ in range(n_samples):
            idx = np.random.choice(len(self.k_encoder.centroids))
            centroid = self.k_encoder.centroids[idx]
            covariance = np.eye(centroid.shape[0])  # assuming identity covariance matrix
            sample = np.random.multivariate_normal(centroid, covariance)
            samples.append(sample)
        return torch.tensor(samples)

    def pdf(self, x: torch.Tensor) -> float:
        """
        Computes the probability density function (pdf) at the given point x.
        This function assumes that the density is represented by a mixture of 
        Gaussian distributions, each centered at one of the k-means cluster centroids.

        :param x: Input tensor representing the point at which the pdf is evaluated.
        :return: Probability density at the given point.
        """
        probabilities = []
        x_np = x.numpy()
        for centroid in self.k_encoder.centroids:
            covariance = np.eye(centroid.shape[0])  # assuming identity covariance matrix
            probabilities.append(multivariate_normal.pdf(x_np, mean=centroid, cov=covariance))
        return sum(probabilities) / len(probabilities)


    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the cumulative distribution function (cdf) at the given point x.
        This function integrates the pdf across the space up to x, taking into account 
        the mixture of Gaussians based on the clusters found by the k-means algorithm.

        :param x: Input tensor representing the point at which the cdf is evaluated.
        :return: Cumulative probability at the given point.
        """
        cumulative_prob = 0
        for i in range(len(self.k_encoder.centroids)):
            integral, _ = quad(lambda y: self.pdf(y), -np.inf, x[i])
            cumulative_prob += integral
        return cumulative_prob / len(self.k_encoder.centroids)


    def entropy(self) -> float:
        raise NotImplementedError()
    
    def learn(self, next_state: torch.Tensor) -> None:
        raise NotImplementedError() 
    