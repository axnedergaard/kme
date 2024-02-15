import torch

class Density():
    """
    A base class representing a loose abstraction of a density function on a space on a manifold.
    This class provides a framework for building specific density estimators, allowing for various implementations
    that can extend it. The methods provided serve as a common interface for density estimation, including
    functionality to compute the probability density function (pdf), sample from the density, generate random walks,
    compute entropy and learn the density from given states.
    """

    def __init__(self, dim: int) -> None:
        assert dim > 0, "Dimension must be positive."
        self.dim = dim
    
    def pdf(self, x: torch.Tensor) -> float:
        """
        Compute the probability density function (pdf) at a given point.
        Args: x (torch.Tensor): Point at which to evaluate the PDF.
        Returns: float: Value of the PDF at the given point.
        """
        raise NotImplementedError()

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Generate n samples from the density.
        Args: n_samples (int): Number of samples to generate.
        Returns: torch.Tensor: Samples from the density, of shape `(n_samples, dim)`.
        """

    
    def random_walk(self, n_samples: int, x: torch.Tensor = None) -> torch.Tensor:
        """
        Generate samples from n steps of a random walk in the density.
        Args:
            n_samples (int): Number of samples to generate in the walk.
            x (torch.Tensor, optional): Initial state of the walk.
        Returns: torch.Tensor: Random walk samples, of shape `(n_samples, dim)`.
        """
        raise NotImplementedError()
    
    def entropy(self) -> float:
        """
        Compute the entropy of the density.
        Returns: float: Entropy value.
        """
        raise NotImplementedError()
