import torch

class Density():
    """
    A base class representing a loose abstraction of a density function in an ambient Euclidean space.
    This class provides a framework for building specific density estimators, allowing for various implementations
    that can extend it. The methods provided serve as a common interface for density estimation, including
    functionality to compute the probability density function (PDF), sample from the density, generate random walks,
    compute entropy, and learn the density from given states.
    """

    def __init__(self, ambient_dim: int) -> None:
        # dimension of the ambient Euclidean space
        assert ambient_dim > 0, "Ambient dimension must be positive."
        self.ambient_dim = ambient_dim
    
    def pdf(self, x: torch.Tensor) -> float:
        """
        Compute the probability density function (PDF) at a given point.
        Args: x (torch.Tensor): Point at which to evaluate the PDF.
        Returns: float: Value of the PDF at the given point.
        """
        raise NotImplementedError()

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Generate n samples from the density.
        Args: n_samples (int): Number of samples to generate.
        Returns: torch.Tensor: Samples from the density, of shape `(n_samples, ambient_dim)`.
        """
        raise NotImplementedError()
    
    def rw(self, n_samples: int, x: torch.Tensor = None) -> torch.Tensor:
        """
        Generate samples from n steps of a random walk in the density.
        Args:
            n_samples (int): Number of samples to generate in the walk.
            x (torch.Tensor, optional): Initial state of the walk.
        Returns: torch.Tensor: Random walk samples, of shape `(n_samples, ambient_dim)`.
        """
        raise NotImplementedError()
    
    def entropy(self) -> float:
        """
        Compute the entropy of the density.
        Returns: float: Entropy value.
        """
        raise NotImplementedError()

    def learn(self, states: torch.Tensor) -> None:
        """
        One iteration of learning underlying density function from given states.
        Args: x (torch.Tensor): States from which to learn the density, of shape `(B, ambient_dim)`.
        """
        raise NotImplementedError()
