from torch import Tensor, FloatTensor
import torch

class Geometry():
    """
    A base class representing a geometric structure on a space.
    This class provides a generic interface for computing geometric properties such as distance and interpolation
    and can be extended by specific geometric models.
    """
   
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def __call__(self, x: Tensor, y: Tensor) -> FloatTensor:
        return self.distance_function(x, y)

    def distance_function(self, x: Tensor, y: Tensor) -> FloatTensor:
        """
        Compute the distance between states x and y in the geometric space.
        Args:
            x (torch.Tensor): First Tensor. (1, dim) or (B1, dim)
            y (torch.Tensor): Second Tensor. (B2, dim)
        Returns:
            FloatTensor: (B,) Pairwise distance if (1,dim):(B,dim) or (B,dim):(B,dim)
            FloatTensor: (B1,B2) Matrix of pairwise distances if (B1,dim):(B2,dim)
        """
        raise NotImplementedError()

    def interpolate_function(self, x: Tensor, y: Tensor, alpha: float) -> Tensor:
        """
        Interpolate between states x and y, using a specified weight.
        Args:
            x (torch.Tensor): Point from which to start interpolation.
            y (torch.Tensor): Point towards which x is drifting.
            alpha (float): Interpolation weight of y. Typically in the range [0, 1].
        Returns: torch.Tensor: Interpolated state between x and y.
        """
        raise NotImplementedError()
    