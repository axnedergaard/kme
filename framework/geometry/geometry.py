from torch import Tensor, FloatTensor
import torch

class Geometry():
    """
    A base class representing a geometric structure in an ambient Euclidean space.
    This class provides a generic interface for computing geometric properties such as distance and interpolation
    and can be extended by specific geometric models.
    """
   
    def __init__(self, ambient_dim: int) -> None:
        # dimension of the ambient Euclidean space
        self.ambient_dim = ambient_dim

    def __call__(self, x: Tensor, y: Tensor) -> FloatTensor:
        with torch.no_grad():
            return self.distance(x, y)

    def distance(self, x: Tensor, y: Tensor) -> FloatTensor:
        """
        Compute the distance between states x and y in the geometric space.
        Args:
            x (torch.Tensor): First state.
            y (torch.Tensor): Second state.
        Returns: FloatTensor: Distance between the states x and y.
        """
        raise NotImplementedError()
    
    def interpolate(self, x: Tensor, y: Tensor, alpha: float) -> Tensor:
        """
        Interpolate between states x and y, using a specified weight.
        Args:
            x (torch.Tensor): First state.
            y (torch.Tensor): Second state.
            alpha (float): Interpolation weight, typically in the range [0, 1].
        Returns: torch.Tensor: Interpolated state between x and y.
        """
        raise NotImplementedError()
    
    def learn(self, states: Tensor = None) -> FloatTensor:
        """
        One iteration of learning underlying geometric representation from given states.
        Args: states (torch.Tensor): States from which to learn the geometric representation.
        Returns: FloatTensor: Loss incurred during learning on that iteration.
        """
        raise NotImplementedError()
