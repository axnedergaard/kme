from .geometry import Geometry
from torch import Tensor, FloatTensor
import torch

class EuclideanGeometry(Geometry):

    def __init__(self, dim: int) -> None:
        super().__init__(dim)


    def distance_function(self, x: Tensor, y: Tensor) -> FloatTensor:
        if len(x.shape) != 2 or len(y.shape) != 2:
            raise ValueError("Tensors must be 2D")
        if x.shape[1] != self.dim or y.shape[1] != self.dim:
            raise ValueError("Tensors must lie in ambient space")

        if x.shape == y.shape or x.shape[0] == 1:
            # (B, dim) x (B, dim) -> (B,)
            # (1, dim) x (B, dim) -> (B,)
            d = torch.norm(x - y, p=2, dim=1)
        
        else:
            # (B1, dim) x (B2, dim) -> (B1, B2)
            x_expanded = x[:, None, :]  # shape (B1, 1, embedding_dim)
            y_expanded = y[None, :, :]  # shape (1, B2, embedding_dim)
            d = torch.norm(x_expanded - y_expanded, p=2, dim=-1)
        
        return d # pairwise (B,) or matrix (B1, B2)


    def interpolate(self, x: Tensor, y: Tensor, alpha: float) -> Tensor:
        if x.shape != (self.dim,) or y.shape != (self.dim,):
            raise ValueError("Tensors must lie in ambient space")
        return (1 - alpha) * x + alpha * y


    def learn(self, states: Tensor = None) -> FloatTensor:
        pass # No learning is required.
