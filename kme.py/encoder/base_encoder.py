from abc import ABC, abstractmethod
from torch import Tensor


class BaseEncoder(ABC):

    def __init__(self, n_states: int, n_dims: int):
        self.n_states: int = n_states
        self.n_dims: int = n_dims
        self.reward: float = 0.0

    @abstractmethod
    def embed(self, state: Tensor, repr: Tensor) -> Tensor:
        """Embeds the input state into a lower-dimensional representation."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Resets the internal state of the encoder."""
        raise NotImplementedError
