from density import Density
from torch import Tensor, LongTensor
from typing import Union
import numpy as np
import torch
import copy

class OnlineEstimator(Density):

    def __init__(self, ambient_dim: int, device, dtype) -> None:
        super().__init__(ambient_dim)
        self.device = device
        self.dtype = dtype

    def update(self, states: Tensor) -> LongTensor:
        raise NotImplementedError()
    
    def clone(self) -> Density:
        return copy.deepcopy(self)
    
    def sim_update(self, states: Tensor) -> LongTensor:
        # simulates the update on a clone of estimator
        estimator: Density = self.clone()
        return estimator.update(states)
    
    def _port_to_tensor(self, input: Union[np.ndarray, Tensor]) -> Tensor:
        if isinstance(input, np.ndarray):
            return torch.tensor(input, dtype=self.dtype, device=self.device)
        elif isinstance(input, torch.Tensor):
            return input.to(dtype=self.dtype, device=self.device)
        else:
            raise ValueError("Unsupported input type. Expected numpy.ndarray \
                    or torch.Tensor, got: {}".format(type(input)))
