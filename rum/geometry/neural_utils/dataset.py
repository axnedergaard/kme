import torch.utils.data as D
from torch import Tensor
import numpy as np
import random
import torch


class PositiveDataset(D.IterableDataset):
    
    def __init__(self, dim: int):
        super(PositiveDataset, self).__init__()
        self.samples = []
        self.dim = dim

    
    def __iter__(self):
        while True:
            idx = random.randrange(len(self.samples) - 1)
            yield torch.stack([self.samples[idx], self.samples[idx + 1]])

    
    def add(self, data: Tensor):
        if data is None: return
        if isinstance(data, np.ndarray):
            data = torch.tensor(data)
        if len(data.shape) == 1:
            self.samples.append(data)
        elif len(data.shape) == 2:
            self.samples.extend(data)


class RandomDataset(D.IterableDataset):
    
    def __init__(self, dim: int):
        super(RandomDataset, self).__init__()
        self.dim = dim


    def __iter__(self):
        while True:
            r1 = torch.FloatTensor(self.dim).uniform_(-1.0, 1.0)
            r2 = torch.FloatTensor(self.dim).uniform_(-1.0, 1.0)
            yield torch.stack([r1, r2])

