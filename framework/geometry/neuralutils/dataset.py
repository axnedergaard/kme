import torch.utils.data as D
from torch import Tensor
import numpy as np
import random
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PositiveDataset(D.IterableDataset):
    
    def __init__(self, ambient_dim: int):
        super(PositiveDataset, self).__init__()
        self.samples = []
        self.ambient_dim = ambient_dim

    
    def __iter__(self):
        while True:
            if len(self.samples) < 2:
                base_point = torch.FloatTensor(self.ambient_dim).uniform_(-1.0, 1.0)
                perturbation = torch.FloatTensor(self.ambient_dim).uniform_(-0.01, 0.01)
                yield torch.stack([base_point, base_point + perturbation])
            else:
                idx = random.randrange(len(self.samples) - 1)
                yield torch.stack([self.samples[idx], self.samples[idx + 1]])

    
    def add(self, data: Tensor):
        if data is None: return
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, device=device)
        if len(data.shape) == 1:
            self.samples.append(data)
        elif len(data.shape) == 2:
            self.samples.extend(data)


class RandomDataset(D.IterableDataset):
    
    def __init__(self, ambient_dim: int):
        super(RandomDataset, self).__init__()
        self.ambient_dim = ambient_dim


    def __iter__(self):
        while True:
            r1 = torch.FloatTensor(self.ambient_dim).uniform_(-1.0, 1.0)
            r2 = torch.FloatTensor(self.ambient_dim).uniform_(-1.0, 1.0)
            yield torch.stack([r1, r2])

