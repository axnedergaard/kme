from typing import Callable, Union
from geometry import Geometry
from .neuralutils.mlp import MLP
from .neuralutils.dataset import *
import torch.utils.data as D
from itertools import islice
from torch import Tensor, FloatTensor
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

LEARNING_RATE = 0.0001
BATCH_SIZE = 128

BATCHES_PER_LEARN = 2
NEGATIVE_SAMPLE_SCALING = 1.0
NEGATIVE_MARGIN = 1.0


class NeuralDistance(Geometry):
    
    def __init__(
        self, 
        ambient_dim: int,
        hidden_dims: list[int],
        embedding_dim: int,
        activation: torch.nn.Module = torch.nn.ReLU()
    ) -> None:
        super().__init__(ambient_dim)

        self.network = MLP(
            input_dim=ambient_dim,
            output_dim=embedding_dim,
            hidden_dims=hidden_dims,
            activation=activation
        ).to(device)

        # self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        # self.optimizer = torch.optim.SGD(self.network.parameters(), lr=LEARNING_RATE)
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)

        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=LEARNING_RATE)
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.1)

        self.positive_batcher = D.DataLoader(PositiveDataset(ambient_dim), batch_size=BATCH_SIZE)
        self.negative_batcher = D.DataLoader(RandomDataset(ambient_dim), batch_size=BATCH_SIZE)
        self.loader = zip(self.positive_batcher, self.negative_batcher)

    
    def __call__(self, x: Tensor, y: Tensor) -> FloatTensor:
        self.network.eval()
        with torch.no_grad():
            return self.distance(x, y)


    def distance(self, x: Tensor, y: Tensor, d: Callable = None) -> FloatTensor:
        x, y = self._port_to_tensor(x), self._port_to_tensor(y)
        
        assert x.dim() in [1, 2] and y.dim() in [1, 2] # (ambient_dim,) or (B, ambient_dim)
        x, y = x.unsqueeze(0) if x.dim() == 1 else x, y.unsqueeze(0) if y.dim() == 1 else y # (B, ambient_dim)
        
        if x.shape == y.shape: # (B, ambient_dim) vs (B, ambient_dim)
            concatenated = torch.cat((x, y), dim=0) # (2B, ambient_dim)
            embeddings = self.network(concatenated) # (2B, embedding_dim)
            embedded_x, embedded_y = torch.chunk(embeddings, 2, dim=0) # (B, embedding_dim)
            d = self.euclidean if d is None else d
        
        else: # kmeans optim: (1, ambient_dim) vs (B, ambient_dim)
            concatenated = torch.cat((x, y), dim=0) # (B+1, ambient_dim)
            embeddings = self.network(concatenated) # (B+1, embedding_dim)
            embedded_x, embedded_y = embeddings[:1], embeddings[1:] # (1, embedding_dim), (B, embedding_dim)
            d = self.euclidean if d is None else d

        return d(embedded_x, embedded_y) # (B,)


    def learn(self, states: Tensor = None) -> FloatTensor:
        self.positive_batcher.dataset.add(states)
        self.network.train()
        total_loss = 0.0

        for positive_batch, negative_batch in islice(self.loader, BATCHES_PER_LEARN):
            self.optimizer.zero_grad()

            positive_batch = positive_batch.to(device)
            pos_difference = self.distance(positive_batch[:, 0, :], positive_batch[:, 1, :])
            positive_loss = torch.sum(pos_difference)

            negative_batch = negative_batch.to(device)
            neg_difference = self.distance(negative_batch[:, 0, :], negative_batch[:, 1, :])
            negative_loss = - NEGATIVE_SAMPLE_SCALING * torch.sum(torch.relu(neg_difference - NEGATIVE_MARGIN))

            loss = positive_loss + negative_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        self.scheduler.step()

        return total_loss
    
    
    def euclidean(self, x: Tensor, y: Tensor) -> FloatTensor:
        assert x.dim() == y.dim() == 2 # (B, ambient_dim)
        return torch.norm(x - y, p=2, dim=1) # (B,)

    def _port_to_tensor(self, input: Union[np.ndarray, Tensor]) -> Tensor:
        if isinstance(input, np.ndarray):
            return torch.tensor(input, device=device, dtype=dtype)
        elif isinstance(input, torch.Tensor):
            return input.to(device=device, dtype=dtype)
        else:
            raise ValueError("Unsupported input type. Expected numpy.ndarray \
                    or torch.Tensor, got: {}".format(type(input)))


if __name__ == '__main__':
    d = NeuralDistance(3, [4, 8, 16], 32)
    x = torch.tensor([1, 2, 3], dtype=torch.float)
    y = torch.tensor([4, 5, 6], dtype=torch.float)

    # test computing a distance
    print('neural dist', d(x, y))
    print('euclidean dist', d.euclidean(x.unsqueeze(0), y.unsqueeze(0)))
    d.learn()

    # test learning for one iteration
    print('neural dist', d(x, y))
