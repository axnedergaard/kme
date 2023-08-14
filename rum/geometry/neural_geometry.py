from .geometry import Geometry
from .euclidean_geometry import EuclideanGeometry
from .neural_utils import MLP
from .neural_utils.dataset import *
from typing import Callable, Union
import torch.utils.data as D
from itertools import islice
from torch import Tensor, FloatTensor
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import torch

def_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def_dtype = torch.float32


LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.01
BATCH_SIZE = 128

BATCHES_PER_LEARN = 2
NEGATIVE_SAMPLE_SCALING = 1.0
NEGATIVE_MARGIN = 1.0


class NeuralGeometry(Geometry):
    
    def __init__(
        self,
        # ARCHITECTURE MODEL
        dim: int,
        hidden_dims: list[int],
        embedding_dim: int,
        activation: torch.nn.Module = torch.nn.ReLU(),
        # HYPERPARAMETERS
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = 0.01,
        batch_size: int = BATCH_SIZE,
        batches_per_learn: int = BATCHES_PER_LEARN,
        negative_sample_scaling: float = NEGATIVE_SAMPLE_SCALING,
        negative_margin: float = NEGATIVE_MARGIN,
        # TORCH PARAMETERS
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
        # AMBIENT DIM DISTANCE
        d: Callable = None
    ) -> None:
        super().__init__(dim)

        self.device = device
        self.dtype = dtype

        self.network = MLP(
            input_dim=dim,
            output_dim=embedding_dim,
            hidden_dims=hidden_dims,
            activation=activation
        ).to(self.device)

        self.d = EuclideanGeometry(embedding_dim) if d is None else d

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.batches_per_learn = batches_per_learn
        self.negative_sample_scaling = negative_sample_scaling
        self.negative_margin = negative_margin

        # self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        # self.optimizer = torch.optim.SGD(self.network.parameters(), lr=LEARNING_RATE)
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)

        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.1)

        self.positive_batcher = D.DataLoader(PositiveDataset(dim), batch_size=self.batch_size)
        self.negative_batcher = D.DataLoader(RandomDataset(dim), batch_size=self.batch_size)
        self.loader = zip(self.positive_batcher, self.negative_batcher)

    
    def __call__(self, x: Tensor, y: Tensor) -> FloatTensor:
        self.network.eval()
        with torch.no_grad():
            return self.distance(x, y)


    def distance(self, x: Tensor, y: Tensor, d: Callable = None) -> FloatTensor:
        x, y = self._port_to_tensor(x), self._port_to_tensor(y)
        
        assert x.dim() in [1, 2] and y.dim() in [1, 2] # (dim,) or (B, dim)
        x, y = x.unsqueeze(0) if x.dim() == 1 else x, y.unsqueeze(0) if y.dim() == 1 else y # (B, dim)
        
        if x.shape == y.shape: # (B, dim) vs (B, dim)
            concatenated = torch.cat((x, y), dim=0) # (2B, dim)
            embeddings = self.network(concatenated) # (2B, embedding_dim)
            embedded_x, embedded_y = torch.chunk(embeddings, 2, dim=0) # (B, embedding_dim)
        
        else: # kmeans optim: (1, dim) vs (B, dim)
            concatenated = torch.cat((x, y), dim=0) # (B+1, dim)
            embeddings = self.network(concatenated) # (B+1, embedding_dim)
            embedded_x, embedded_y = embeddings[:1], embeddings[1:] # (1, embedding_dim), (B, embedding_dim)

        return self.d(embedded_x, embedded_y) # (B,)


    def learn(self, states: Tensor = None) -> FloatTensor:
        self.positive_batcher.dataset.add(states)
        if len(self.positive_batcher.dataset.samples) < 2: 
            return -1.0
        
        self.network.train()
        total_loss = 0.0

        for positive_batch, negative_batch in islice(self.loader, self.batches_per_learn):
            self.optimizer.zero_grad()

            positive_batch = positive_batch.to(self.device)
            pos_difference = self.distance(positive_batch[:, 0, :], positive_batch[:, 1, :])
            positive_loss = torch.sum(pos_difference)

            negative_batch = negative_batch.to(self.device)
            neg_difference = self.distance(negative_batch[:, 0, :], negative_batch[:, 1, :])
            negative_loss = - self.negative_sample_scaling * torch.sum(torch.relu(neg_difference - self.negative_margin))

            loss = positive_loss + negative_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        self.scheduler.step()

        return total_loss
    
    
    def euclidean(self, x: Tensor, y: Tensor) -> FloatTensor:
        assert x.dim() == y.dim() == 2 # (B, dim)
        return torch.norm(x - y, p=2, dim=1) # (B,)


    def _port_to_tensor(self, input: Union[np.ndarray, Tensor]) -> Tensor:
        if isinstance(input, np.ndarray):
            return torch.tensor(input, device=self.device, dtype=self.dtype)
        elif isinstance(input, torch.Tensor):
            return input.to(device=self.device, dtype=self.dtype)
        else:
            raise ValueError("Unsupported input type. Expected numpy.ndarray \
                    or torch.Tensor, got: {}".format(type(input)))


if __name__ == '__main__':
    d = NeuralGeometry(3, [4, 8, 16], 32)
    x = torch.tensor([1, 2, 3], dtype=torch.float)
    y = torch.tensor([4, 5, 6], dtype=torch.float)

    # test computing a distance
    print('neural dist', d(x, y))
    print('euclidean dist', d.euclidean(x.unsqueeze(0), y.unsqueeze(0)))
    d.learn()

    # test learning for one iteration
    print('neural dist', d(x, y))
