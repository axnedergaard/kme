import torch
from itertools import islice
import torch.utils.data as D
import numpy as np
import random


LEARNING_RATE = 0.0001
BATCH_SIZE = 128

BATCHES_PER_LEARN = 8
NEGATIVE_SAMPLE_SCALING = 1


class Distance():

    def __init__(self, ambient_dim: int) -> None:
        # dimension of the ambient Euclidean space
        self.ambient_dim = ambient_dim

    def compute(self, state_x: torch.Tensor, state_y: torch.Tensor) -> float:
        # computes the distance between states x and y
        raise NotImplementedError()
    
    def learn(self, data: torch.Tensor) -> None:
        # learns the distance from data
        raise NotImplementedError()
    

class MLP(torch.nn.Module):

    def __init__(
            self, 
            input_dim: int, 
            hidden_dim: int, 
            num_hidden_layers: int, 
            output_dim: int,
            activation_func: torch.nn.Module = torch.nn.ReLU()
    ) -> None:
        # generic mlp network architecture
        super(MLP, self).__init__()

        # dimension of the hidden layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim

        # model architecture definition
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim, bias=True),
            activation_func,
            *[torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim, bias=True),
                activation_func
            ) for _ in range(num_hidden_layers)],
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(
            self, x: torch.Tensor, 
            activation_head: torch.nn.Module = torch.nn.Tanh()
    ) -> torch.Tensor:
        logits = self.model(x)
        return activation_head(logits)
    

class PositiveDataset(D.IterableDataset):
  def __init__(self, ambient_dim: int):
    super(PositiveDataset, self).__init__()
    self.data = []
    self.ambient_dim = ambient_dim

  def __iter__(self):
    while True:
        if len(self.data) < 2:
            rp = np.random.uniform(-1.0, 1.0, [2, self.ambient_dim]).astype(np.float32)
            yield np.array([rp, rp], dtype=np.float32)
        else:
            index = random.randrange(len(self.data)) - 1
            yield np.array([self.data[index], self.data[index + 1]], dtype=np.float32)

  def add(self, datum):
    self.data += [datum]


class RandomDataset(D.IterableDataset):
  def __init__(self, ambient_dim: int):
    super(RandomDataset, self).__init__()
    self.ambient_dim = ambient_dim

  def __iter__(self):
    while True:
      yield np.random.uniform(-1.0, 1.0, [2, self.ambient_dim]).astype(np.float32)


class NeuralDistance(Distance):
    
    def __init__(self, ambient_dim: int) -> None:
        super().__init__(ambient_dim)
        
        # made up numbers, to be turned into params
        self.network = MLP(
            input_dim=ambient_dim,
            hidden_dim=64,
            num_hidden_layers=2,
            output_dim=3
        )

        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=LEARNING_RATE)
        self.positive_batcher = D.DataLoader(PositiveDataset(ambient_dim), batch_size=BATCH_SIZE)
        self.negative_batcher = D.DataLoader(RandomDataset(ambient_dim), batch_size=BATCH_SIZE)


    def _euclidean_dist(self, x: torch.Tensor, y: torch.Tensor) -> float:
        # computes euclidean distance in the ambiant space
        return torch.norm(x - y, p=2)

    def compute(self, x: torch.Tensor, y: torch.Tensor) -> float:
        embedded_x = self.network(x).detach()
        embedded_y = self.network(y.detach())
        d = self._euclidean_dist # Replace with more general dists
        return d(embedded_x, embedded_y)

    def learn(self, data_points: torch.Tensor = None):

        if data_points is not None:
            for datum in data_points:
                self.positive_batcher.dataset.add(datum)

        loss = 0.0
        for batch in islice(self.positive_batcher, BATCHES_PER_LEARN):
            embedded = self.network(batch)
            difference = embedded[:, 1, :] - embedded[:, 0, :]
            norm = torch.linalg.norm(difference, dim=1)
            loss += torch.sum(norm)
        for batch in islice(self.negative_batcher, BATCHES_PER_LEARN):
            embedded = self.network(batch)
            difference = embedded[:, 1, :] - embedded[:, 0, :]
            norm = torch.linalg.norm(difference, dim=1)
            loss -= NEGATIVE_SAMPLE_SCALING * torch.sum(norm)
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    d = NeuralDistance(3)

    # test computing a distance
    x = torch.tensor([1, 2, 3], dtype=torch.float)
    y = torch.tensor([4, 5, 6], dtype=torch.float)
    print('neural dist', d.compute(x, y))
    print('euclidean dist', d._euclidean_dist(x, y))

    # test learning for one iteration
    d.learn()
    print('neural dist', d.compute(x, y))