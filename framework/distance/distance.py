import torch
from itertools import islice

LEARNING_RATE = 0.0001
BATCH_SIZE = 128

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
        super(MLP).__init__()

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
            activation_head: torch.nn.Module = torch.nn.Tanh
    ) -> torch.Tensor:
        logits = self.model(x)
        return activation_head(logits)
    

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

    def compute(self, x: torch.Tensor, y: torch.Tensor, d) -> float:
        embedded_x = self.network(x).detach()
        embedded_y = self.network(y.detach())
        return d(embedded_x, embedded_y)

    def learn(self):
        raise NotImplementedError()
