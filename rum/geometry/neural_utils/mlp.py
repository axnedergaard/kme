import torch

class MLP(torch.nn.Module):

    def __init__(
        self, 
        input_dim: int, 
        output_dim: int,
        hidden_dims: list[int], 
        activation: torch.nn.Module
    ) -> None:
        super(MLP, self).__init__()
        assert len(hidden_dims) >= 1

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation

        # model architecture definition
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dims[0], bias=True),
            torch.nn.BatchNorm1d(hidden_dims[0]),
            self.activation,
            *[torch.nn.Sequential(
                torch.nn.Linear(dim, hidden_dims[idx + 1], bias=True),
                torch.nn.BatchNorm1d(hidden_dims[idx + 1]),
                self.activation
            ) for idx, dim in enumerate(hidden_dims[:-1])],
            torch.nn.Linear(hidden_dims[-1], output_dim)
        )


    def forward(self, x: torch.Tensor, phi: torch.nn.Module = None) -> torch.Tensor:
        logits = self.model(x)
        f = self.activation if phi is None else phi
        return f(logits)
