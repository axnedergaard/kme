import torch

class Density():

    def __init__(self, ambient_dim: int) -> None:
        # dimension of the ambient Euclidean space
        self.ambient_dim = ambient_dim

    def sample(self, n_samples: int) -> torch.Tensor:
        # torch.Tensor of size (n_samples, ambient_dim)
        raise NotImplementedError()
    
    def pdf(self, x: torch.Tensor) -> float:
        raise NotImplementedError()

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def log_pdf(self, x: torch.Tensor) -> float:
        # for numerical stability, returns log(pdf(x))
        return torch.log(self.pdf(x))
    
    def entropy(self) -> float:
        raise NotImplementedError()

    def learn(self, data: torch.Tensor) -> None:
        # learns the density from data
        raise NotImplementedError()    
