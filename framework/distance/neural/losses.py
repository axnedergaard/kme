import torch
from torch import nn
from torch import Tensor


class MSE(nn.Module):
    
        # implements MSE loss function:
        # L(x, y) = 1/n * sum_{i=1}^n (x_i - y_i)^2

        def __init__(self) -> None:
            super(MSE, self).__init__()
    
        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            return torch.mean((x - y)**2)


class MAE(nn.Module):
        
    # implements MAE loss function:
    # L(x, y) = 1/n * sum_{i=1}^n |x_i - y_i|

    def __init__(self) -> None:
        super(MAE, self).__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.mean(torch.abs(x - y))
    

class BCE(nn.Module):
            
    # implements BCE loss function:
    # L(x, y) = 1/n * sum_{i=1}^n y_i * log(x_i) + (1 - y_i) * log(1 - x_i)

    def __init__(self) -> None:
        super(BCE, self).__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.mean(y * torch.log(x) + (1 - y) * torch.log(1 - x))
    

class BCEWithLogits(nn.Module):
                
    # implements BCE loss function:
    # L(x, y) = 1/n * sum_{i=1}^n y_i * log(x_i) + (1 - y_i) * log(1 - x_i)

    def __init__(self) -> None:
        super(BCEWithLogits, self).__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.mean(y * torch.log(torch.sigmoid(x)) + (1 - y) * torch.log(1 - torch.sigmoid(x)))
        

class CrossEntropy(nn.Module):
                        
    # implements CrossEntropy loss function:
    # L(x, y) = 1/n * sum_{i=1}^n y_i * log(x_i) + (1 - y_i) * log(1 - x_i)
    
    def __init__(self) -> None:
        super(CrossEntropy, self).__init__()
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.mean(y * torch.log(torch.softmax(x, dim=1)) + (1 - y) * torch.log(1 - torch.softmax(x, dim=1)))
        

class NLLLoss(nn.Module):
     
    # implements NLLLoss loss function:
    # L(x, y) = 1/n * sum_{i=1}^n y_i * log(x_i) + (1 - y_i) * log(1 - x_i)
        
    def __init__(self) -> None:
        super(NLLLoss, self).__init__()
        
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.mean(y * torch.log(torch.softmax(x, dim=1)) + (1 - y) * torch.log(1 - torch.softmax(x, dim=1)))
    

class KLDivLoss(nn.Module):
             
    # implements KLDivLoss loss function:
    # L(x, y) = 1/n * sum_{i=1}^n y_i * log(x_i) + (1 - y_i) * log(1 - x_i)
            
    def __init__(self) -> None:
        super(KLDivLoss, self).__init__()
            
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.mean(y * torch.log(torch.softmax(x, dim=1)) + (1 - y) * torch.log(1 - torch.softmax(x, dim=1)))
        
