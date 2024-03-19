from inspect import signature, Parameter
import torch


class EntropicFunction:

    _FUNCTIONS = {
        "LOG": lambda x, eps: torch.log(x + eps),
        "ENTROPY": lambda x, eps: -x * torch.log(x + eps),
        "EXPONENTIAL": lambda x: -torch.exp(-x),
        "POWER": lambda x, power: torch.pow(x, power),
        "IDENTITY": lambda x: x
    }

    def __init__(self, func_name, **kwargs):
        func_name = func_name.upper()
        if func_name not in self._FUNCTIONS:
            raise ValueError(f"'{func_name}' is not a supported function.") 
        
        self.func = self._FUNCTIONS[func_name]
        self.kwargs = kwargs

        params = signature(self.func).parameters
        for name, param in list(params.items())[1:]:  # skip parameter 'x'
            if param.default == Parameter.empty and name not in self.kwargs:
                raise ValueError(f"Missing required parameter: {name}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(x, **self.kwargs)
