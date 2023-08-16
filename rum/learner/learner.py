import torch
from torch import Tensor


class _Buffer():

    def __init__(self, dim_states: int, buffer_size: int):
        self.B = torch.zeros((buffer_size, dim_states))
        self.dim_state = dim_states
        self.max_size = buffer_size
        self.size = 0

    def append(self, states):
        num_states = states.size(0)
        self.B[self.size:self.size+num_states] = states
        self.size += num_states

    def flush(self):
        self.B = torch.zeros((self.max_size, self.dim_state))
        self.size = 0


class Learner():
    
    def __init__(self, dim_states: int, buffer_size: int):
        self.buffer = _Buffer(dim_states, buffer_size)
        self.dim_states = dim_states

    def learn(self, states: Tensor):
        if not isinstance(states, Tensor) or states.dim() != 2:
            raise ValueError("States must be a 2D tensor.")
        
        num_states = states.size(0)
        while num_states > 0:
            available_space = self.buffer.max_size - self.buffer.size
            if available_space == 0:
                self._learn(self.buffer.B)
                self.buffer.flush()
                continue

            add_states_count = min(available_space, num_states)
            self.buffer.append(states[:add_states_count])
            states = states[add_states_count:]
            num_states -= add_states_count


    def _learn(self, states: Tensor):
        raise NotImplementedError("Learner must implement _learn method")
