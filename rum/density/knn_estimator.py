import faiss
import torch

from rum.density import Density
from rum.density.entropic_functions import EntropicFunction

from torch import Tensor
from typing import Optional


class KNNDensityEstimator(Density):

    def __init__(self, k, dim, buffer_max_size, entropic_func: EntropicFunction = None, **kwargs):
        self.k: int = k
        self.dim = dim
        self.buffer = torch.zeros((buffer_max_size, dim))
        self.buffer_size = 0
        self.entropic_func = entropic_func if entropic_func is not None \
            else EntropicFunction("log", eps=1)

    # --- Public interface methods ---
        
    def learn(self, states: Tensor) -> None:
        # Update the buffer state with incoming states.
        self.compute_buffer(states, inplace=True)

    def simulate_step(self, state: Tensor) -> Tensor:
        b, bsize = self.compute_buffer(state, inplace=False)
        return self.compute_distances(b[:bsize], b, bsize) # shape: (bsize, bsize)

    # --- knn density estimators methods ---

    def pdf(self, x: Tensor) -> float:
        self.pdf_approx(x)
    
    def pdf_approx(self, x: Tensor) -> float:
        x = x.view(-1, self.dim) # shape: (1, dim)
        b, bsize = self.compute_buffer(x, inplace=False)
        distances = self.compute_distances(x, b, bsize).view(-1) # shape: (bsize,)
        return (1.0 / self.k) * torch.sum(distances)
    
    def information(self, x: Tensor) -> float:
        pdx_approx = self.pdf_approx(x)
        return self.entropic_func(pdx_approx)

    def entropy(self) -> Tensor:
        self.entropy_approx()

    def entropy_approx(self, distances: Optional[Tensor] = None) -> Tensor:
        # TODO This function still has a bug with dimensions.
        assert distances is None or distances.shape == (self.buffer_size, self.buffer_size)
        distances = distances or self.compute_distances(self.buffer[:self.buffer_size], self.buffer, self.buffer_size)
        pdf_approxs = (1.0 / self.k) * torch.sum(distances, dim=1)
        return torch.sum(self.entropic_func(pdf_approxs))
    
    # --- knn private computation methods ---

    def compute_buffer(self, states: Tensor, inplace=True) -> Tensor:
        if states.size(0) > self.buffer.size(0):
            raise ValueError("States size must be less than buffer size.")
        
        if inplace:
            # Use the buffer from current state.
            b = self.buffer
        else:
            # Create a fresh copy of the buffer.
            b = torch.zeros(self.buffer.shape)
            b = self.buffer[0:self.buffer_size]

        # Compute the number of slots available in the buffer.
        num_new_states = states.size(0)
        num_free_slots = self.buffer.size(0) - self.buffer_size
        size_overflow = num_new_states - num_free_slots

        if size_overflow > 0:
            # First pad the buffer with free slots.
            b[self.buffer_size:self.buffer_size+num_free_slots] = states[:num_free_slots]
            # Then flush some states to make room for the new ones.
            # Note: We might drop some incoming states in early stages. (This is okay)
            drop_idx = torch.randint(low=0, high=self.buffer.size(0), size=(size_overflow,))
            b[drop_idx] = states[num_free_slots:]
            return b, self.buffer_size
        
        # If there is enough space in the buffer, just append.
        b[self.buffer_size:self.buffer_size+num_new_states] = states
        return b, self.buffer_size + num_new_states
    
    def compute_distances(self, states: Tensor, buffer: Tensor, bsize: int) -> Tensor:
        # Create a faiss index (L2 distance)
        index = faiss.IndexFlatL2(self.dim)
        index.add(buffer[:bsize].numpy())
        # Search for the k nearest neighbors.
        distances, _ = index.search(states.numpy(), self.k) # shape: (states.size(0), k)
        return torch.tensor(distances)
