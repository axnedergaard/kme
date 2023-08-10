from density import OnlineEstimator
from torch import Tensor, LongTensor, FloatTensor
from typing import Callable, Union
import numpy as np
import torch

def_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def_dtype = torch.float32

class OnlineKMeansEstimator(OnlineEstimator):

    def __init__(
        self,
        K: int,
        dim_states: int,
        learning_rate: float,
        balancing_strength: float,
        distance_func: Callable = None,
        origin: Union[Tensor, np.ndarray] = None,
        init_method: str = 'uniform',
        homeostatis: bool = True,
        device: torch.device = def_device,
        dtype: torch.dtype = def_dtype
    ):
        super().__init__(dim_states, device, dtype)    
        
        assert K > 0, "Number of clusters K must be greater than 0"
        assert 0 < learning_rate <= 1, "Learning rate must be in the range (0, 1]"
        assert balancing_strength >= 0, "Balancing strength must be non-negative"
        assert init_method in ['uniform', 'zeros', 'gaussian'], "Initialization method is not supported"
        
        if origin is not None:
            origin = self._port_to_tensor(origin)
            assert origin.shape == (dim_states,), "Starting point must be a single state (dim_states,)"

        self.device: torch.device = device
        self.dtype: torch.dtype = dtype

        # k-means spec
        self.K: int = K
        self.dim_states: int = dim_states
        self.init_method: str = init_method

        # regarding underlying manifold
        self.distance: Callable = self._euclidean_distance if distance_func is None else distance_func
        self.origin: Tensor = torch.zeros((1, self.dim_states), dtype=self.dtype, device=self.device) if origin is None else origin
        
        # tunable hyperparameters
        self.hp_learning_rate: float = learning_rate
        self.hp_balancing_strength: float = balancing_strength
        self.hp_homeostasis: bool = homeostatis

        # internal k-means state
        self.centroids: Tensor = self._init_centroids() # (K, dim_states) mu_i
        self.cluster_sizes: Tensor = torch.zeros((self.K,), dtype=self.dtype, device=self.device) # (K,) n_i


    # --- public interface methods ---

    def update(self, states: Tensor) -> LongTensor:
        # Updates the internal state of the KMeansEncoder with a new state.
        # according to algorithm (1) in https://arxiv.org/pdf/2205.15623.pdf
        states = self._port_to_tensor(states).requires_grad_(False) # detach any gradients
        assert states.dim() == 2 and states.shape[1] == self.dim_states, "States must be of size (B, dim_states)"
        shuffled_states: Tensor = states[torch.randperm(states.shape[0])]
        # closest_cluster_idx: Tensor = [self._update_single(s) for s in shuffled_states]
        # return torch.stack(closest_cluster_idx) # to flatten list of tensors
        closest_cluster_idx = torch.zeros(shuffled_states.shape[0], dtype=torch.long, device=self.device)
        for idx, state in enumerate(shuffled_states): closest_cluster_idx[idx] = self._update_single(state)
        return closest_cluster_idx
    

    # --- private interface methods ---

    def _update_single(self, state: Tensor) -> LongTensor:
        closest_cluster_idx: Tensor = self._find_closest_cluster(state)
        self._update_centroid(state, closest_cluster_idx)
        return closest_cluster_idx


    def _find_closest_cluster(self, state: Tensor) -> LongTensor:
        # Finds closest cluster center for a given state.
        distances: Tensor = self._distance_to_clusters(state)
        closest_cluster_idx: Tensor = torch.argmin(distances)
        return closest_cluster_idx


    def _update_centroid(self, state: Tensor, closest_cluster_idx: int) -> None:
        # Update centroid of the closest cluster in Euclidean metric space (+)
        # May induce drift towards state outside of the surface of manifold
        adj_lr: float = self.hp_learning_rate / (self.cluster_sizes[closest_cluster_idx] + 1)
        # state_contribution: Tensor = adj_lr * state
        # centroid_contribution: Tensor = (1 - adj_lr) * self.centroids[closest_cluster_idx]
        # self.centroids[closest_cluster_idx] = state_contribution.add(centroid_contribution)
        self.centroids[closest_cluster_idx].mul_(1 - adj_lr) # centroid contribution
        self.centroids[closest_cluster_idx].add_(adj_lr * state) # state contribution
        self.cluster_sizes[closest_cluster_idx] += 1


    def _distance_to_clusters(self, state: Tensor) -> FloatTensor:
        s, cs = state.unsqueeze(0), self.centroids  # s(1, dim_states) cs(K, dim_states) 
        distances: Tensor = self.distance(s, cs).view(-1) # distances(K,)

        if self.hp_homeostasis:
            mean = torch.mean(self.cluster_sizes)
            adj = self.hp_balancing_strength * (self.cluster_sizes - mean)
            distances += adj # .CHECK investigate consequence of (adj < 0)

        return distances


    def _init_centroids(self) -> FloatTensor:
        if self.init_method == 'zeros':
            return self.origin.repeat(self.K, 1)
        elif self.init_method == 'uniform':
            return 2 * torch.rand((self.K, self.dim_states), dtype=self.dtype, device=self.device) - 1
        elif self.init_method == 'gaussian':
            cov = torch.eye(self.dim_states, dtype=self.dtype, device=self.device)
            return torch.distributions.MultivariateNormal(self.origin, cov).sample((self.K,)).clamp(-1, 1)

        
    def _euclidean_distance(self, x: Tensor, ys: Tensor) -> FloatTensor:
        return torch.norm(x - ys, dim=1)
