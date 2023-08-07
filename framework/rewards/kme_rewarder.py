from rewarder import Rewarder
from density import OnlineKMeansEstimator
from typing import Callable, Union, Optional
from torch import Tensor, FloatTensor, LongTensor
from enum import Enum
import numpy as np
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32


class EntropicFunctionType(Enum):
    LOG = "log"
    ENTROPY = "entropy"
    EXPONENTIAL = "exponential"
    POWER = "power"
    IDENTITY = "identity"


class KMERewarder(Rewarder):

    def __init__(
        self,
        # KMEANS ESTIMATOR
        K: int,
        dim_states: int,
        learning_rate: float,
        balancing_strength: float,
        distance_func: Callable = None,
        origin: Union[Tensor, np.ndarray] = None,
        init_method: str = 'uniform',
        homeostatis: bool = True,
        # KME REWARDER HYPERPARAMS
        entropic_func: str = 'exponential',
        power_fn_exponent: float = 0.5,
        differential: bool = True,
        eps: Optional[float] = 1e-9
    ) -> None:
        super().__init__()

        assert power_fn_exponent > 0, "power_fn_exponent must be greater than 0"
        assert 0 < eps < 1, "eps must be in the range (0, 1)"
        assert EntropicFunctionType(entropic_func), "Unsupported entropic function type"

        self.K = K
        self.differential: bool = differential
        self.fn_type: EntropicFunctionType = EntropicFunctionType(entropic_func)
        self.power_fn_exponent: Tensor = torch.tensor(power_fn_exponent, device=device)
        self.eps: Tensor = torch.tensor(eps, device=device)

        # underlying kmeans density estimator
        self.kmeans = OnlineKMeansEstimator(
            K, dim_states, learning_rate, balancing_strength,
            distance_func, origin, init_method, homeostatis
        )
        
        m = self._pairwise_distances() # (K,K) O(K^2)
        # closest distances and respective idx to all centroids
        self.closest_distances = torch.min(m, dim=1).values # (K,)
        self.closest_idx = torch.argmin(m, dim=1) # (K,)
        self.entropy_buff = 0.0 # store previous entropy


    def infer(self, states: Tensor, learn: bool = True, sequential: bool = True) -> FloatTensor:
        # 1) Update kmeans with all states and get updated centroids idx
        states = self._port_to_tensor(states) # (B, dim_states)
        updated_idx = self.kmeans(states, learn=learn) # (B,)
        # 2) Update distances and compute reward (seq or batch)
        # Usage notice: if B > K, then batch update is more efficient
        # Sidenote: batch infer will only produce one reward for B states
        if sequential: return self._infer_seq(updated_idx, learn) # (B,)
        return self._infer_batch(learn) # (1,)


    def _infer_seq(self, updated_idx: Tensor, learn: bool = True) -> FloatTensor:
        # Extract a reward for each state in the batch sequentially
        # Leverages sparse kmeans update to update distance in O(K)
        # Not deterministic (regardless of learn) as states are shuffled in kmeans
        tmp_d, tmp_idx = self.closest_distances, self.closest_idx
        rewards = torch.zeros(updated_idx.size(0))
        for i, idx in enumerate(updated_idx):
            self._update_distances(idx) # adjust state sequentially
            rewards[i] = self._compute_reward(learn)
        if not learn: self.closest_distances, self.closest_idx = tmp_d, tmp_idx
        return rewards.view(-1)
    

    def _infer_batch(self, learn: bool = True) -> FloatTensor:
        # Extract a single reward from the whole batch of states to kmeans
        # Computes pairwise distances between all centroids from scratch in O(K^2)
        # This is deterministic if learn=False; as we only depend on final state
        m = self._pairwise_distances() # (K,K)
        if learn: self.closest_distances = torch.min(m, dim=1).values
        if learn: self.closest_idx = torch.argmin(m, dim=1)
        return self._compute_reward(learn).unsqueeze(0)


    def _update_distances(self, updated_idx: LongTensor) -> None:
        # Time complexity: O(K * dim_states) (x B)
        # 1) Compute distances from the updated centroid to all others
        centroid = self.kmeans.centroids[updated_idx] # (dim_states,)
        dist_updated = torch.norm(self.kmeans.centroids - centroid, dim=1, p=2) # (K,)
        dist_updated[updated_idx] = float('inf') # exclude updated centroid itself
        # 2) Update closest distances and idx of updated centroid
        min_val, min_idx = torch.min(dist_updated, dim=0)
        self.closest_distances[updated_idx] = min_val.item()
        self.closest_idx[updated_idx] = min_idx.item()
        # 3) Check if the update affected any other centroids
        mask = dist_updated < self.closest_distances
        self.closest_distances[mask] = dist_updated[mask]
        self.closest_idx[mask] = updated_idx
    

    def _pairwise_distances(self, diag: float = float('inf')) -> Tensor:
        # Time-complexity: O(K^2 * dim_states)
        m = torch.zeros(self.K, self.K, device=device)
        x = self.kmeans.centroids.unsqueeze(0)
        y = self.kmeans.centroids.unsqueeze(1)
        m = torch.norm(x - y, dim=2, p=2)
        m.fill_diagonal_(diag)
        return m
    

    def _compute_reward(self, learn: bool) -> Tensor:
        r = self._entropy() # entropy of the current kmeans state
        reward = r - self.entropy_buff if self.differential else r
        if self.differential and learn: self.entropy_buff = r
        return reward


    def _entropy(self, distances: Optional[Tensor] = None) -> Tensor:
        ds = self.closest_distances if distances is None else distances
        entropies = self._entropic_function(ds)
        return torch.sum(entropies)


    def _entropic_function(self, x: Tensor) -> Tensor:
        if self.fn_type == EntropicFunctionType.LOG:
            return torch.log(x + self.eps)
        elif self.fn_type == EntropicFunctionType.ENTROPY:
            return -x * torch.log(x + self.eps)
        elif self.fn_type == EntropicFunctionType.EXPONENTIAL:
            return -torch.exp(-x)
        elif self.fn_type == EntropicFunctionType.POWER:
            return torch.pow(x, self.power_fn_exponent)
        elif self.fn_type == EntropicFunctionType.IDENTITY:
            return x
        else:
            raise ValueError("Entropic function type not found.")


    def _port_to_tensor(self, input: Union[np.ndarray, Tensor]) -> Tensor:
        if isinstance(input, np.ndarray):
            return torch.tensor(input, device=device, dtype=dtype)
        elif isinstance(input, torch.Tensor):
            return input.to(device=device, dtype=dtype)
        else:
            raise ValueError("Unsupported input type. Expected numpy.ndarray \
                    or torch.Tensor, got: {}".format(type(input)))
        


if __name__ == '__main__':
    r = KMERewarder(3, 3, 0.1, 3)
    s = torch.tensor([[1,2,3], [4,5,6], [7,8,9], [10,11,12]], dtype=dtype)
    
    print("sequential tests")
    print(r.infer(s, sequential=True, learn=False))
    print(r.infer(s, sequential=True, learn=False))

    print("batch tests")
    print(r.infer(s, sequential=False))
    print(r.infer(s, sequential=False))
