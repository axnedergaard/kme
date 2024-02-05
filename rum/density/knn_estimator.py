import faiss
import torch

from rum.density import Density
from rum.learner import Learner


class KNNDensityEstimator(Density, Learner):
    def __init__(self, state_dim, k, constant_c=1e-3):
        self.state_dim = state_dim
        self.k = k
        self.constant_c = constant_c
        self.index = None  # To be created with the replay buffer

    def update_replay_buffer(self, replay_buffer_states: torch.Tensor):
        replay_buffer_states_np = replay_buffer_states.cpu().numpy()
        self.index = faiss.IndexFlatL2(self.state_dim)
        self.index = faiss.index_cpu_to_all_gpus(self.index)
        self.index.add(replay_buffer_states_np)

    def entropy(self, states: torch.Tensor) -> torch.Tensor:
        # search k-nearest neighbors in the replay buffer
        distances, _ = self.index.search(states.cpu().numpy(), self.k + 1)
        distances = distances[:, 1:]  # exclude distance to himself
        entropy = torch.log(
            self.constant_c
            + torch.mean(torch.from_numpy(distances).to(states.device), dim=1)
        )
        return entropy
