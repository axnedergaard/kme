from abc import ABC, abstractmethod
import numpy as np

class BaseRewarder(ABC):

    def __init__(self, simple: bool, n_actions: int, n_states: int):
        # public attributes
        self.simple = simple
        self.n_actions = n_actions
        self.n_states = n_states

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def infer(self, next_state: np.ndarray, action: np.ndarray, state: np.ndarray, learn: bool):
        raise NotImplementedError()
