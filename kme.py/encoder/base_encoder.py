import abc


class BaseEncoder(abc.ABC):
    def __init__(self, n_states, n_components):
        self.n_states = n_states
        self.n_components = n_components

    @abc.abstractmethod
    def embed(self, state, learn=True):
        """
        Embeds the input state into a lower-dimensional representation.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """
        Resets the internal state of the encoder.
        """
        raise NotImplementedError
