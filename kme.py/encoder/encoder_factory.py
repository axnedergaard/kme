from .kmeans_encoder import KMeansEncoder


class EncoderFactory:
    @staticmethod
    def create_encoder(
        encoder_type,
        n_states,
        n_components,
        learning_rate=None,
        balancing_strength=None,
        differential=None,
    ):
        if encoder_type == "kmeans":
            return KMeansEncoder(
                n_states, n_components, learning_rate, balancing_strength, differential
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
