from rewarder.kmeans_rewarder import KMeansRewarder, EntropicFunctionType
from encoder.encoder_factory import EncoderFactory


def main():
    n_states = 10
    k = 5
    learning_rate = 0.1
    balancing_strength = 0.1
    eps = 1e-9
    differential = True
    power_fn_exponent = 0.5
    entropic_function_type = EntropicFunctionType.LOG

    # Create a KMeansRewarder
    kmeans_rewarder = KMeansRewarder(
        k,
        entropic_function_type,
        n_states,
        learning_rate,
        balancing_strength,
        eps,
        differential,
        power_fn_exponent,
    )

    # Create a KMeansEncoder
    kmeans_encoder = EncoderFactory.create_encoder(
        "kmeans", n_states, k, learning_rate, balancing_strength
    )

    # Example usage
    next_state = [0.5, 0.1, 0.3, 0.6, 0.2, 0.7, 0.9, 0.8, 0.4, 0.0]
    state = [0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7, 0.9, 0.0]
    action = None
    learn = True

    reward = kmeans_rewarder.infer(next_state, action, state, learn)
    print("Reward:", reward)

    encoded_state = kmeans_encoder.embed(state)
    print("Encoded state:", encoded_state)


if __name__ == "__main__":
    main()
