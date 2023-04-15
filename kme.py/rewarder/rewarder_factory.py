from .kmeans_rewarder import KMeansRewarder, EntropicFunctionType


class RewarderFactory:
    @staticmethod
    def create(rewarder_type, **kwargs):
        if rewarder_type == "kmeans":
            return KMeansRewarder(
                k=kwargs.get("k", 2),
                function_type=kwargs.get("function_type", EntropicFunctionType.LOG),
                n_states=kwargs.get("n_states", 1),
                learning_rate=kwargs.get("learning_rate", 0.1),
                balancing_strength=kwargs.get("balancing_strength", 1.0),
                eps=kwargs.get("eps", 1e-9),
                differential=kwargs.get("differential", True),
                power_fn_exponent=kwargs.get("power_fn_exponent", 0.5),
            )
        else:
            raise ValueError(f"Unknown rewarder type: {rewarder_type}")
