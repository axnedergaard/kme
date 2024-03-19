#include "kmeans_rewarder.h"
#include "math.h"

double KMeansRewarder::EntropicFunction(double x) {
  if (function_type == EntropicFunctionType::LOG) {
    return log(x + eps);
  } else if (function_type == EntropicFunctionType::ENTROPY) {
    return -x * log(x + eps);
  } else if (function_type == EntropicFunctionType::EXPONENTIAL) {
    return -exp(-x);
  } else if (function_type == EntropicFunctionType::POWER) {
    return pow(x, power_fn_exponent);
  } else if (function_type == EntropicFunctionType::IDENTITY) {
    return x;
  } else {
    std::cout << "Warning: Entropic function type not found." << std::endl;
    return 0.0;
  }
}

double KMeansRewarder::EstimateEntropy(KMeansEncoder *_encoder) {
  double entropy = 0.0;
  for (int i = 0; i < k; i++) {
    entropy += EntropicFunction(_encoder->closest_distances[i]);
  }
  return entropy;
}

KMeansRewarder::KMeansRewarder(int k, EntropicFunctionType function_type, int n_states, double learning_rate, double balancing_strength, double eps = 1e-9, bool differential = true, double power_fn_exponent = 0.5) : Rewarder(true, 0, n_states), k(k), eps(eps), differential(differential), power_fn_exponent(power_fn_exponent), function_type(function_type) {
  encoder = new KMeansEncoder(n_states, k, learning_rate, balancing_strength, true);
}

double KMeansRewarder::Infer(VectorXd *next_state, VectorXd *action, VectorXd *state, bool learn) {
  VectorXd encoded_next_state = *next_state;
  double entropy_before, entropy_after;
  if (differential) {
    entropy_before = EstimateEntropy(encoder);
  }
  if (learn) {
    encoder->Embed(encoded_next_state, NULL);
    entropy_after = EstimateEntropy(encoder);
  } else {
    KMeansEncoder tmp_encoder = KMeansEncoder(*encoder);
    tmp_encoder.Embed(encoded_next_state, NULL);
    entropy_after = EstimateEntropy(&tmp_encoder);
  }
  if (differential) {
    double entropy_change = entropy_after - entropy_before;
    return entropy_change;
  } else {
    return entropy_after;
  }
}

void KMeansRewarder::Reset() {
  encoder->Reset();
}
