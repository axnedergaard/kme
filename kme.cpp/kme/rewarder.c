#include "../src/rewarder/kmeans_rewarder.h"

#include <iostream>

using namespace std;

void ArrayToEigenVector(double *array, VectorXd *vector) {
  for (int i = 0; i < vector->size(); i++) {
    (*vector)(i) = array[i];
  }
}

void EigenVectorToArray(VectorXd *vector, double *array) {
  for (int i = 0; i < vector->size(); i++) {
    array[i] = (*vector)(i);
  }
}

VectorXd CreateEigenVectorFromArray(double *array, int n) {
  VectorXd vector(n);
  ArrayToEigenVector(array, &vector);
  return vector;
}

extern "C" {
  KMeansRewarder *rewarder_make(int k, char *function_type, int n_states, double learning_rate, double balancing_strength, double power_fn_exponent) {
    EntropicFunctionType function_type_enum;
    if (strcmp(function_type, "log") == 0) {
      function_type_enum = EntropicFunctionType::LOG;
    } else if (strcmp(function_type, "entropy") == 0) {
      function_type_enum = EntropicFunctionType::ENTROPY;
    } else if (strcmp(function_type, "exponential") == 0) {
      function_type_enum = EntropicFunctionType::EXPONENTIAL;
    } else if (strcmp(function_type, "power") == 0) {
      function_type_enum = EntropicFunctionType::POWER;
    } else if (strcmp(function_type, "identity") == 0) {
      function_type_enum = EntropicFunctionType::IDENTITY;
    } else {
      std::cout << "Error: Entropic function type " << function_type << " not found." << std::endl;
      return NULL;
    }
    return new KMeansRewarder(k, function_type_enum, n_states, learning_rate, balancing_strength, 1e-9, true, power_fn_exponent);
  }

  void rewarder_destroy(KMeansRewarder *rewarder) {
    free(rewarder);
  }

  void rewarder_reset(KMeansRewarder *rewarder) {
    rewarder->Reset();
  }

  double rewarder_infer(KMeansRewarder *rewarder, int n_actions, int n_states, double *state, bool learn) {
    VectorXd state_eigen(n_states);
    ArrayToEigenVector(state, &state_eigen);
    return rewarder->Infer(&state_eigen, NULL, NULL, learn);
  }

  int rewarder_pathological_updates(KMeansRewarder *rewarder) {
    return rewarder->encoder->pathological_updates;
  }
}
