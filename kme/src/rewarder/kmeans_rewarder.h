#pragma once

#include "rewarder.h"
#include "../encoder/kmeans_encoder.h"

enum EntropicFunctionType {
  LOG,
  ENTROPY,
  EXPONENTIAL,
  POWER,
  IDENTITY,
};

class KMeansRewarder : public Rewarder {
  private:
    int k;
    double eps;
    bool differential;
    double power_fn_exponent;
    EntropicFunctionType function_type;
    double EntropicFunction(double x);
  public:
    KMeansEncoder *encoder;
    KMeansRewarder(int k, EntropicFunctionType function_type, int n_states, double learning_rate, double balancing_strength, double eps, bool differential, double power_fn_exponent);
    double EstimateEntropy(KMeansEncoder *_encoder);
    double Infer(VectorXd *next_state, VectorXd *action, VectorXd *state, bool learn);
    void Reset();
};
