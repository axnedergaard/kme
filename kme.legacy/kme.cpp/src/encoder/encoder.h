#pragma once

#include <eigen3/Eigen/Dense>

using namespace Eigen;

class Encoder {
  public:
    double reward;
    int n_states;
    int n_dims;
    Encoder(int n_states, int n_dims) : n_states(n_states), n_dims(n_dims) {};
    virtual void Embed(VectorXd state, VectorXd *repr) = 0;
    virtual void Reset() {};
};
