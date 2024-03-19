#pragma once

#include "../util/neural_network.h"

#include <eigen3/Eigen/Dense>

using namespace Eigen;

class Rewarder {
  public:
    bool simple; // Dependency only on current state.
    int n_actions;
    int n_states;
    Rewarder(bool simple, int n_actions, int n_states) : simple(simple), n_actions(n_actions), n_states(n_states) {};
    virtual ~Rewarder() {};
    virtual void Reset() = 0;
    virtual double Infer(VectorXd *next_state, VectorXd *action, VectorXd *state, bool learn) = 0;
};
