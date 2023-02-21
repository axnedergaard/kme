#pragma once

#include "sampler.h"

#include <vector>
#include <eigen3/Eigen/Dense>

class NeuralNetwork {
  protected:
    Sampler *sampler;
  public:
    double debug_average_update;
    bool bias;
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::VectorXd> biases; // Only used if biases not part of weight matrix.
    NeuralNetwork(Sampler *sampler = NULL);
    void Reset(double variance);
    void ForwardPass(Eigen::MatrixXd input, Eigen::MatrixXd *output, std::vector<Eigen::MatrixXd> *activations);
    void BackwardPass(Eigen::MatrixXd errors, std::vector<Eigen::MatrixXd> activations, std::vector<Eigen::MatrixXd> *weight_gradients, std::vector<Eigen::VectorXd> *bias_gradients);
    void GradientStep(std::vector<Eigen::MatrixXd> weight_gradients, std::vector<Eigen::VectorXd> bias_gradients, double learning_rate);
};
