#pragma once

#include <eigen3/Eigen/Dense>

#include <boost/random.hpp>

using namespace Eigen;

class Sampler {
  public:
    boost::mt19937 rng; //(std::chrono::system_clock::now().time_since_epoch().count());
    Sampler(double seed = 0);
    void SampleNormalMatrix(MatrixXd *matrix, MatrixXd *means, double variance);
    double PdfNormal(double mean, double std, double x);
    double SampleNormal(double mean, double std);
    double SampleUniform(double min, double max);
    int SampleUniformInteger(int min, int max);
};

