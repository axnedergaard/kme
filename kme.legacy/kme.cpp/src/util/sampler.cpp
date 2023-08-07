#include "sampler.h"

#include <chrono>
#include <boost/math/distributions/normal.hpp>

Sampler::Sampler(double seed) {
  if (seed == 0) {
    seed = std::chrono::system_clock::now().time_since_epoch().count();
  }
  rng = boost::mt19937(seed);
}

void Sampler::SampleNormalMatrix(MatrixXd *matrix, MatrixXd *means, double variance) {
  for (int i = 0; i < matrix->rows(); i++) {
    for (int j = 0; j < matrix->cols(); j++) {
      double mean = (means != NULL) ? (*means)(i, j) : 0.0;
      (*matrix)(i, j) = SampleNormal(mean, variance);
    }
  }
}

double Sampler::PdfNormal(double mean, double std, double x) {
  boost::math::normal_distribution<double> dist(mean, std);
  return boost::math::pdf<double>(dist, x);
}

double Sampler::SampleNormal(double mean, double std) {
  boost::normal_distribution<double> dist(mean, std);
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<double>> sample(rng, dist);
  return sample();
}

double Sampler::SampleUniform(double min, double max) {
  boost::uniform_real<double> dist(min, max);
  boost::variate_generator<boost::mt19937&, boost::uniform_real<double>> sample(rng, dist);
  return sample();
}

int Sampler::SampleUniformInteger(int min, int max) {
  boost::uniform_int<int> dist(min, max);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<int>> sample(rng, dist);
  return sample();
}
