#include "util.h"

#include "sampler.h"

#include <math.h>

using namespace std;

double EuclideanDistance(double *a, double *b, int size) {
  double distance = 0.0;
  for (int i = 0; i < size; i++) {
    distance += pow(a[i] - b[i], 2);
  }
  distance = sqrt(distance);
  return distance;
}

void RandomColor(Color *c, Sampler* sampler) {
  c->r = sampler->SampleUniform(0, 1);
  c->g = sampler->SampleUniform(0, 1);
  c->b = sampler->SampleUniform(0, 1);
  if (c->r < 0.2 && c->g < 0.2 && c->b < 0.2) RandomColor(c, sampler);
}

string PrintVector(vector<string> vec) {
  string str = "";
  for (int i = 0; i < vec.size(); i++) {
    str += vec[i];
    if (i + 1 < vec.size()) str += ",";
  }
  return str;
}

double DistanceToSimilarity(double distance) {
  //return 1.0 / (1.0 + distance);
  return exp(-distance);
}

double SimilarityToDistance(double similarity) {
  //return 1.0 / similarity - 1.0;
  return -log(similarity);
}

double Sigmoid(double x) {
  return (1.0 / (1 + exp(-x)));
}

double Signd(double x) {
  if (x < 0) return -1.0;
  else return 1.0;
}

double Relu(double x) {
  if (x < 0) return 0.0;
  else return x;
}

double Tanhd(double x) {
  return tanh(x);
}

double TanhDerivative(double x) {
  return 1.0 - tanhf(x) * tanhf(x);
}

double RescaleTanh(double x) {
  return (x + 1.0) / 2.0;
}

double Clip(double x, double min, double max) {
  if (x > max) return max;
  else if (x < min) return min;
  else return x;
}

double ClipUnit(double x) {
  return Clip(x, -1.0, 1.0);
}

double ClipGradients(double x) {
  return Clip(x, -MAX_GRADIENT, MAX_GRADIENT);
}
