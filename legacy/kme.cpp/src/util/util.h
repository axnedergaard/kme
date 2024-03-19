#pragma once

#include "sampler.h"

#include <string>
#include <vector>

#define MAX_GRADIENT 0.1

struct Color {
  float r, g, b;
};

std::string PrintVector(std::vector<std::string> vec);

double EuclideanDistance(double *a, double *b, int size);

void RandomColor(Color *c, Sampler *sampler);

double DistanceToSimilarity(double distance);

double SimilarityToDistance(double similarity);

double MergeAngles(double a1, double a2);

double Sigmoid(double x);

double Signd(double x);

double Relu(double x);

double Tanhd(double x); // Eigen complains if tanh used directly.

double TanhDerivative(double x);

double RescaleTanh(double x);

double Clip(double x, double min, double max);

double ClipUnit(double x);

double ClipGradients(double x);
