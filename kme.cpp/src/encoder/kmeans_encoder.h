#pragma once

#include "encoder.h"

class KMeansEncoder : public Encoder {
  private:
    double learning_rate;
    double balancing_strength;
    int homeostasis;
    double *distances;
    int n_points_std;
    double FindClosest(int index, int *max_distance_index);
    double ComputeDistance(double *array_1, VectorXd array_2);
    double ComputeDistance(double *array_1, double *array_2);
    void UpdatePointStatistics();
  public:
    int *closest_clusters;
    double *cluster_centers;
    double *closest_distances;
    double *n_points;
    int pathological_updates; // For investigating practical time complexity.
    KMeansEncoder(int n_states, int k, double learning_rate, double balancing_strength, bool homeostasis = true);
    KMeansEncoder(const KMeansEncoder &encoder);
    ~KMeansEncoder();
    void Embed(VectorXd state, VectorXd *repr);
    void Reset();
};
