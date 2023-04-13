#include "kmeans_encoder.h"

#include "../util/util.h"

#include <vector>
#include <limits>

#include <iostream>

#define EPS 1

KMeansEncoder::KMeansEncoder(int n_states, int k, double learning_rate, double balancing_strength, bool homeostasis) : Encoder(n_states, k), learning_rate(learning_rate), balancing_strength(balancing_strength), homeostasis(homeostasis) {
  closest_distances = new double[n_dims];
  closest_clusters = new int[n_dims];
  distances = new double[(n_dims + 1) * (n_dims + 1)];
  cluster_centers = new double[n_dims * n_states];
  n_points = new double[n_dims];
  Reset();
}

KMeansEncoder::KMeansEncoder(const KMeansEncoder &encoder) : Encoder(encoder.n_states, encoder.n_dims) {
  learning_rate = encoder.learning_rate;
  homeostasis = encoder.homeostasis;
  closest_distances = new double[n_dims];
  closest_clusters = new int[n_dims];
  distances = new double[(n_dims + 1) * (n_dims + 1)];
  cluster_centers = new double[n_dims * n_states];
  n_points = new double[n_dims];

  for (int i = 0; i < n_dims; i++) {
    closest_distances[i] = encoder.closest_distances[i];
    closest_clusters[i] = encoder.closest_clusters[i];
    n_points[i] = encoder.n_points[i];
    for (int j = 0; j < n_states; j++) {
      cluster_centers[i * n_states + j] = encoder.cluster_centers[i * n_states + j];
    }
    for (int j = 0; j < n_dims + 1; j++) {
      distances[i * (n_dims + 1) + j] = encoder.distances[i * (n_dims + 1) + j];
      distances[n_dims * (n_dims + 1) + j] = encoder.distances[n_dims * (n_dims + 1) + j];
    }
  }
}

KMeansEncoder::~KMeansEncoder() {
  delete closest_distances;
  delete closest_clusters;
  delete distances;
  delete cluster_centers;
  delete n_points;
}

double KMeansEncoder::ComputeDistance(double *array_1, VectorXd array_2) {
  double distance = 0.0;
  for (int j = 0; j < n_states; j++) {
    distance += pow(array_1[j] - array_2(j), 2);
  }
  return sqrt(distance);
}

double KMeansEncoder::ComputeDistance(double *array_1, double *array_2) {
  double distance = 0.0;
  for (int j = 0; j < n_states; j++) {
    distance += pow(array_1[j] - array_2[j], 2);
  }
  return sqrt(distance);
}

double KMeansEncoder::FindClosest(int index, int *closest_distance_index) {
  double closest_distance = std::numeric_limits<double>::max();
  if (homeostasis) {
    double mean = 0.0;
    for (int i = 0; i < n_dims; i++) {
      mean += n_points[i];
    }
    mean /= n_dims;
    for (int i = 0; i < n_dims; i++) {
      if (i == index) continue;
      double distance = distances[index * (n_dims + 1) + i] - balancing_strength * (mean - n_points[i]);
      if (distance < closest_distance) {
        closest_distance = distance;
        *closest_distance_index = i;
      }
    }
  } else {
    for (int i = 0; i < n_dims; i++) {
      if (i == index) continue;
      double distance = distances[index * (n_dims + 1) + i];
      if (distance <= closest_distance) {
        closest_distance = distance;
        *closest_distance_index = i;
      }
    }
  }
  return closest_distance;
}

void KMeansEncoder::Embed(VectorXd state, VectorXd *repr) {
  // Find closest neuron.
  for (int i = 0; i < n_dims; i++) {
    double distance = ComputeDistance(cluster_centers + i * n_states, state);
    distances[n_dims * (n_dims + 1) + i] = distances[i * (n_dims + 1) + n_dims] = distance;
  }
  distances[n_dims * (n_dims + 1) + n_dims] = 0.0;

  int index;
  FindClosest(n_dims, &index);

  // Merge new snapshot with existing neurons (k-means).
  for (int i = 0; i < n_states; i++) {
    cluster_centers[index * n_states + i] = learning_rate * (double)state(i) + (1.0 - learning_rate) * cluster_centers[index * n_states + i];
  }

  // Increase count for closest neuron.
  if (homeostasis) {
    n_points[index] += 1.0;
  }

  // Recompute distances.
  double updated_closest_distance_homeostatic = std::numeric_limits<float>::max();
  double updated_closest_cluster = 0;
  pathological_updates = 0;
  for (int i = 0; i < n_dims; i++) {
    if (i == index) {
      distances[index * (n_dims + 1) + i] = distances[i * (n_dims + 1) + index] = 0.0;
      continue;
    }

    double distance = ComputeDistance(cluster_centers + index * n_states, cluster_centers + i * n_states);
    distances[index * (n_dims + 1) + i] = distances[i * (n_dims + 1) + index] = distance;
    double distance_homeostatic = homeostasis ? distance + balancing_strength * (n_points[i] - n_points[index]) : distance;
    double distance_homeostatic_dual = homeostasis ? distance + balancing_strength * (n_points[index] - n_points[i]) : distance;
    if (distance_homeostatic < 0) distance_homeostatic = 0.0;
    if (distance_homeostatic_dual < 0) distance_homeostatic_dual = 0.0;

    if (distance_homeostatic <= updated_closest_distance_homeostatic) {
      updated_closest_distance_homeostatic = distance_homeostatic;
      updated_closest_cluster = i;
    }

    // Update closest distance if distance to new cluster is shorter.
    if (distance_homeostatic_dual <= closest_distances[i]) {
      closest_distances[i] = distance_homeostatic_dual;
      closest_clusters[i] = index;
    } else if (closest_clusters[i] == index) {
      pathological_updates += 1;
      // If updated cluster was previously closest cluster and moved further away, we must recompute.
      // This part messes up the otherwise O(kd) time complexity of the function, but should not
      // happen that much practice (expected number of neighbors depends on d according to Poisson
      // Voronoi tesselation theory).
      double closest_distance_homeostatic = std::numeric_limits<double>::max();
      int closest_cluster;
      for (int j = 0; j < n_dims; j++) {
        if (j == i) continue;
        double _distance = ComputeDistance(cluster_centers + i * n_states, cluster_centers + j * n_states);
        double _distance_homeostatic = homeostasis ? _distance + balancing_strength * (n_points[j] - n_points[i]) : _distance;
        if (_distance_homeostatic < 0) _distance_homeostatic = 0.0;
        if (_distance_homeostatic <= closest_distance_homeostatic) {
          closest_distance_homeostatic = _distance_homeostatic;
          closest_cluster = j;
        }
      }
      closest_distances[i] = closest_distance_homeostatic;
      closest_clusters[i] = closest_cluster;
    }

    closest_distances[index] = updated_closest_distance_homeostatic;
    closest_clusters[index] = updated_closest_cluster;
  }

  // Set embedding.
  if (repr != NULL) {
    *repr = VectorXd::Zero(n_dims);
    (*repr)(index) = 1.0;
  }
}

void KMeansEncoder::Reset() {
  reward = 0.0;
  for (int i = 0; i < (n_dims + 1) * (n_dims + 1); i++) {
    distances[i] = 0.0;
  }
  for (int i = 0; i < n_dims * n_states; i++) {
    cluster_centers[i] = 0.0;
  }
  for (int i = 0; i < n_dims; i++) {
    closest_distances[i] = 0.0;
  }
  for (int i = 0; i < n_dims; i++) {
    closest_clusters[i] = 0;
  }
  for (int i = 0; i < n_dims; i++) {
    n_points[i] = 1.0;
  }
}
