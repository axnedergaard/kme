#include <iostream>
#include <fstream>
#include <filesystem>
#include <functional>
#include <eigen3/Eigen/Dense>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <numeric>
#include <boost/program_options.hpp>
#include <boost/random.hpp>
#include "../encoder/kmeans_encoder.h"
#include "../rewarder/kmeans_rewarder.h"

using namespace std;
using namespace Eigen;
using namespace boost::program_options;

// Rendering.
int height = 800;
int width = 800;
bool show_states = true;
bool show_clusters = true;
bool show_boundaries = false;
bool quit = false;

// Sampling.
int max_rejections = 500;

void log(ofstream *entropy_fp,
         ofstream *pat_upd_fp,
         ofstream *clusters_fp,
         ofstream *states_fp,
         double entropy,
         int pat_upd,
         double *clusters,
         double *states,
         int n_states,
         int k,
         int batch_size) {
  if (entropy_fp->is_open()) {
    *entropy_fp << entropy << endl;
  }
  if (pat_upd_fp->is_open()) {
    *pat_upd_fp << pat_upd << endl;
  }
  if (clusters_fp->is_open()) {
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n_states; j++) {
        *clusters_fp << clusters[i * n_states + j];
        if (j + 1 < n_states) *clusters_fp << " ";
      }
      *clusters_fp << endl;
    }
    }
  if (states_fp->is_open()) {
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < n_states; j++) {
        *states_fp << states[i * n_states + j];
        if (j + 1 < n_states) *states_fp << " ";
      }
      *states_fp << endl;
    }
  }
}


double ComputeDistance(double *array_1, double *array_2, int d) {
  double distance = 0.0;
  for (int j = 0; j < d; j++) {
    distance += pow(array_1[j] - array_2[j], 2);
  }
  return distance;
}

class Distribution {
  public:
    virtual double pdf(VectorXd x) = 0;
};

double variance_to_bound(double var) {
  // 0.1 -> 1, 0.01 -> 0.25,, 0.001 -> 0.125...
  return pow(2.0f, 1.0f + log10(var));
}

class GaussianDistribution : public Distribution {
  protected:
    MatrixXd inv_cov;
    double gaussian_constant;
    VectorXd mean;
  public:
    double var;
    GaussianDistribution(VectorXd mean, double var) : mean(mean), var(var) {
      int dim = mean.size();
      MatrixXd cov = MatrixXd::Identity(dim, dim) * var;
      inv_cov = cov.inverse();
      double det_cov = cov.determinant();
      gaussian_constant = pow(2 * M_PI, -1) * pow(det_cov, -0.5);
    }

    double pdf(VectorXd x) {
      double exponent = -0.5 * (x - mean).transpose() * inv_cov * (x - mean);
      double prob = gaussian_constant * exp(exponent);
      return prob;
    }
};

class GaussianMixtureDistribution : public GaussianDistribution {
  private:
    int n_mixtures;
    std::vector<VectorXd> means;
  public:
    GaussianMixtureDistribution(VectorXd mean, double var, int n_mixtures) : GaussianDistribution(mean, var), n_mixtures(n_mixtures) {
      if (mean.size() != 2) {
        std::cout << "Error: Dimension must be 2." << std::endl;
        return;
      }
      if (n_mixtures != 2 && n_mixtures != 4) {
        std::cout << "Error: Number of mixtures must be 2 or 4." << std::endl;
        return;
      }
      means.push_back(VectorXd({{mean(0), mean(1)}}));
      means.push_back(VectorXd({{-mean(0), -mean(1)}}));
      if (n_mixtures > 2) {
        means.push_back(VectorXd({{mean(0), -mean(1)}}));
        means.push_back(VectorXd({{-mean(0), mean(1)}}));
      }
    }

  double pdf(VectorXd x) {
    double prob = 0.0;
    for (int i = 0; i < n_mixtures; i++) {
      double exponent = -0.5 * (x - means[i]).transpose() * inv_cov * (x - means[i]);
      prob += exp(exponent);
    }
    prob *= gaussian_constant;;
    return prob;
  }
};


class UniformDistribution : public Distribution {
  private:
    int dim;
  public:
    UniformDistribution(int dim) : dim(dim) {};
    double pdf(VectorXd x) {
      for (int i = 0; i < dim; i++) {
        if (x(i) < -1 || x(i) < -1) {
          return 0.0f;
        }
      }
      return 1.0f;
    }
};

VectorXd sample_random_walk(VectorXd x, Distribution *distribution, int reset_every_n, double step_size, double max_prob, int dim, boost::variate_generator<boost::random::mt19937&, boost::uniform_on_sphere<double>> random_on_sphere) {
  VectorXd proposal_x(dim);
  int n_rejections = 0;
  while (true) {
    std::vector<double> direction = random_on_sphere();
    for (int i = 0; i < dim; i++) {
      proposal_x(i) = x(i) + step_size * direction[i];
    }
    double proposal_prob = distribution->pdf(proposal_x);
    double uniform_prob = (double)rand() / RAND_MAX;
    if (uniform_prob * max_prob < proposal_prob) {
      return proposal_x;
    }
    n_rejections++;
    if (n_rejections > max_rejections) {
      proposal_x.setZero();
      return proposal_x;
    }
  }
}

VectorXd sample_random(Distribution *distribution, double max_prob, int dim, bool gaussian_efficiency_hack=true) {
  // Acceptance-rejection sampling.
  VectorXd proposal_x(dim);
  while (true) {
    for (int i = 0; i < dim; i++) {
      double uniform_x = ((double)rand() / RAND_MAX) * 2 - 1;
      if (gaussian_efficiency_hack) {
        double scaling = variance_to_bound(((GaussianDistribution*)distribution)->var);
        uniform_x *= scaling;
      }
      proposal_x[i] = uniform_x;
    }
    double uniform_z = ((double)rand() / RAND_MAX) * max_prob;
    double proposal_prob = distribution->pdf(proposal_x);
    if (uniform_z < proposal_prob) {
      return proposal_x;
    }
  }
}

void render_trajectories(double *trajectories, int n_steps) {
  glColor3f(0, 1, 0);
 glPointSize(1);
  glBegin(GL_POINTS);
  for (int step = 0; step < n_steps; step++) {
    double x = (double)trajectories[step * 2];
    double y = (double)trajectories[step * 2 + 1];
    glVertex2f(x, y);
  }
  glEnd();
}

void render_clusters(KMeansEncoder *encoder) {
  glColor3f(1, 1, 1);
  glPointSize(5);
  glBegin(GL_POINTS);
  double *cluster_centers = encoder->cluster_centers;
  for (int cluster = 0; cluster < encoder->n_dims; cluster++) {
    double x = (double)cluster_centers[cluster * 2];
    double y = (double)cluster_centers[cluster * 2 + 1];
    glVertex2f(x, y);
  }
  glEnd();
}

void render_decision_regions(KMeansEncoder *encoder, int granularity) {
  double cell_size = 1.0 / granularity;
  double distances[encoder->n_dims];
  //std::vector<int> neighbors;
  glColor3f(1, 1, 1);
  glPointSize(1);
  glBegin(GL_POINTS);
  for (int i = 0; i < granularity; i++) {
    for (int j = 0; j < granularity; j++) {
      int closest_cluster = 0;
      double closest_distance = std::numeric_limits<double>::max();
      double point[2];
      point[0] = (i + 0.5) * cell_size * 2 - 1;
      point[1] = (j + 0.5) * cell_size * 2 - 1;
      for (int k = 0; k < encoder->n_dims; k++) {
        double distance = ComputeDistance(encoder->cluster_centers + k * 2, point, 2);
        distances[k] = distance;
        if (distance < closest_distance) {
          closest_distance = distance;
          closest_cluster = k;
        }
      }
      for (int k = 0; k < encoder->n_dims; k++) {
        if (k == closest_cluster) continue;
        if (distances[k] - closest_distance < cell_size * 0.5) {
          glVertex2f(point[0], point[1]);
          break;
        }
      }
    }
  }
  glEnd();
}

void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
  if (action == GLFW_PRESS && key == GLFW_KEY_Q) {
    quit = true;
  }
  else if (action == GLFW_RELEASE) {
    if (key == GLFW_KEY_1) {
      show_states = !show_states;
    } else if (key == GLFW_KEY_2) {
      show_clusters = !show_clusters;
    } else if (key == GLFW_KEY_3) {
      show_boundaries = !show_boundaries;
    }
  }
}

int main(int argc, char **argv) {
  // Process parameters.
  options_description desc("options");
  desc.add_options()
    ("n_steps", value<int>()->default_value(1000000))
    ("step_size", value<double>()->default_value(0.01))
    ("distribution", value<string>()->default_value("gaussian"))
    ("random_walk", value<bool>()->default_value(false))
    ("dim", value<int>()->default_value(2))
    ("mean", value<std::vector<double> >()->multitoken())
    ("var", value<double>()->default_value(0.1))
    ("n_mixtures", value<int>()->default_value(1))
    ("k", value<int>()->default_value(300))
    ("balancing_strength", value<double>()->default_value(0.0001))
    ("learning_rate", value<double>()->default_value(0.05))
    ("batch_size", value<int>()->default_value(1))
    ("reset", value<int>()->default_value(0))
    ("fn", value<string>()->default_value(""))
    ("store_states", value<bool>()->default_value(false))
    ("render", value<bool>()->default_value(true))
    ("print", value<bool>()->default_value(false))
    ("help", "Print help message.")
  ;
  variables_map vm;
  try {
    store(parse_command_line(argc, argv, desc), vm);
  } catch(exception &e) {
    cout << e.what() << endl;
    return 1;
  }
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }
  int k = vm["k"].as<int>();
  int batch_size = vm["batch_size"].as<int>();
  int reset = vm["reset"].as<int>();
  double learning_rate = vm["learning_rate"].as<double>();
  double balancing_strength = vm["balancing_strength"].as<double>();

  bool random_walk = vm["random_walk"].as<bool>();
  double step_size = vm["step_size"].as<double>();
  int dim = vm["dim"].as<int>();
  int n_mixtures = vm["n_mixtures"].as<int>();

  srand(time(NULL));

  // Init distribution.
  Distribution *distribution;
  string distribution_type = vm["distribution"].as<string>();
  std::vector<double> mean_vec;
  if (!vm["mean"].empty()) {
    mean_vec = vm["mean"].as<std::vector<double>>();
    dim = mean_vec.size();
  } else if (distribution_type == "gaussian") {
    for (int i = 0; i < dim; i++) {
      mean_vec.push_back(0);
    }
  }
  double max_prob;
  if (distribution_type == "uniform") {
    distribution = new UniformDistribution(dim);
  } else if (distribution_type == "gaussian") {
    VectorXd mean(dim);
    for (int i = 0; i < dim; i++) {
      mean(i) = mean_vec[i];
    }
    double var = vm["var"].as<double>();
    if (n_mixtures > 1) {
      distribution = new GaussianMixtureDistribution(mean, var, n_mixtures);
    } else {
      distribution = new GaussianDistribution(mean, var);
    }
    max_prob = distribution->pdf(mean);
  } else {
    cout << "Distribution " << distribution_type << " not found." << endl;
    return 1;
  }

  // Init kmeans.
  KMeansRewarder *rewarder = new KMeansRewarder(k, EntropicFunctionType::POWER, dim, learning_rate, balancing_strength, 1e-9, false, 0.5);
  KMeansEncoder *encoder = rewarder->encoder;
  rewarder->Reset();

  // Init renderer.
  bool render = vm["render"].as<bool>();
  if (dim != 2 && render) {
    cout << "Warning: Not rendering as dim not equal to 2." << endl;
    render = false;
  }
  GLFWwindow *window;
  if (render) {
    glfwSetErrorCallback(error_callback);
    if (!glfwInit()) cout << "Failed to init glfw." << endl;
    window = glfwCreateWindow(width, height, "GLFW", NULL, NULL);
    if (!window) {
      cout << "Failed to init window." << endl;
      exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);
    gladLoadGL(glfwGetProcAddress);
    glfwSwapInterval(1);
    glfwSetKeyCallback(window, key_callback);
  }

  bool print = vm["print"].as<bool>();

  // Init logging.
  bool store_states =  vm["store_states"].as<bool>();
  string output_path = "output";
  string fn = vm["fn"].as<string>();
  ofstream config_fp, entropy_fp, pat_upd_fp, clusters_fp, states_fp;
  int n_written = 0;
  if (fn != "") {
    fn = output_path + "/" + fn;
    filesystem::create_directory(output_path);
    // Write config.
    config_fp.open(fn + "_config");
    for (int i = 0; i < argc; i++) {
      config_fp << argv[i];
      if (i + 1 < argc) config_fp << " ";
    }
    config_fp.close();
    entropy_fp.open(fn + "_entropy");
    pat_upd_fp.open(fn + "_pat_upd");
    if (store_states) {
      states_fp.open(fn + "_states");
    }
  }

  // Init random.
  boost::random::mt19937 rand_gen;
  rand_gen.seed(static_cast<unsigned int>(std::time(0)));
  boost::uniform_on_sphere<double> unif_sphere(dim);
  boost::variate_generator<boost::random::mt19937&, boost::uniform_on_sphere<double> > random_on_sphere(rand_gen, unif_sphere);

  // Main loop.
  int step = 0;
  int n = vm["n_steps"].as<int>();
  int n_steps_per_render = 1000;
  double *trajectories = (double*)malloc(sizeof(double) * n * dim);
  double *batch = (double*)malloc(sizeof(double) * batch_size * dim);
  VectorXd x(dim);
  x.setZero();
  for (step = 0; step < n && !quit; step++) {
    if (reset > 0 && step > 0 && step % reset == 0) {
      x.setZero();
    }
    // Update position.
    if (random_walk) {
      x = sample_random_walk(x, distribution, -1, step_size, max_prob, dim, random_on_sphere);
    } else {
      x = sample_random(distribution, max_prob, dim, distribution_type == "gaussian" && n_mixtures == 1);
    }
    for (int i = 0; i < dim; i++) {
      trajectories[step * dim + i] = x(i);
    }
    // Update k-means clustering.
    if (batch_size > 1) {
      if (step > 0 && step % batch_size == 0) {
        // Shuffle states into batch.
        int shuffled_indices[batch_size];
        iota(shuffled_indices, shuffled_indices + batch_size, 0);
        random_shuffle(shuffled_indices, shuffled_indices + batch_size);
        for (int i = 0; i < batch_size; i++) {
          int index = shuffled_indices[i];
          for (int j = 0; j < dim; j++) {
            batch[i * dim + j] = trajectories[(step - batch_size + index) * dim + j];
          }
        }
        // Learn.
        for (int i = 0; i < batch_size; i++) {
          VectorXd sampled_x(dim);
          for (int j = 0; j < dim; j++) {
            sampled_x(j) = batch[i * dim + j];
          }
          encoder->Embed(sampled_x, NULL);
        }
        double entropy_lower_bound = rewarder->EstimateEntropy(encoder) / k;
        if (print) {
          cout << entropy_lower_bound << endl;
        }
        if (fn != "") {
          int pat_upds = encoder->pathological_updates;
          log(&entropy_fp, &pat_upd_fp, &clusters_fp, &states_fp, entropy_lower_bound, pat_upds, NULL, batch, dim, k, batch_size);
          n_written++;
        }
      }
    } else {
      encoder->Embed(x, NULL);
      double entropy_lower_bound = rewarder->EstimateEntropy(encoder) / k;
      if (print) {
        cout << entropy_lower_bound << endl;
      }
      if (fn != "") {
        int pat_upds = encoder->pathological_updates;
        log(&entropy_fp, &pat_upd_fp, &clusters_fp, &states_fp, entropy_lower_bound, pat_upds, NULL, batch, dim, k, batch_size);
        n_written++;
      }
    }

    // Render.
    if (render && step % n_steps_per_render == 0) {
      glClear(GL_COLOR_BUFFER_BIT);
      glfwGetFramebufferSize(window, &width, &height);
      glViewport(0, 0, width, height);
      //cout << x.transpose() << endl;
      if (show_states) render_trajectories(trajectories, step);
      if (show_clusters) render_clusters(encoder);
      if (show_boundaries) render_decision_regions(encoder, 400);
      glfwSwapBuffers(window);
      glfwPollEvents();
    }
  }

  if (fn != "") {
    entropy_fp.close();
    pat_upd_fp.close();
    if (store_states) states_fp.close();
    clusters_fp.open(fn + "_clusters");
    log(&entropy_fp, &pat_upd_fp, &clusters_fp, &states_fp, 0, 0, encoder->cluster_centers, 0, dim, k, batch_size);
    clusters_fp.close();
    cout << "Wrote " << n_written << " entries to " << fn << "." << endl;
  }

  free(trajectories);
  free(batch);
  if (render && !quit) getchar();
}
