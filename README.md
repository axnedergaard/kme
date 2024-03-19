# Entropy Estimation for Exploration

- Exploration in high-dimensional, continuous spaces with sparse rewards in RL
- This work introduce a novel k-means density estimator to perform maximum entropy
- Our density estimator is benchmarked against knn on [DeepMind Mujoco](https://github.com/google-deepmind/mujoco) environments

## How to install dependencies

Git clone and install repo dependencies;
```bash
$ git clone https://github.com/andreakiro/rum && cd rum
$ pip install -r requirements.txt # install deps
$ pip install -e . # install rum
```

You will also need our hacked fork of SB3;
```bash
$ pip install git+https://github.com/andreakiro/stable-baselines3-rum.git
```

You should be ready to go now.

## Rum package

Rum is a package with important abstraction;
- `rum.density` : kmeans and knn density estimators
- `rum.environment` : mujoco environment gym wrappers
- `rum.geometry` : networks to learn manifold geometry
- `rum.manifold` : handcrafted Riemanian manifolds
- `rum.rewarder` : curiosity rewarders (density wrappers)

## Run experiments

Feel free to modify the configs in `experiments.config` then
```bash
$ python experiments/run.py --kwargs # run experiment
$ python experiments/plot.py --kwargs # plot experimental results
$ python experiments/visualize.py --kwargs # visualize manifold
```

The idea is that you can pass any of the following hook scripts to`run.py`;
- `intrinsic_reward(rollouts, **kwargs)`
- `extrinsic_reward(rollouts, **kwargs)`
- `pathological_updates(density, **kwargs)`
- `entropy(density, **kwargs)`
- `kmeans_loss(density, manifold, n=1e4, **kwargs)`
- `kmeans_count_variance(density, **kwargs)`
- `pdf_loss(manifold, density, n_points=1000, **kwargs)`
- `distance_loss(manifold, geometry, n_points=1000, **kwargs)`
- `state(samples, **kwargs)`

This will store results in your outputs directory and on wnb. You can then plot;
- `pdf_loss_vs_steps(data, **kwargs)`
- `distance_loss_vs_steps(data, **kwargs)`
- `count_variance_vs_beta(data, **kwargs)`
- `kmeans_loss_vs_k(data, **kwargs)`
- `extrinsic_rewards_vs_steps_single_env(data, ax: Axes = None, **kwargs)`

## A few examples

Density estimation experiments;
```bash
$ python experiments/run.py density=kmeans +script.pdf_loss=1 --kwargs
$ python experiments/run.py density=knn +script.pdf_loss=1 --kwargs
$ python experiments/plot.py +script.pdf_loss_vs_steps=1
```

Entropy estimation experiments;
```bash
$ python experiments/run.py density=kmeans +script.pdf_loss=1 --kwargs
$ python experiments/run.py density=knn +script.pdf_loss=1 --kwargs
$ python experiments/plot.py +script.pdf_loss_vs_steps=1
```

Kmeans loss experiments;
```bash
$ python experiments/run.py density=kmeans density.k=100 +script.kmeans_loss=1
$ python experiments/plot.py +script.kmeans_loss_vs_k=1
```

Kmeans count variance experiments;
```bash
$ python experiments/run.py density=kmeans +script.kmeans_count_variance=1
$ python experiments/plot.py +script.count_variance_vs_beta=1
```

Reinforcement learning experiments;
```bash
$ python experiments/run.py +script.extrinsic_reward=1
$ python experiments/plot.py exp_names=\["frivolous-heroine","waiting-penalty"\] +script.extrinsic_rewards_vs_steps_single_env=1
```

Kmeans bound uniformity (Voronoi);
```bash
$ python  experiments/voronoi.py
```

Please refer to the paper for optimal parameter values. 
