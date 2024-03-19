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

Please refer to the paper for optimal parameter values. 
