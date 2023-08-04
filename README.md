# k-Means Maximum Entropy Exploration

https://arxiv.org/abs/2205.15623

## To install

Experimental framework and PyTorch KME
```
pip install -r requirements.txt
# will install torchkme from GitHub or manually:
# cd kme.all/kme.py && pip install -e .
```

C++ KME implementation and experiments
```
cd kme.all/kme.cpp && make && pip install -e .
```

## Repo structure

#### `framework`
- a simple framework to explore and play
- implements manifolds to run experiments

#### `kme.all/kme.cpp`
- a C++ implementation of KME
- a Python wrapped library for the C++ implementation
- code to reproduce entropy experiments

#### `kme.all/kme.py`
- Python package of PyTorch KME implementation

#### `mujoco`
- code to reproduce experiments in mujoco envs


## Experiments

To run entropy experiments, use the compiled binary kme/entropy. To see options
```
./entropy --help  
```

To run exploration experiments, use the script mujoco/train.py. You must install the modified stable-baselines3 repository found in mujoco/libs. The script requires the environment variable DEVICE to set the PyTorch device. To see options

```
python train.py --help
```

Please refer to the paper for parameter values. 

# To-Dos

- [] Add initialization techniques to Online KMeans @ap
- [] Handle batches in Online KMeans (w/ shuffling) @ap
- [] Optimize calls for batches and matmuls @ap
- [] Fix bug of point spwaning at (0) in viz @xan
- [] Fix vizualizer to handle any np.array shape @xan
- [] Write down skeleton of rl experiment script @xan
- [] Restructure repository according to rl exp @both
- [] Lay down clear plan for all experiments in paper @both