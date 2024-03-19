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
