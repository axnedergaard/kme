# k-Means Maximum Entropy Exploration

https://arxiv.org/abs/2205.15623

The directory mujoco contains code to reproduce exploration experiments and the directory kme contains
- a C++ implementation of KME
- a Python wrapped library for the C++ implementation
- code to reproduce entropy experiments

To install 
```
cd kme && make && pip install .
```

To run entropy experiments, use the compiled binary kme/entropy. To see options
```
./entropy --help  
```

To run exploration experiments, use the script mujoco/train.py. You must install the modified stable-baselines3 repository found in mujoco/libs. The script requires the environment variable DEVICE to set the PyTorch device. To see options
```
python train.py --help
```

Please refer to the paper for parameter values. 
