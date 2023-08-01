from util import visualizer
from manifold import manifold 
import numpy as np
import time


if __name__ == '__main__':
  samples_per_render = 100
  max_samples = 100000
  min_time = 0.01
  low = -1.0
  high = 1.0
  sampler = {'type': 'uniform', 'low': low, 'high': high}
  #sampler = {'type': 'gaussian', 'mean': 0.0, 'std': 0.1}
  #sampler = {'type': 'vonmises_fisher', 'mu': [0, 1, 0], 'kappa': 10}
  #man = manifold.EuclideanManifold(3, low, high, sampler)
  man = manifold.SphericalManifold(2, sampler)
  visualizer = visualizer.Visualizer()
  #points = man.sample(n_samples) 
  n_samples = 0
  points = None 
  while n_samples < max_samples:
    time_start = time.time()
    #points = man.sample(samples_per_render)
    points = man.random_walk(samples_per_render, points[-1] if points is not None else None)
    n_samples += samples_per_render
    data = {'name': 'samples', 'points': points, 'color': [0, 255, 0]}
    visualizer.add(data)
    visualizer.render()
    time_end = time.time()
    time_elapsed = time_end - time_start
    if time_elapsed < min_time: 
      time.sleep(min_time - time_elapsed)
  #import pdb; pdb.set_trace()
