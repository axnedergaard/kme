import os
import ctypes
import numpy as np

lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'librewarder.so'))

lib.rewarder_make.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]
lib.rewarder_make.restype = ctypes.c_void_p
lib.rewarder_destroy.argtypes = [ctypes.c_void_p]
lib.rewarder_destroy.restype = None
lib.rewarder_reset.argtypes = [ctypes.c_void_p]
lib.rewarder_reset.restype = None
lib.rewarder_infer.restype = ctypes.c_double
lib.rewarder_pathological_updates.argtypes = [ctypes.c_void_p]
lib.rewarder_pathological_updates.restype = ctypes.c_int

class Rewarder():
  def __init__(self, n_actions, n_states, k, learning_rate, balancing_strength, fn_type, power_fn_exponent=0.5):
    self.n_actions = n_actions
    self.n_states = n_states
    self.rewarder = lib.rewarder_make(k, fn_type.encode('utf-8'), n_states, learning_rate, balancing_strength, power_fn_exponent)
    lib.rewarder_infer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape=(self.n_states,)), ctypes.c_bool]

  def reset(self):
    lib.rewarder_reset(self.rewarder);

  def infer(self, state, learn):
    state = state.astype(np.float64)
    reward = lib.rewarder_infer(self.rewarder, self.n_actions, self.n_states, state, learn)
    pathological_updates = lib.rewarder_pathological_updates(self.rewarder)
    return reward, pathological_updates
