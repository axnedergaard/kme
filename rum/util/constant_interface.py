import numpy as np

class ConstantInterface():
  def __init__(self, parameters, changes={}, defaults={}):
    self.changes = changes
    self.values = {}
    self.low = {}
    self.high = {}
    for parameter in parameters:
      name = parameter[0]
      if name in defaults:
        self.values[name] = defaults[name]
      else:
        self.values[name] = (parameter[2] - parameter[1]) / 2.0
      self.low[name] = parameter[1]
      self.high[name] = parameter[2]

  def get_values(self):
    for name in self.changes.keys():
      self.values[name] += self.changes[name]
      if self.values[name] < self.low[name]:
        offset = self.low[name] - self.values[name]
        self.values[name] = self.high[name] - offset 
      if self.values[name] > self.high[name]:
        offset = self.values[name] - self.high[name]
        self.values[name] = self.low[name] + offset
    return self.values.copy()



