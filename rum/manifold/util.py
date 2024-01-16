def sphere_sample_uniform(dim):
  x = np.random.normal(0, 1, dim + 1)
  return x / np.linalg.norm(x)

# TODO: Consider deleting all below if not necessary.

def standardize_angle(x):
  x = x % (2 * np.pi)
  if x < 0:
    x += 2 * np.pi
  return x

def modular_distance(x, y, m):
  d = 0
  for _x, _y in zip(x, y):
    _x = _x % m
    if _x < 0:
      _x += m
    _y = _y % m
    if _y < 0:
      _y += m
    d += (_x - _y) ** 2
  return d

def modular_equals(x, y, m):
  for _x, _y in zip(x, y):
    _x = _x % m
    if _x < 0:
      _x += m
    _y = _y % m
    if _y < 0:
      _y += m
    if _x != _y:
      return False
  return True

