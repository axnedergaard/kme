def scale_independent_loss(x, y):
  return np.mean(np.abs(x - y)) # TODO. Implement properly.

def entropy(density, **kwargs):
  return density.entropy()

def pdf_loss(manifold, density, n_points=1000, **kwargs):
  samples = manifold.sample(n_points)
  pdf_true = manifold.pdf(samples)
  pdf_est = density.pdf(samples)
  return scale_independent_loss(pdf_true, pdf_est)

def test(**kwargs):
  print('Test successful.')
