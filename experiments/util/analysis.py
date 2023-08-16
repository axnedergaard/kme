def scale_independent_loss(x, y):
  return np.mean(np.abs(x - y)) # TODO. Implement properly.

def entropy(density, **kwargs):
  return density.entropy()

def pdf_loss(manifold, density, n_points=1000, **kwargs):
  samples = manifold.sample(n_points)
  pdf_true = manifold.pdf(samples)
  pdf_est = density.pdf(samples)
  return scale_independent_loss(pdf_true, pdf_est)

def state(samples, **kwargs):
  return samples

def test(success=True, **kwargs):
  if success:
    print('Test succeeded.')
  else:
    print('Test failed (but succeeeded).')
