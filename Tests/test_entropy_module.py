import numpy as np
from entropy_module import (
    entropy, entropy_gradient, entropy_regularized_update,
    projected_simplex, portfolio_variance_gradient
)

def test_entropy_and_gradient_shapes():
    w = np.array([0.3, 0.2, 0.5])
    H = entropy(w)
    gH = entropy_gradient(w)
    assert np.isscalar(H)
    assert gH.shape == w.shape

def test_update_invariants_sum1_nonneg():
    w = np.array([0.25, 0.25, 0.25, 0.25])
    cov = np.diag([0.02, 0.03, 0.01, 0.04])
    gL = portfolio_variance_gradient(w, cov)
    w2 = entropy_regularized_update(w, gL, eta=0.05, gamma=0.05)
    assert np.all(w2 >= 0)
    assert np.isclose(w2.sum(), 1.0)

def test_gamma_increases_uniformity():
    w = np.array([0.6, 0.2, 0.2])
    cov = np.diag([0.02, 0.03, 0.01])
    gL = portfolio_variance_gradient(w, cov)
    w_low = entropy_regularized_update(w, gL, eta=0.1, gamma=0.01)
    w_high = entropy_regularized_update(w, gL, eta=0.1, gamma=0.2)
    from entropy_module import entropy as H
    assert H(w_high) >= H(w_low) - 1e-9
