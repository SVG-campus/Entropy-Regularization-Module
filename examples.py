import numpy as np
from entropy_module import (
    entropy, entropy_gradient, entropy_regularized_update,
    portfolio_variance_gradient
)

if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    w = np.array([0.4, 0.3, 0.2, 0.1])
    cov = np.diag([0.02, 0.03, 0.01, 0.04])
    gL = portfolio_variance_gradient(w, cov)
    print("H(w) =", entropy(w))
    w_next = entropy_regularized_update(w, gL, eta=0.05, gamma=0.1)
    print("w_next:", w_next, "sum:", w_next.sum())
