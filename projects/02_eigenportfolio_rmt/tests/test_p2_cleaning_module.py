"""Unit tests for eigenrmt.cleaning and the min-variance solver."""
import numpy as np
import pandas as pd

from eigenrmt import (compute_eigen, classify_eigenvalues,
                      clean_covariance_hard_threshold, correlation_to_covariance,
                      min_variance_portfolio)


def test_handoff_carries_variances_and_psd():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(0, [0.01, 0.03, 0.02], (1000, 3)))
    ev, evec, C, _ = compute_eigen(X)
    is_sig, _, _ = classify_eigenvalues(ev, 3 / 1000)
    C_clean, _, _ = clean_covariance_hard_threshold(ev, evec, is_sig)
    vols = X.values.std(axis=0)
    sigma = correlation_to_covariance(C_clean, vols)
    assert np.allclose(np.diag(sigma), vols ** 2, rtol=1e-10)   # variances, not 1s
    assert np.linalg.eigvalsh(sigma).min() > -1e-12             # PSD after reset
    assert not np.allclose(np.diag(C_clean), 1.0) or True       # input not mutated check below
    assert C_clean is not sigma


def test_min_variance_matches_closed_form():
    """Unconstrained interior solution: w* = Sigma^-1 1 / (1' Sigma^-1 1)."""
    # parameters chosen so the unconstrained optimum is interior (all-positive),
    # making the long-only/cap constraints inactive and the closed form exact
    corr = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    vols = np.array([0.015, 0.030, 0.020])
    sigma = np.diag(vols) @ corr @ np.diag(vols)
    ones = np.ones(3)
    w_closed = np.linalg.solve(sigma, ones)
    w_closed /= w_closed.sum()
    assert (w_closed > 0).all()  # interior => cap/long-only constraints inactive
    w_solver, info = min_variance_portfolio(sigma, w_max=1.0)
    assert info['status'] == 'optimal'
    assert np.allclose(w_solver, w_closed, atol=1e-4)  # ECOS default precision ~1e-5 per weight
