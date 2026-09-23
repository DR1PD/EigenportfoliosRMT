"""Unit tests for eigenrmt.rmt — the synthetic MP cases now exercise the REAL
extracted functions (the repo-level tests/test_mp_edge.py keeps independent
recipe replicas as a verify-two-ways cross-check).
"""
import numpy as np
import pandas as pd

from eigenrmt import (compute_eigen, mp_edges, fit_mp_sigma,
                      fit_mp_sigma_naive, classify_eigenvalues)

N, T = 100, 250
Q = N / T


def _eigvals(X):
    ev, _, _, _ = compute_eigen(pd.DataFrame(X))
    return ev


def test_pure_noise_zero_signals():
    rng = np.random.default_rng(11)
    ev = _eigvals(rng.standard_normal((T, N)))
    _, _, n_sig = classify_eigenvalues(ev, Q)
    assert n_sig == 0


def test_single_factor_exactly_one_signal():
    rng = np.random.default_rng(11)
    X = 0.6 * rng.standard_normal(T)[:, None] * np.ones(N) + rng.standard_normal((T, N))
    _, _, n_sig = classify_eigenvalues(_eigvals(X), Q)
    assert n_sig == 1


def test_iterative_edge_beats_naive_on_spiked_model():
    """11 planted factors, blocks tuned below the naive sigma^2=1 edge:
    iterative recovers 10 (11th below BBP detectability), naive 7."""
    B = np.zeros((11, N))
    B[0] = 0.7
    for j in range(10):
        B[j + 1, 10 * j:10 * (j + 1)] = 0.45
    rng = np.random.default_rng(1)
    ev = _eigvals(rng.standard_normal((T, 11)) @ B + rng.standard_normal((T, N)))

    def count(sigma2):
        return int((ev > mp_edges(Q, sigma2)[1]).sum())

    k_iter, k_naive = count(fit_mp_sigma(ev, Q)), count(fit_mp_sigma_naive(ev, Q))
    assert (k_iter, k_naive) == (10, 7)
    assert fit_mp_sigma(ev, Q) < fit_mp_sigma_naive(ev, Q)
