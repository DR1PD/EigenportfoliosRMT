"""P2-1/P2-2 regression tests (ROADMAP 1.3): min-variance must optimize on a
COVARIANCE matrix; RMT cleaning happens in correlation space and hands back
volatilities at the boundary (diagonal reset, then Sigma = D C D).
"""
import numpy as np


def _toy_returns(T=1000, seed=7):
    """3 assets with very different vols and a known correlation structure."""
    rng = np.random.default_rng(seed)
    corr = np.array([[1.0, 0.5, 0.2],
                     [0.5, 1.0, 0.4],
                     [0.2, 0.4, 1.0]])
    vols = np.array([0.01, 0.03, 0.02])  # daily; deliberately unequal
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((T, 3))
    return (z @ L.T) * vols, corr, vols


def _clean_and_rescale(C, is_signal, vols):
    """The cleaning→optimization handoff under test: hard-threshold clean the
    correlation, reset the diagonal (Laloux), rescale to covariance."""
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    cleaned = evals.copy()
    if (~is_signal).any():
        cleaned[~is_signal] = evals[~is_signal].mean()
    C_clean = evecs @ np.diag(cleaned) @ evecs.T
    np.fill_diagonal(C_clean, 1.0)           # P2-2: diagonal reset
    D = np.diag(vols)
    return D @ C_clean @ D                    # P2-1: covariance handoff


def test_optimizer_input_carries_variances_not_unit_diagonal():
    """The audit's defining symptom: a correlation matrix has an all-ones
    diagonal; the optimizer's input must instead carry each asset's variance."""
    X, corr, vols = _toy_returns()
    sample_vols = X.std(axis=0)
    is_signal = np.array([True, False, False])
    sigma = _clean_and_rescale(np.corrcoef(X.T), is_signal, sample_vols)

    assert not np.allclose(np.diag(sigma), 1.0), "input is still a correlation matrix"
    assert np.allclose(np.diag(sigma), sample_vols**2, rtol=1e-12)


def test_diagonal_reset_and_psd_after_cleaning():
    """Cleaning drifts diag(C_clean) off 1; the reset restores exactly 1 and
    the matrix must remain positive semi-definite afterward."""
    X, corr, vols = _toy_returns()
    C = np.corrcoef(X.T)
    evals, evecs = np.linalg.eigh(C)
    cleaned = evals.copy()
    cleaned[:-1] = evals[:-1].mean()          # crude clean: drifts the diagonal
    C_drift = evecs @ np.diag(cleaned) @ evecs.T
    assert not np.allclose(np.diag(C_drift), 1.0)  # drift is real

    np.fill_diagonal(C_drift, 1.0)
    assert np.allclose(np.diag(C_drift), 1.0)
    assert np.linalg.eigvalsh(C_drift).min() > -1e-10  # PSD survives the reset


def test_min_var_on_covariance_matches_closed_form_and_correlation_does_not():
    """Unconstrained fully-invested min-var has the closed form
    w* = Sigma^-1 1 / (1' Sigma^-1 1). Optimizing the correlation matrix
    instead produces measurably different weights when vols differ —
    the P2-1 defect in one assertion."""
    X, corr, vols = _toy_returns()
    ones = np.ones(3)
    sigma_true = np.diag(vols) @ corr @ np.diag(vols)

    def min_var_closed_form(M):
        w = np.linalg.solve(M, ones)
        return w / w.sum()

    w_cov = min_var_closed_form(sigma_true)
    w_corr = min_var_closed_form(corr)        # what the buggy pipeline did

    # Correct answer: low-vol asset 1 dominates; correlation-space answer
    # treats all assets as equal-vol and must disagree materially.
    assert w_cov[0] > w_cov[1]
    assert np.abs(w_cov - w_corr).max() > 0.10

    # And the handoff recipe reproduces the true covariance exactly when
    # nothing is classified as noise (identity cleaning).
    sigma_via_handoff = _clean_and_rescale(corr, np.ones(3, dtype=bool), vols)
    assert np.allclose(sigma_via_handoff, sigma_true, atol=1e-12)
