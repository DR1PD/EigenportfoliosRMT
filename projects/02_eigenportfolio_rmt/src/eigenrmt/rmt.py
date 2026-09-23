"""Marchenko-Pastur machinery (extracted verbatim from the notebook, 2.2).

Eigendecomposition in correlation space, the MP null density and edges,
iterative sigma^2 renormalization (P2-3 fix), and signal/noise classification.
"""
import numpy as np
import pandas as pd


def compute_eigen(returns_window):
    """
    Compute eigendecomposition of the sample correlation matrix.
    
    Args:
        returns_window: (T, n) DataFrame of returns
    
    Returns:
        eigenvalues: (n,) sorted descending
        eigenvectors: (n, n) columns are eigenvectors, sorted by eigenvalue
        corr_matrix: (n, n) sample correlation matrix
        meta: dict with diagnostics
    """
    # Standardize to zero mean, unit variance (correlation matrix)
    X = returns_window.values
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Sample correlation matrix
    T, n = X_std.shape
    C = X_std.T @ X_std / T
    
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(C)
    
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Diagnostics
    meta = {
        'n': n, 'T': T, 'q': n / T,
        'condition_number': eigvals[0] / max(eigvals[-1], 1e-10),
        'trace': eigvals.sum(),
        'top1_var_pct': eigvals[0] / eigvals.sum(),
        'top5_var_pct': eigvals[:5].sum() / eigvals.sum(),
        'top10_var_pct': eigvals[:10].sum() / eigvals.sum(),
    }
    
    return eigvals, eigvecs, C, meta


def marchenko_pastur_pdf(x, q, sigma2=1.0):
    """
    Marchenko-Pastur probability density function.
    
    Args:
        x: eigenvalue(s)
        q: ratio n/T
        sigma2: variance of individual returns (1.0 for correlation matrix)
    
    Returns:
        density at each x
    """
    lambda_plus = sigma2 * (1 + np.sqrt(q))**2
    lambda_minus = sigma2 * (1 - np.sqrt(q))**2
    
    pdf = np.zeros_like(x, dtype=float)
    mask = (x >= lambda_minus) & (x <= lambda_plus) & (x > 0)
    
    pdf[mask] = (1 / (2 * np.pi * sigma2 * q * x[mask])) * \
                np.sqrt((lambda_plus - x[mask]) * (x[mask] - lambda_minus))
    
    return pdf


def mp_edges(q, sigma2=1.0):
    """Compute Marchenko-Pastur upper and lower edges."""
    lambda_plus = sigma2 * (1 + np.sqrt(q))**2
    lambda_minus = sigma2 * (1 - np.sqrt(q))**2
    return lambda_minus, lambda_plus


def fit_mp_sigma_naive(eigvals, q):
    """Pre-P2-3 one-pass estimate, kept ONLY for the naive-vs-iterative
    comparison in the signature plot: mean of the bulk below the
    preliminary sigma^2=1 edge — moderate factors below that edge
    contaminate the average and inflate lambda_+."""
    _, lp_init = mp_edges(q, 1.0)
    noise_eigvals = eigvals[eigvals <= lp_init]
    return noise_eigvals.mean() if len(noise_eigvals) > 5 else 1.0


def fit_mp_sigma(eigvals, q, max_iter=100, tol=1e-8):
    """
    P2-3 fix: iterative sigma^2 renormalization (Laloux-style).

    The correlation trace is fixed at n, and signal eigenvalues absorb part
    of it; the noise bulk only gets what is left, per noise dimension:

        sigma^2_(k) = (1 - sum_signal(lambda_i) / n) / (1 - k / n)

    Start at sigma^2 = 1, classify against lambda_+, re-estimate excluding
    the signals, reclassify — the edge only moves down, so the signal set
    only grows and the loop must terminate. Converged when the signal set
    is stable and sigma^2 moves < tol.
    """
    n = len(eigvals)
    sigma2, k_prev = 1.0, -1
    for _ in range(max_iter):
        _, lam_plus_it = mp_edges(q, sigma2)
        is_signal = eigvals > lam_plus_it
        k = int(is_signal.sum())
        if k >= n:
            break
        sigma2_new = (1.0 - eigvals[is_signal].sum() / n) / (1.0 - k / n)
        if sigma2_new <= 0:
            break
        if k == k_prev and abs(sigma2_new - sigma2) < tol:
            sigma2 = sigma2_new
            break
        k_prev, sigma2 = k, sigma2_new
    return sigma2


def mp_edges(q, sigma2=1.0):
    """Compute Marchenko-Pastur upper and lower edges."""
    lambda_plus = sigma2 * (1 + np.sqrt(q))**2
    lambda_minus = sigma2 * (1 - np.sqrt(q))**2
    return lambda_minus, lambda_plus


def fit_mp_sigma_naive(eigvals, q):
    """Pre-P2-3 one-pass estimate, kept ONLY for the naive-vs-iterative
    comparison in the signature plot: mean of the bulk below the
    preliminary sigma^2=1 edge — moderate factors below that edge
    contaminate the average and inflate lambda_+."""
    _, lp_init = mp_edges(q, 1.0)
    noise_eigvals = eigvals[eigvals <= lp_init]
    return noise_eigvals.mean() if len(noise_eigvals) > 5 else 1.0


def fit_mp_sigma(eigvals, q, max_iter=100, tol=1e-8):
    """
    P2-3 fix: iterative sigma^2 renormalization (Laloux-style).

    The correlation trace is fixed at n, and signal eigenvalues absorb part
    of it; the noise bulk only gets what is left, per noise dimension:

        sigma^2_(k) = (1 - sum_signal(lambda_i) / n) / (1 - k / n)

    Start at sigma^2 = 1, classify against lambda_+, re-estimate excluding
    the signals, reclassify — the edge only moves down, so the signal set
    only grows and the loop must terminate. Converged when the signal set
    is stable and sigma^2 moves < tol.
    """
    n = len(eigvals)
    sigma2, k_prev = 1.0, -1
    for _ in range(max_iter):
        _, lam_plus_it = mp_edges(q, sigma2)
        is_signal = eigvals > lam_plus_it
        k = int(is_signal.sum())
        if k >= n:
            break
        sigma2_new = (1.0 - eigvals[is_signal].sum() / n) / (1.0 - k / n)
        if sigma2_new <= 0:
            break
        if k == k_prev and abs(sigma2_new - sigma2) < tol:
            sigma2 = sigma2_new
            break
        k_prev, sigma2 = k, sigma2_new
    return sigma2


def fit_mp_sigma_naive(eigvals, q):
    """Pre-P2-3 one-pass estimate, kept ONLY for the naive-vs-iterative
    comparison in the signature plot: mean of the bulk below the
    preliminary sigma^2=1 edge — moderate factors below that edge
    contaminate the average and inflate lambda_+."""
    _, lp_init = mp_edges(q, 1.0)
    noise_eigvals = eigvals[eigvals <= lp_init]
    return noise_eigvals.mean() if len(noise_eigvals) > 5 else 1.0


def fit_mp_sigma(eigvals, q, max_iter=100, tol=1e-8):
    """
    P2-3 fix: iterative sigma^2 renormalization (Laloux-style).

    The correlation trace is fixed at n, and signal eigenvalues absorb part
    of it; the noise bulk only gets what is left, per noise dimension:

        sigma^2_(k) = (1 - sum_signal(lambda_i) / n) / (1 - k / n)

    Start at sigma^2 = 1, classify against lambda_+, re-estimate excluding
    the signals, reclassify — the edge only moves down, so the signal set
    only grows and the loop must terminate. Converged when the signal set
    is stable and sigma^2 moves < tol.
    """
    n = len(eigvals)
    sigma2, k_prev = 1.0, -1
    for _ in range(max_iter):
        _, lam_plus_it = mp_edges(q, sigma2)
        is_signal = eigvals > lam_plus_it
        k = int(is_signal.sum())
        if k >= n:
            break
        sigma2_new = (1.0 - eigvals[is_signal].sum() / n) / (1.0 - k / n)
        if sigma2_new <= 0:
            break
        if k == k_prev and abs(sigma2_new - sigma2) < tol:
            sigma2 = sigma2_new
            break
        k_prev, sigma2 = k, sigma2_new
    return sigma2


def fit_mp_sigma(eigvals, q, max_iter=100, tol=1e-8):
    """
    P2-3 fix: iterative sigma^2 renormalization (Laloux-style).

    The correlation trace is fixed at n, and signal eigenvalues absorb part
    of it; the noise bulk only gets what is left, per noise dimension:

        sigma^2_(k) = (1 - sum_signal(lambda_i) / n) / (1 - k / n)

    Start at sigma^2 = 1, classify against lambda_+, re-estimate excluding
    the signals, reclassify — the edge only moves down, so the signal set
    only grows and the loop must terminate. Converged when the signal set
    is stable and sigma^2 moves < tol.
    """
    n = len(eigvals)
    sigma2, k_prev = 1.0, -1
    for _ in range(max_iter):
        _, lam_plus_it = mp_edges(q, sigma2)
        is_signal = eigvals > lam_plus_it
        k = int(is_signal.sum())
        if k >= n:
            break
        sigma2_new = (1.0 - eigvals[is_signal].sum() / n) / (1.0 - k / n)
        if sigma2_new <= 0:
            break
        if k == k_prev and abs(sigma2_new - sigma2) < tol:
            sigma2 = sigma2_new
            break
        k_prev, sigma2 = k, sigma2_new
    return sigma2


def classify_eigenvalues(eigvals, q, sigma2=None):
    """
    Classify eigenvalues as signal or noise using MP upper edge.
    
    Returns:
        is_signal: boolean array
        lambda_plus: MP upper edge
        n_signal: number of signal eigenvalues
        effective_rank: same as n_signal
    """
    if sigma2 is None:
        sigma2 = fit_mp_sigma(eigvals, q)
    
    _, lambda_plus = mp_edges(q, sigma2)
    
    is_signal = eigvals > lambda_plus
    n_signal = is_signal.sum()
    
    return is_signal, lambda_plus, n_signal


def rolling_mean_corr(rets, window=63, step=21):
    """Mean pairwise correlation over a rolling window (EDA diagnostic)."""
    dates, vals = [], []
    for i in range(window, len(rets), step):
        c = rets.iloc[i - window:i].corr().values
        tri = c[np.triu_indices_from(c, k=1)]
        dates.append(rets.index[i])
        vals.append(tri.mean())
    return pd.Series(vals, index=dates)
