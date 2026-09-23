"""Eigenportfolio construction (extracted from the notebook, 2.2)."""
import numpy as np


def build_eigenportfolios(eigvals, eigvecs, is_signal, asset_names):
    """
    Construct eigenportfolios from significant eigenvectors.

    P2-4 fix: eigvals is an explicit parameter (the notebook version read
    it from the enclosing scope, which breaks on extraction — AUDIT P2-4).
    Normalize to sum of absolute weights = 1 (long-short).
    """
    portfolios = []
    for k in range(is_signal.sum()):
        v = eigvecs[:, k]
        w = v / np.abs(v).sum()  # normalize
        
        portfolios.append({
            'index': k + 1,
            'weights': w,
            'long_assets': [(asset_names[i], w[i]) for i in np.argsort(w)[::-1] if w[i] > 0.01],
            'short_assets': [(asset_names[i], w[i]) for i in np.argsort(w) if w[i] < -0.01],
            'explained_var': eigvals[k] / eigvals.sum(),
        })
    
    return portfolios
