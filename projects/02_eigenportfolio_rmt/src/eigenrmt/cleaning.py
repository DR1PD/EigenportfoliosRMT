"""Covariance cleaning variants + the correlation-to-covariance handoff.

Cleaning operates in CORRELATION space (right for the MP test); optimization
needs covariance, so correlation_to_covariance applies the Laloux diagonal
reset (P2-2) and rescales by asset vols (P2-1): Sigma = D C_clean D.
"""
import numpy as np


def correlation_to_covariance(C_clean, vols):
    """P2-1/P2-2 handoff: reset diag(C_clean)=1 (cleaning drifts it), then
    Sigma = D C_clean D with D = diag(vols). Input C_clean is not mutated."""
    C = C_clean.copy()
    np.fill_diagonal(C, 1.0)
    D = np.diag(np.asarray(vols))
    return D @ C @ D


def clean_covariance_hard_threshold(eigvals, eigvecs, is_signal):
    """
    Method A: Replace noise eigenvalues with their mean.
    Preserves trace. Laloux et al. (1999).
    """
    cleaned = eigvals.copy()
    noise_mean = eigvals[~is_signal].mean() if (~is_signal).any() else 1.0
    cleaned[~is_signal] = noise_mean
    
    C_clean = eigvecs @ np.diag(cleaned) @ eigvecs.T
    
    return C_clean, cleaned, {'method': 'hard_threshold', 'noise_replacement': noise_mean}


def clean_covariance_targeted_shrinkage(eigvals, eigvecs, is_signal, alpha=0.5):
    """
    Method B: Shrink noise eigenvalues toward their mean.
    alpha=0 → full replacement (same as Method A)
    alpha=1 → no change (same as raw)
    """
    cleaned = eigvals.copy()
    noise_mean = eigvals[~is_signal].mean() if (~is_signal).any() else 1.0
    cleaned[~is_signal] = alpha * eigvals[~is_signal] + (1 - alpha) * noise_mean
    
    C_clean = eigvecs @ np.diag(cleaned) @ eigvecs.T
    
    return C_clean, cleaned, {'method': 'targeted_shrinkage', 'alpha': alpha}


def clean_covariance_constant_residual(eigvals, eigvecs, is_signal):
    """
    Method C: Replace noise eigenvalues with a constant that
    preserves the trace of the original matrix exactly.
    """
    cleaned = eigvals.copy()
    n_noise = (~is_signal).sum()
    
    if n_noise > 0:
        # Total trace to preserve
        target_trace = eigvals.sum()
        signal_trace = eigvals[is_signal].sum()
        noise_replacement = (target_trace - signal_trace) / n_noise
        cleaned[~is_signal] = noise_replacement
    
    C_clean = eigvecs @ np.diag(cleaned) @ eigvecs.T
    
    trace_error = abs(C_clean.trace() - eigvals.sum())
    
    return C_clean, cleaned, {'method': 'constant_residual', 'trace_error': trace_error}


def clean_covariance_targeted_shrinkage(eigvals, eigvecs, is_signal, alpha=0.5):
    """
    Method B: Shrink noise eigenvalues toward their mean.
    alpha=0 → full replacement (same as Method A)
    alpha=1 → no change (same as raw)
    """
    cleaned = eigvals.copy()
    noise_mean = eigvals[~is_signal].mean() if (~is_signal).any() else 1.0
    cleaned[~is_signal] = alpha * eigvals[~is_signal] + (1 - alpha) * noise_mean
    
    C_clean = eigvecs @ np.diag(cleaned) @ eigvecs.T
    
    return C_clean, cleaned, {'method': 'targeted_shrinkage', 'alpha': alpha}


def clean_covariance_constant_residual(eigvals, eigvecs, is_signal):
    """
    Method C: Replace noise eigenvalues with a constant that
    preserves the trace of the original matrix exactly.
    """
    cleaned = eigvals.copy()
    n_noise = (~is_signal).sum()
    
    if n_noise > 0:
        # Total trace to preserve
        target_trace = eigvals.sum()
        signal_trace = eigvals[is_signal].sum()
        noise_replacement = (target_trace - signal_trace) / n_noise
        cleaned[~is_signal] = noise_replacement
    
    C_clean = eigvecs @ np.diag(cleaned) @ eigvecs.T
    
    trace_error = abs(C_clean.trace() - eigvals.sum())
    
    return C_clean, cleaned, {'method': 'constant_residual', 'trace_error': trace_error}


def clean_covariance_constant_residual(eigvals, eigvecs, is_signal):
    """
    Method C: Replace noise eigenvalues with a constant that
    preserves the trace of the original matrix exactly.
    """
    cleaned = eigvals.copy()
    n_noise = (~is_signal).sum()
    
    if n_noise > 0:
        # Total trace to preserve
        target_trace = eigvals.sum()
        signal_trace = eigvals[is_signal].sum()
        noise_replacement = (target_trace - signal_trace) / n_noise
        cleaned[~is_signal] = noise_replacement
    
    C_clean = eigvecs @ np.diag(cleaned) @ eigvecs.T
    
    trace_error = abs(C_clean.trace() - eigvals.sum())
    
    return C_clean, cleaned, {'method': 'constant_residual', 'trace_error': trace_error}
