"""Min-variance solver, transaction costs, and the rolling RMT backtest
(extracted verbatim from the notebook, 2.2; P2-5 solver statuses included).
"""
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf

from .rmt import compute_eigen, classify_eigenvalues
from .cleaning import (clean_covariance_hard_threshold,
                       clean_covariance_targeted_shrinkage,
                       clean_covariance_constant_residual,
                       correlation_to_covariance)


def min_variance_portfolio(cov_matrix, w_max=0.03):
    """
    Solve constrained minimum variance portfolio.
    Long-only, weight cap, fully invested.
    """
    n = cov_matrix.shape[0]
    w = cp.Variable(n)
    
    objective = cp.Minimize(cp.quad_form(w, cov_matrix, assume_PSD=True))
    constraints = [cp.sum(w) == 1, w >= 0, w <= w_max]
    
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except:
        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)
        except:
            return np.ones(n) / n, {'status': 'failed'}
    
    if prob.status in ['optimal', 'optimal_inaccurate']:
        w_opt = np.maximum(np.array(w.value).flatten(), 0)
        w_opt /= w_opt.sum()
        return w_opt, {'status': prob.status, 'obj': prob.value}
    
    return np.ones(n) / n, {'status': prob.status}


def compute_tc(w_new, w_old, c_prop=0.001, c_impact=0.0003):
    """Transaction cost: proportional + impact."""
    dw = np.abs(w_new - w_old)
    return c_prop * dw.sum() + c_impact * np.sum(dw ** 1.5)


def compute_tc(w_new, w_old, c_prop=0.001, c_impact=0.0003):
    """Transaction cost: proportional + impact."""
    dw = np.abs(w_new - w_old)
    return c_prop * dw.sum() + c_impact * np.sum(dw ** 1.5)


def run_rmt_backtest(returns, config, verbose=True):
    """
    Rolling backtest for all covariance methods simultaneously.
    """
    lookback = config['lookback_days']
    rebal_freq = config['rebal_frequency']
    n = returns.shape[1]
    w_max = config['w_max']
    
    all_dates = returns.index[lookback:]
    rebal_dates = all_dates[::rebal_freq]
    
    # Strategy names
    strategies = ['rmt_hard', 'rmt_shrink', 'rmt_const', 'ledoit_wolf', 'raw_sample', 'equal_weight']
    
    # Storage
    results = {s: {'returns': [], 'dates': [], 'weights': [], 'turnover': [],
                    'costs': [], 'n_signal': [], 'condition': [], 'solve_status': []} for s in strategies}
    
    prev_weights = {s: None for s in strategies}
    
    for idx, rebal_date in enumerate(rebal_dates):
        if idx == len(rebal_dates) - 1:
            break
        
        # Trailing window
        end_loc = returns.index.get_loc(rebal_date)
        start_loc = end_loc - lookback
        window = returns.iloc[start_loc:end_loc]
        
        # ── EIGENDECOMPOSITION ──
        evals, evecs, C_sample, _ = compute_eigen(window)
        q_local = n / lookback
        is_sig, lp, n_sig = classify_eigenvalues(evals, q_local)
        
        # ── COVARIANCE CLEANING ──
        C_hard_t, _, _ = clean_covariance_hard_threshold(evals, evecs, is_sig)
        C_shrink_t, _, _ = clean_covariance_targeted_shrinkage(evals, evecs, is_sig, alpha=0.3)
        C_const_t, _, _ = clean_covariance_constant_residual(evals, evecs, is_sig)
        
        # ── P2-1/P2-2 FIX: correlation → covariance at the handoff ──
        # Cleaned matrices live in CORRELATION space (unit variances) — right
        # for the MP test, wrong for min-var, which must see asset vols.
        # Reset diag(C_clean)=1 (Laloux; cleaning drifts it), then rescale
        # Σ_clean = D·C_clean·D with D = diag of in-window daily vols.
        vols = window.values.std(axis=0)

        def to_cov(C):
            return correlation_to_covariance(C, vols)

        # Ledoit-Wolf now fits raw returns — an actual covariance estimator,
        # matching its label (it previously fit standardized returns).
        C_lw_t = LedoitWolf().fit(window.values).covariance_

        cov_inputs = {
            'rmt_hard': to_cov(C_hard_t),
            'rmt_shrink': to_cov(C_shrink_t),
            'rmt_const': to_cov(C_const_t),
            'ledoit_wolf': C_lw_t,
            'raw_sample': to_cov(C_sample),
        }
        
        # ── PORTFOLIO CONSTRUCTION ──
        for strat in strategies:
            if strat == 'equal_weight':
                w_new = np.ones(n) / n
            else:
                w_new, solve_info = min_variance_portfolio(cov_inputs[strat], w_max)
                results[strat]['solve_status'].append(solve_info.get('status', 'unknown'))
            
            # Transaction costs
            tc = 0
            turnover = 0
            if prev_weights[strat] is not None:
                tc = compute_tc(w_new, prev_weights[strat])
                turnover = np.abs(w_new - prev_weights[strat]).sum()
            
            results[strat]['weights'].append(w_new)
            results[strat]['turnover'].append(turnover)
            results[strat]['costs'].append(tc)
            results[strat]['n_signal'].append(n_sig)
            results[strat]['condition'].append(
                evals[0] / max(evals[-1], 1e-10) if strat == 'raw_sample' 
                else np.linalg.cond(cov_inputs.get(strat, C_sample)))
            
            prev_weights[strat] = w_new.copy()
        
        # ── FORWARD RETURNS ──
        next_rebal = rebal_dates[idx + 1] if idx + 1 < len(rebal_dates) else returns.index[-1]
        fwd_idx = returns.index[(returns.index > rebal_date) & (returns.index <= next_rebal)]
        
        if len(fwd_idx) == 0:
            continue
        
        # G-2 fix: `returns` holds LOG returns (kept for eigen/covariance
        # estimation); portfolio aggregation needs SIMPLE returns (Jensen).
        fwd_rets_simple = np.expm1(returns.loc[fwd_idx])
        
        for strat in strategies:
            w = results[strat]['weights'][-1]
            daily_tc = results[strat]['costs'][-1] / len(fwd_idx) if len(fwd_idx) > 0 else 0
            
            for day in fwd_idx:
                r = fwd_rets_simple.loc[day].values @ w - daily_tc
                results[strat]['returns'].append(r)
                results[strat]['dates'].append(day)
        
        if verbose and idx % 20 == 0:
            print(f"  Rebalance {idx+1}/{len(rebal_dates)-1} at {rebal_date.date()} | signals: {n_sig}")
    
    # Convert to Series
    for strat in strategies:
        results[strat]['returns'] = pd.Series(
            results[strat]['returns'], index=results[strat]['dates'])
    
    return results
