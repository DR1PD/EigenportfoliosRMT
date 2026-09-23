"""P2 strategy package (DESIGN Section L via src-layout; ROADMAP 2.2)."""
from .rmt import (compute_eigen, marchenko_pastur_pdf, mp_edges,
                  fit_mp_sigma, fit_mp_sigma_naive, classify_eigenvalues, rolling_mean_corr)
from .cleaning import (clean_covariance_hard_threshold,
                       clean_covariance_targeted_shrinkage,
                       clean_covariance_constant_residual,
                       correlation_to_covariance)
from .eigenportfolios import build_eigenportfolios
from .backtest import min_variance_portfolio, compute_tc, run_rmt_backtest
