Eigenportfolio Construction via RMT  |  D. Colindres

PROJECT SPECIFICATION

**Eigenportfolio Construction via Random Matrix Theory**

Separating Signal from Noise in Large Covariance Matrices

**David Colindres**

B.S./M.S. Industrial & Systems Engineering

University of Oklahoma

*Prepared for MFE / MSCF / MSFE Admissions Portfolio — Project 2 of 3*

March 2026

# A. Project Title

***Eigenportfolio Construction via Random Matrix Theory: Separating Signal from Noise in Large Covariance Matrices***

# B. Project Thesis

The sample covariance matrix is the foundation of nearly all portfolio optimization, risk modeling, and factor analysis in quantitative finance. But most of it is noise. When the number of assets n is comparable to the number of observations T, random matrix theory (RMT) shows that the bulk of the eigenvalue spectrum is indistinguishable from what a purely random matrix would produce. Only the eigenvalues that escape the Marchenko–Pastur distribution carry genuine information about the correlation structure of asset returns.

This project implements a complete pipeline that computes the empirical eigenvalue spectrum of a large equity covariance matrix, fits the Marchenko–Pastur distribution as a null model, identifies statistically significant eigenvalues, removes noise eigenvalues, constructs cleaned covariance matrices, and builds eigenportfolios from the significant eigenvectors. The central question is whether RMT-cleaned covariance matrices produce portfolios with superior out-of-sample risk-adjusted performance compared to portfolios built from the raw sample covariance.

# C. Why This Project Is Admissions-Relevant

**Mathematical maturity. **This project lives at the intersection of linear algebra, probability theory, and statistics. Computing eigendecompositions, fitting theoretical spectral distributions, performing hypothesis testing on eigenvalues, and understanding the Marchenko–Pastur law all require serious quantitative preparation — the kind that Berkeley MFE, CMU MSCF, Columbia FE, and Princeton MFin/ORFE explicitly screen for.

**Statistical sophistication. **Most applicants know sample covariance. Very few understand why it breaks down in high dimensions, how to diagnose the problem (via the eigenvalue spectrum), or how to fix it (via RMT cleaning). This places the project above standard coursework.

**Programming competence. **The implementation requires eigendecomposition on large matrices, spectral density estimation, distribution fitting, matrix reconstruction, rolling-window portfolio construction, and detailed 3D visualization — all from scratch in Python. [As implemented: MP fit, cleaning, eigenportfolios, and backtest are hand-rolled; the Ledoit–Wolf benchmark deliberately uses scikit-learn as the reference implementation.]

**Financial relevance. **Covariance estimation is the single most important input to portfolio construction, risk management, and derivatives pricing. A project that directly attacks noise contamination in covariance speaks to the central problem of quantitative finance.

**ISE alignment. **Linear algebra, multivariate statistics, dimensionality reduction, and data-driven decision-making under uncertainty are core ISE competencies. This project applies them with a rigorous noise model from mathematical physics.

**Intellectual signal. **Random Matrix Theory is a topic most undergraduates never encounter. Building a project around it signals genuine intellectual curiosity — the obsession signal that quant firms and admissions committees look for.

# D. Research Question

*Do portfolios constructed from RMT-cleaned covariance matrices — where noise eigenvalues are identified via the Marchenko–Pastur distribution and either removed, shrunk, or replaced — produce statistically superior out-of-sample Sharpe ratios, lower realized volatility, and more stable weight allocations compared to portfolios constructed from the raw sample covariance, Ledoit–Wolf shrinkage, and equal-weight baselines?*

# E. Quant Finance Concepts Involved

- Sample covariance estimation and its breakdown in high dimensions

- Eigenvalue decomposition of covariance matrices

- Random Matrix Theory: Marchenko–Pastur distribution

- Signal-noise separation in the eigenvalue spectrum

- Covariance matrix cleaning / denoising (Laloux et al., 1999; Bun, Bouchaud, Potters, 2017)

- Eigenportfolio construction (principal portfolios)

- Minimum variance portfolio optimization

- Ledoit–Wolf shrinkage estimation (benchmark)

- Portfolio turnover and weight stability analysis

- Risk decomposition: principal component risk attribution

- Explained variance ratio and effective dimensionality

- Condition number diagnostics and matrix invertibility

- Rolling-window out-of-sample evaluation

- Tracy–Widom test for borderline eigenvalues

# F. Math/Statistics Requirements

The following mathematics must appear explicitly as implemented computation:

## 1. Eigendecomposition of the Sample Covariance

Σ = (1/T) XᵀX = V Λ Vᵀ

where X is the T×n matrix of demeaned returns, Λ = diag(λ₁, ..., λₙ) is the eigenvalue matrix, and V contains the eigenvectors.

## 2. Marchenko–Pastur Distribution

ρ_MP(λ) = (T / 2πσ²q) · √((λ₊ - λ)(λ - λ₋)) / λ

λ± = σ²(1 ± √q)²     where q = n/T

This is the theoretical eigenvalue density for a random correlation matrix. Eigenvalues exceeding λ₊ carry statistically significant information.

## 3. Signal/Noise Classification

An eigenvalue λᵢ is classified as noise if λᵢ ≤ λ₊. Eigenvalues exceeding λ₊ represent the true correlation structure. The number of significant eigenvalues is the effective rank of the correlation matrix.

## 4. Covariance Cleaning Methods

**Method A — Hard thresholding: **Replace all noise eigenvalues with their average, preserving the trace. Reconstruct: Σ_clean = V Λ_clean Vᵀ

**Method B — Targeted shrinkage: **Shrink noise eigenvalues toward a target while preserving signal eigenvalues intact.

**Method C — Constant-residual replacement: **Replace noise eigenvalues with a constant chosen to preserve the total variance (trace of the matrix).

## 5. Eigenportfolio Construction

wₖ = vₖ / (1ᵀ|vₖ|)

The k-th eigenportfolio has weights proportional to the k-th eigenvector. The first eigenportfolio approximates the market portfolio; higher eigenportfolios capture sector, style, or idiosyncratic structure.

## 6. Minimum Variance Portfolio

w_MV = Σ⁻¹·1 / (1ᵀΣ⁻¹1)

Computed using both raw and cleaned covariance matrices [RESOLVED 1.3: cleaned CORRELATION matrices are reset to unit diagonal (Laloux) and rescaled Σ = D·C_clean·D with in-window vols before optimization — `tests/test_p2_covariance.py`] to test whether denoising improves out-of-sample risk minimization.

## 7. Explained Variance Ratio

EVRₖ = λₖ / Σᵢ λᵢ

Cumulative EVR measures the effective dimensionality of the return-generating process.

## 8. Condition Number

κ(Σ) = λ_max / λ_min

Diagnostic for matrix invertibility. High condition numbers destabilize portfolio optimization. RMT cleaning should dramatically reduce the condition number.

# G. Data

## Primary Data Sources

| **Data** | **Source** | **Variables** | **Frequency** |
| --- | --- | --- | --- |
| US equity prices | yfinance | Adj. close for 200 equities | Daily |
| S&P 500 benchmark | yfinance (SPY) | Adj. close | Daily |
| Risk-free rate | Kenneth French | 1-month T-bill | Daily |

## Universe Construction

Investment universe [REVISED 2026-07-19; corrected 2026-07-20]: hand-picked 2024-era large/mid caps — a **survivor universe**. The list actually contains **180 tickers** (the design's 200 was never reached; the `[:200]` slice was a no-op); 172 have usable data and **151 survive cleaning** (see `docs/LIMITATIONS.md` §1). A larger universe is critical — the ratio q = n/T must be large enough that the Marchenko–Pastur distribution is a meaningful benchmark. With n = 200 and T = 252, q ≈ 0.79 was the design target; **[REVISED 2026-07-19] the realized universe is 151 → q = 0.599**, a milder regime (narrower MP band, less for cleaning to do) — a direct consequence of survivorship-driven universe shrinkage.

Sample period: January 2005 – December 2024 (20 years). In-sample: 2005–2017. Out-of-sample: 2018–2024.

## Data Cleaning

- Adjust for splits/dividends (use adjusted close)

- Remove assets with >5% missing data in any rolling window

- Forward-fill gaps ≤3 days; drop longer gaps

- Standardize returns to zero mean and unit variance before eigendecomposition

- Winsorize daily returns at ±5 standard deviations

# H. Modeling Pipeline

- **Step 1: Data Ingestion. **Pull adjusted close for 200 equities via yfinance. Compute log returns, apply cleaning, store in Parquet.

- **Step 2: Eigendecomposition. **At each rebalancing date, compute sample correlation matrix from trailing 252-day window. Compute eigenvalues and eigenvectors, sorted descending.

- **Step 3: Marchenko–Pastur Fit. **Compute q = n/T. Estimate σ² [RESOLVED 1.4: iterative Laloux-style trace renormalization σ² = (1 − Σ_signal λ/n)/(1 − k/n) to a stable signal set; naive one-pass retained only for the signature-plot comparison]. Compute MP edges λ±. Generate theoretical MP density. Overlay empirical histogram against theory.

- **Step 4: Signal/Noise Classification. **Classify eigenvalues as signal (λ > λ₊) or noise (λ ≤ λ₊). Record number of significant eigenvalues (effective rank).

- **Step 5: Covariance Cleaning. **Implement three cleaning methods. Reconstruct cleaned covariance. Verify PSD and trace preservation.

- **Step 6: Eigenportfolio Construction. **Build eigenportfolios from significant eigenvectors. Analyze composition (sector loadings). Interpret first 3–5 eigenportfolios economically.

- **Step 7: Portfolio Construction. **Build min-var portfolios using raw, RMT-cleaned, Ledoit–Wolf covariance, plus equal-weight. Apply long-only and weight-cap constraints.

- **Step 8: Rolling Backtest. **Rebalance monthly. Walk-forward: only trailing data for estimation. Track gross/net returns, turnover, weight stability.

- **Step 9: Performance Evaluation. **Compute all metrics from Section J. Generate comparison tables and visualizations.

- **Step 10: Sensitivity

> **[STATUS 2026-07-19]** Implemented: last-window q-sensitivity table and the 1.6 survival-profile/random-subset OOS comparison. Full window-length (126/504) and universe-size rolling grids **[PLANNED — Phase 3.2]**. & Robustness. **Vary universe size (50, 100, 150, 200), window length (126, 252, 504), cleaning method. Regime-conditioned evaluation.

# I. Benchmarks

| **#** | **Benchmark** | **Description** | **Purpose** |
| --- | --- | --- | --- |
| 1 | **Raw Sample Cov → Min-Var** | Min variance using unmodified sample covariance | Tests whether denoising adds value |
| 2 | **Ledoit–Wolf → Min-Var** | Min variance using LW shrunk covariance | Industry-standard alt. RMT vs linear shrinkage |
| 3 | **Equal-Weight (1/N)** | Uniform allocation, rebalanced monthly | Naive baseline (DeMiguel et al., 2009) |
| 4 | **S&P 500 (Buy & Hold)** | Market-cap weighted index | Passive benchmark |
| 5 | **Top-K Eigenportfolio Blend** | Equal blend of top K eigenportfolios (no cleaning) [PLANNED — Phase 3; not in current backtest] | Tests eigenportfolios without denoising |

# J. Evaluation Framework

## Performance Metrics

| **Metric** | **Description** |
| --- | --- |
| **Annualized return (net)** | Geometric mean of net daily returns, annualized |
| **Annualized volatility** | Std dev of daily returns × √252 |
| **Realized / predicted vol ratio** | OOS realized vol vs in-sample predicted vol — key for covariance quality |
| **Net Sharpe ratio** | (Net return - RF) / Volatility |
| **Sharpe ratio 95% CI** | Lo (2002) standard error |
| **Maximum drawdown** | Largest peak-to-trough decline |
| **Average monthly turnover** | Mean L1 weight change at rebalancing |
| **Weight stability** | Std dev of weights across rebalancing dates |
| **Portfolio HHI** | Herfindahl concentration index |

## Covariance Quality Metrics

| **Metric** | **Description** |
| --- | --- |
| **Condition number** | λ_max / λ_min — lower is better for optimization stability |
| **Frobenius distance** | Distance to next-period realized covariance (prediction accuracy) |
| **Significant eigenvalues** | Number exceeding MP upper edge (effective dimensionality) |
| **Trace preservation error** | Abs difference in trace before/after cleaning |
| **Eigenvalue prediction MSE** | MSE of predicted vs realized eigenvalue spectrum |

## Robustness Checks

- Universe size: n ∈ {50, 100, 150, 200} — how does signal/noise ratio change?

- Estimation window: T ∈ {126, 252, 504} — how does q = n/T affect results?

- Cleaning method comparison: hard threshold vs targeted shrinkage vs Ledoit–Wolf

- Regime-conditioned analysis: bull, bear, sideways

- Eigenvalue stability: how stable are significant eigenvalues across rolling windows?

## Out-of-Sample Discipline

All estimation uses trailing data only. The cleaning threshold is computed fresh at each rebalancing date. No retroactive parameter tuning. IS and OOS results reported separately.

# K. Failure Modes

- **Marchenko–Pastur fit may be poor. **MP assumes i.i.d. returns with constant variance. Real returns exhibit volatility clustering, fat tails, and serial correlation. If the empirical spectrum deviates systematically from MP even in the noise region, the threshold λ₊ may be miscalibrated.

- **Too few significant eigenvalues. **If only 1–2 eigenvalues exceed λ₊, the cleaned covariance is extremely low-rank. Portfolios will be poorly diversified and dominated by the market factor.

- **Eigenvector instability. **Even when eigenvalues are stable, eigenvectors can rotate substantially between windows, especially when eigenvalues are close. This causes high turnover.

- **Cleaning destroys useful structure. **Hard thresholding assumes binary signal/noise. Eigenvalues just below λ₊ may carry partial information. Replacing them with a constant discards potentially useful structure.

- **Ledoit–Wolf may simply be better. **RMT cleaning is more principled in theory, but LW is simpler and widely used for good reason. If it matches or beats RMT, the extra complexity is not justified — an honest finding.

- **Survivorship bias. **Fixed ticker list, no delisting handling. Biases all strategies equally but overstates absolute performance. [STATUS 2026-07-19: measured — selection premium ≈ +350–390 bp/yr vs RSP, stable across IS/OOS (the 'equally' assumption is now data, not hope); mitigated via population reframe and survival-profile sensitivity (tie robust, max spread 0.009); PIT fix gated on WRDS. `docs/LIMITATIONS.md` §1.]

- **Numerical issues with near-singular matrices. **Raw sample covariance with q ≈ 0.8 is poorly conditioned. RMT cleaning should improve this, but edge cases exist.

- **Non-stationarity. **The correlation structure is not stationary. Rolling windows capture this partially, but structural breaks may cause RMT cleaning to lag.

# L. Final GitHub Repository Structure

> **[STATUS 2026-07-19]** Same as P1: `src/` split, config, notebook series, results tree **[PLANNED — Phase 2/3]**; current reality is the executed `notebooks/exploration.ipynb` + repo-level tests.

eigenportfolio-rmt/
├── README.md
├── requirements.txt
├── environment.yml
├── config.yaml
├── data_dictionary.md
├── LICENSE
├── data/
│   ├── raw/ | processed/ | README.md
├── src/
│   ├── data_loader.py    │ covariance.py
│   ├── rmt.py            │ cleaning.py
│   ├── eigenportfolios.py│ optimizer.py
│   ├── backtester.py     │ benchmarks.py
│   ├── metrics.py        │ regime.py
│   └── utils.py
├── notebooks/
│   ├── 01 through 08 (exploration → regime analysis)
├── results/ (figures/ tables/ logs/)
├── tests/ (test_rmt.py, test_cleaning.py, test_metrics.py)
└── docs/ (technical_appendix.pdf, references.bib)

# M. README Outline

- 1. Title and One-Line Summary

- 2. Executive Summary: noise problem, RMT solution, key finding

- 3. Research Question

- 4. Key Results: summary table + eigenvalue spectrum vs MP plot

- 5. Methodology: eigendecomposition, MP fit, cleaning, portfolio construction

- 6. Repository Structure

- 7. How to Reproduce

- 8. Key Visualizations: eigenvalue spectrum vs MP, cumulative returns, eigenvector heatmap, condition number improvement

- 9. Assumptions and Limitations

- 10. References

- 11. Author

# N. Resume / Interview Framing

- Implemented a Random Matrix Theory pipeline to separate signal from noise in the eigenvalue spectrum of large equity covariance matrices (151 realized survivor assets × 252 days [REVISED 2026-07-19]), identifying statistically significant eigenvalues via the Marchenko–Pastur distribution and constructing denoised covariance estimators.

- Built and compared three covariance cleaning methods — hard eigenvalue thresholding, targeted shrinkage, and Ledoit–Wolf — evaluating each on out-of-sample portfolio performance, covariance prediction accuracy, and matrix conditioning across a 20-year backtest.

- Constructed and economically interpreted eigenportfolios from the significant eigenvectors, identifying market, sector-rotation, and style factors emerging from pure statistical decomposition of the correlation structure.

- [REVISED 2026-07-19] Result: RMT-cleaned, Ledoit–Wolf, and raw-sample min-var are statistically **indistinguishable** OOS (Sharpe spread ≤ 0.021; robust across survival-profile cohorts and random sub-universes, max spread 0.009) — that is the honest finding; weight-stability comparison [PLANNED — Phase 3], with all results validated under regime-conditioned evaluation and sensitivity analysis.

# O. Stretch Extensions

## Extension 1: Nonlinear Shrinkage via Ledoit–Péché Oracle

Implement the optimal rotationally invariant estimator (Ledoit & Péché, 2011), which applies a nonlinear function to each eigenvalue based on the Stieltjes transform. This is theoretically optimal under the random matrix model and strictly dominates both hard thresholding and linear shrinkage.

*Why this impresses: Graduate-level material that most MFE students encounter in their second semester. Having it pre-admission signals serious preparation.*

## Extension 2: Dynamic Dimensionality Tracking

Track how the number of significant eigenvalues changes over time. During crises, correlations spike and dimensionality collapses. During calm markets, more eigenvalues emerge. This connects RMT to regime detection (Project 3).

*Why this impresses: Bridges statistical structure and market dynamics, showing the correlation structure is not static.*

## Extension 3: Factor-Mimicking Portfolios

Map significant eigenvectors to interpretable economic factors by regressing each eigenportfolio on known factors (FF3, momentum, sector indices). Tests whether data-driven PCA factors align with or differ from standard academic factors.

*Why this impresses: Connects two traditions — statistical factor models (APT/PCA) and economic factor models (Fama–French) — and tests their consistency empirically.*

# Self-Evaluation Rubric

| **Dimension** | **Score** | **Justification** |
| --- | --- | --- |
| **Finance relevance** | 5/5 | Covariance estimation is the foundation of portfolio construction and risk management. |
| **Mathematical rigor** | 5/5 | Eigendecomposition, MP distribution, spectral analysis, matrix reconstruction — all from scratch. |
| **Programming rigor** | 5/5 | Custom RMT pipeline, multiple cleaning methods, rolling backtest, 3D visualizations. |
| **Statistical validity** | 5/5 | Five benchmarks, IS/OOS, covariance quality metrics, sensitivity to n and T. |
| **Professional presentation** | 5/5 | Eigenvalue spectrum vs MP is a visually striking signature plot. |
| **Interview defensibility** | 5/5 | Can explain why sample cov fails, what MP means, signal vs noise, cleaning mechanics. |
| **Originality** | 5/5 | Very few MFE applicants attempt RMT. Well above standard coursework. |
| **Graduate readiness** | 5/5 | Directly prepares for probability, linear algebra, statistics, comp finance courses. |

# Key Python Libraries

| **Library** | **Role** |
| --- | --- |
| **NumPy / SciPy** | Core: eigendecomposition, MP distribution fitting, matrix operations, spectral density |
| **yfinance** | Data ingestion for 200-asset universe |
| **scikit-learn** | Ledoit–Wolf covariance estimator (benchmark) |
| **PyPortfolioOpt** | Benchmark portfolio construction (min-var, equal-weight) |
| **matplotlib / plotly** | Eigenvalue spectra, 3D surfaces, heatmaps, cumulative returns |
| **statsmodels** | OLS regressions for eigenvector–factor mapping (Extension 3) |
| **pandas** | Time-series manipulation, rolling windows, alignment |

# References

**[1] **Laloux, L., Cizeau, P., Bouchaud, J.-P., & Potters, M. (1999). Noise dressing of financial correlation matrices. Physical Review Letters, 83(7), 1467–1470.

**[2] **Bun, J., Bouchaud, J.-P., & Potters, M. (2017). Cleaning large correlation matrices. Physics Reports, 666, 1–109.

**[3] **Marchenko, V. A., & Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices.

**[4] **Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. JMVA, 88(2), 365–411.

**[5] **Ledoit, O., & Péché, S. (2011). Eigenvectors of some large sample covariance matrix ensembles.

**[6] **Plerou, V., et al. (2002). Random matrix approach to cross correlations in financial data. Phys. Rev. E, 65(6).

**[7] **DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal vs naive diversification. RFS, 22(5), 1915–1953.

**[8] **Lo, A. (2002). The statistics of Sharpe ratios. FAJ, 58(4), 36–52.

**[9] **Tracy, C. A., & Widom, H. (1994). Level-spacing distributions and the Airy kernel.

**[10] **Potters, M., Bouchaud, J.-P., & Laloux, L. (2005). Financial applications of random matrix theory.