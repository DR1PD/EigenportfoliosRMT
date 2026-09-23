# Eigenportfolios via Random Matrix Theory — metric tables (baseline 2026-07-21)

IS = 2005-01-01…2017-12-31; OOS = 2018-01-01…2024-12-31. All parameters
IS- or rule-provenanced; see results/OOS_DISCIPLINE.md.

```
PERFORMANCE COMPARISON — IS
==============================================================================================================
Strategy                     Return      Vol   Sharpe          SR 95% CI    MaxDD  Sortino   CVaR95
--------------------------------------------------------------------------------------------------------------
RMT Hard Threshold          10.73%  13.23%    0.622      [-0.02, 1.26] -38.05%    0.768 -0.0203
RMT Targeted Shrinkage      10.73%  13.23%    0.622      [-0.02, 1.26] -38.00%    0.769 -0.0203
RMT Constant Residual       10.73%  13.23%    0.622      [-0.02, 1.26] -38.05%    0.768 -0.0203
Ledoit-Wolf Min-Var         10.79%  13.22%    0.627      [-0.02, 1.27] -37.91%    0.775 -0.0202
Raw Sample Min-Var          10.76%  13.22%    0.624      [-0.02, 1.27] -37.93%    0.772 -0.0202
Equal Weight (1/N)          14.72%  19.61%    0.623      [-0.02, 1.27] -48.07%    0.763 -0.0308
S&P 500 (Buy & Hold)         9.93%  18.81%    0.395      [-0.17, 0.96] -55.19%    0.482 -0.0289

==============================================================================================================
```

```
PERFORMANCE COMPARISON — OOS
==============================================================================================================
Strategy                     Return      Vol   Sharpe          SR 95% CI    MaxDD  Sortino   CVaR95
--------------------------------------------------------------------------------------------------------------
RMT Hard Threshold          10.79%  14.31%    0.579      [-0.22, 1.38] -28.37%    0.695 -0.0218
RMT Targeted Shrinkage      10.79%  14.32%    0.579      [-0.22, 1.38] -28.46%    0.695 -0.0218
RMT Constant Residual       10.79%  14.31%    0.579      [-0.22, 1.38] -28.37%    0.695 -0.0218
Ledoit-Wolf Min-Var         10.81%  14.31%    0.581      [-0.22, 1.38] -28.50%    0.696 -0.0218
Raw Sample Min-Var          10.80%  14.32%    0.580      [-0.22, 1.38] -28.59%    0.694 -0.0218
Equal Weight (1/N)          15.14%  18.15%    0.697      [-0.13, 1.52] -31.04%    0.868 -0.0276
S&P 500 (Buy & Hold)        14.84%  19.46%    0.634      [-0.18, 1.45] -33.72%    0.768 -0.0300

Strategy                     Avg Turnover   Total Cost (bps)   Non-opt solves
------------------------------------------------------------------------------
RMT Hard Threshold                 0.2067             464.9            0/217
RMT Targeted Shrinkage             0.2059             463.2            0/217
RMT Constant Residual              0.2067             464.9            0/217
Ledoit-Wolf Min-Var                0.2042             458.7            0/217
Raw Sample Min-Var                 0.2066             464.9            0/217
Equal Weight (1/N)                 0.0000               0.0              n/a

→ During crises (GFC, COVID), correlations spike and dimensionality collapses
  (fewer signal eigenvalues). This is a signature of 'risk-on/risk-off' regimes.

Q-ratio sensitivity (subset analysis):
  n_assets    q=n/T   n_signal   Cond (raw)   Cond (RMT)
-------------------------------------------------------
        50    0.198         12          133           30
       100    0.397         21          371           52
       150    0.595         24         1420           76
       151    0.599         24         1435           77

→ As q → 1, more eigenvalues are noise and RMT cleaning has greater impact.

Regime distribution: {'Bull': np.int64(102), 'Sideways': np.int64(89), 'Bear': np.int64(49)}

================================================================================
REGIME-CONDITIONED NET SHARPE RATIOS
================================================================================
Strategy                         Bull       Bear   Sideways
----------------------------------------------------------
RMT Hard Threshold              0.773      0.634      0.506
RMT Targeted Shrinkage          0.779      0.632      0.504
Ledoit-Wolf Min-Var             0.788      0.631      0.506
Raw Sample Min-Var              0.788      0.631      0.500
Equal Weight (1/N)              0.877      0.746      0.417
S&P 500 (Buy & Hold)      
```

```
PERFORMANCE COMPARISON — Full
==============================================================================================================
Strategy                     Return      Vol   Sharpe          SR 95% CI    MaxDD  Sortino   CVaR95
--------------------------------------------------------------------------------------------------------------
RMT Hard Threshold          10.75%  13.66%    0.604       [0.10, 1.11] -38.05%    0.737 -0.0209
RMT Targeted Shrinkage      10.76%  13.66%    0.604       [0.10, 1.11] -38.00%    0.737 -0.0209
RMT Constant Residual       10.75%  13.66%    0.604       [0.10, 1.11] -38.05%    0.737 -0.0209
Ledoit-Wolf Min-Var         10.80%  13.65%    0.608       [0.11, 1.11] -37.91%    0.741 -0.0208
Raw Sample Min-Var          10.78%  13.66%    0.606       [0.10, 1.11] -37.93%    0.739 -0.0208
Equal Weight (1/N)          14.88%  19.06%    0.650       [0.14, 1.16] -48.07%    0.800 -0.0296
S&P 500 (Buy & Hold)        11.65%  19.04%    0.481       [0.02, 0.94] -55.19%    0.585 -0.0293

==============================================================================================================
```

```
Naive vs iterative edge: λ₊ 1.8188 (one-pass) → 1.2325 (iterative)
Marchenko-Pastur Parameters:
  q = n/T = 0.5992
  σ² (fitted) = 0.3916
  λ₋ (lower edge) = 0.0200
  λ₊ (upper edge) = 1.2325
  Eigenvalues above λ₊: 24 out of 151
  → These are the SIGNAL eigenvalues. Everything else is NOISE.


→ The MP distribution fits the noise bulk remarkably well.
   24 eigenvalues escape above λ₊ = 1.233 — these carry REAL structure.
   The remaining 127 are indistinguishable from random noise.

Signal-Noise Classification:
  Total eigenvalues: 151
  Signal (λ > λ₊):  24
  Noise (λ ≤ λ₊):   127
  λ₊ threshold:      1.2325

Signal eigenvalues:
  λ_1 = 30.087  (19.9% of variance)
  λ_2 = 14.834  (9.8% of variance)
  λ_3 = 9.243  (6.1% of variance)
  λ_4 = 4.749  (3.1% of variance)
  λ_5 = 4.379  (2.9% of variance)
  λ_6 = 3.913  (2.6% of variance)
  λ_7 = 2.973  (2.0% of variance)
  λ_8 = 2.562  (1.7% of variance)
  λ_9 = 2.412  (1.6% of variance)
  λ_10 = 2.368  (1.6% of variance)
  λ_11 = 2.330  (1.5% of variance)
  λ_12 = 2.119  (1.4% of variance)
  λ_13 = 2.037  (1.3% of variance)
  λ_14 = 1.967  (1.3% of variance)
  λ_15 = 1.814  (1.2% of variance)
  λ_16 = 1.768  (1.2% of variance)
  λ_17 = 1.647  (1.1% of variance)
  λ_18 = 1.596  (1.1% of variance)
  λ_19 = 1.533  (1.0% of variance)
  λ_20 = 1.491  (1.0% of variance)
  λ_21 = 1.446  (1.0% of variance)
  λ_22 = 1.368  (0.9% of variance)
  λ_23 = 1.352  (0.9% of variance)
  λ_24 = 1.278  (0.8% of variance)

  Total signal variance: 67.1%
  Total noise variance:  32.9%

Cleaning results:
  Raw sample             condition:     1435.2  trace: 151.00  min λ: 0.0210
  Hard threshold         condition:       76.8  trace: 151.00  min λ: 0.3916
  Targeted shrinkage     condition:      107.3  trace: 151.00  min λ: 0.2804
  Constant residual      condition:       76.8  trace: 151.00  min λ: 0.3916
  Ledoit-Wolf            condition:      207.3  trace: 151.00  min λ: 0.1296


→ RMT cleaning dramatically reduces the condition number,
  making matrix inversion (needed for min-var portfolios) much more stable.


============================================================
Eigenportfolio 1 — 19.9% of variance
============================================================
  Top 5 LONG:  []
  Top 5 SHORT: [('USB', '-0.011'), ('BLK', '-0.011'), ('ITW', '-0.011'), ('PNC', '-0.011'), ('TFC', '-0.011')]

============================================================
Eigenportfolio 2 — 9.8% of variance
============================================================
  Top 5 LONG:  [('AMAT', '0.018'), ('NVDA', '0.016'), ('QCOM', '0.016'), ('MU', '0.015'), ('AMD', '0.013')]
  Top 5 SHORT: [('SO', '-0.018'), ('AEP', '-0.017'), ('DUK', '-0.017'), ('WEC', '-0.017'), ('KO', '-0.016')]

============================================================
Eigenportfolio 3 — 6.1% of variance
============================================================
  Top 5 LONG:  [('ISRG', '0.018'), ('SYK', '0.016'), ('MSFT', '0.015'), ('COST', '0.014'), ('EW', 
```

```
Survivorship decay table — annualized returns:
Period    Survivors EW      SPY      RSP    vs SPY    vs RSP
------------------------------------------------------------
IS             14.72%    9.86%   10.80%     +486bp     +392bp
OOS            15.14%   14.84%   11.69%      +30bp     +345bp
Full           14.88%   11.79%   11.15%     +310bp     +374bp
```

```
Cohorts: near-death 76 (maxDD ≤ -61%), smooth 75 | deep casualties (maxDD ≤ −60%): 82
```

```
Max OOS Sharpe spread across all subsets: 0.009
→ The RMT ≈ LW ≈ raw tie HOLDS across survival profiles and random
  sub-universes — robust to the composition dimension of G-3.
  (Censoring of actually-dead names remains untested without PIT data.)

  T=126  n=50   seed 0: q=0.397 RMT 0.658 LW 0.665 raw 0.666 spread 0.007 sig 9 bad 0

  T=126  n=50   seed 1: q=0.397 RMT 0.573 LW 0.566 raw 0.568 spread 0.007 sig 9 bad 0

  T=126  n=100  seed 0: q=0.794 RMT 0.578 LW 0.555 raw 0.557 spread 0.023 sig 16 bad 0

  T=126  n=100  seed 1: q=0.794 RMT 0.653 LW 0.647 raw 0.647 spread 0.007 sig 17 bad 0

  T=126  n=151  seed 0: q=1.198 RMT 0.596 LW 0.590 raw 0.599 spread 0.009 sig 26 bad 0

  T=252  n=50   seed 0: q=0.198 RMT 0.663 LW 0.667 raw 0.666 spread 0.004 sig 13 bad 0

  T=252  n=50   seed 1: q=0.198 RMT 0.529 LW 0.531 raw 0.535 spread 0.006 sig 13 bad 0

  T=252  n=100  seed 0: q=0.397 RMT 0.548 LW 0.554 raw 0.553 spread 0.006 sig 21 bad 0

  T=252  n=100  seed 1: q=0.397 RMT 0.589 LW 0.596 raw 0.593 spread 0.007 sig 22 bad 0

  T=252  n=151  seed 0: q=0.599 RMT 0.579 LW 0.581 raw 0.580 spread 0.002 sig 29 bad 0

  T=504  n=50   seed 0: q=0.099 RMT 0.638 LW 0.639 raw 0.639 spread 0.001 sig 19 bad 0

  T=504  n=50   seed 1: q=0.099 RMT 0.514 LW 0.511 raw 0.511 spread 0.003 sig 19 bad 0

  T=504  n=100  seed 0: q=0.198 RMT 0.476 LW 0.477 raw 0.477 spread 0.001 sig 28 bad 0

  T=504  n=100  seed 1: q=0.198 RMT 0.596 LW 0.597 raw 0.596 spread 0.001 sig 31 bad 0

  T=504  n=151  seed 0: q=0.3   RMT 0.510 LW 0.507 raw 0.508 spread 0.003 sig 39 bad 0

Grid complete: 15 cells in 6.7 min

VERDICT (read off, not tuned):
  Max estimator spread: 0.023 at T=126, n=100 (q=0.794)
  At the design-target q≈0.79 cells: spreads [0.023, 0.007]
  q ≥ 1 (singular sample) cells: spreads [0.009], non-optimal solves 0/669
→ The RMT ≈ LW ≈ raw tie SURVIVES the entire q range tested, including
  the design-target q≈0.79 and the singular q>1 corner — the tie is a
  property of this universe/period, not of the baseline window choice.

============================================================
PROJECT 2 COMPLETE
============================================================

Universe: 151 assets
Period: 2005-01-01 to 2024-12-31
Key ratio: q = n/T = 0.599
Signal eigenvalues (last window): 24 / 151
Strategies tested: 7

Core deliverable: the eigenvalue spectrum vs Marchenko-Pastur plot
  — the single most important visualization in this project.

Next: Project 3 — Hidden Markov Model Regime Detection
```

```
VERDICT (read off, not tuned):
  Max estimator spread: 0.023 at T=126, n=100 (q=0.794)
  At the design-target q≈0.79 cells: spreads [0.023, 0.007]
  q ≥ 1 (singular sample) cells: spreads [0.009], non-optimal solves 0/669
→ The RMT ≈ LW ≈ raw tie SURVIVES the entire q range tested, including
  the design-target q≈0.79 and the singular q>1 corner — the tie is a
  property of this universe/period, not of the baseline window choice.
```
