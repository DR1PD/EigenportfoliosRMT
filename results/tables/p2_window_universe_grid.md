# P2 window × universe grid — the q-sensitivity study (ROADMAP 3.2, 2026-07-21)

Pre-stated question: does RMT≈LW≈raw survive across q, or separate at high q
where cleaning should matter most? Verdict: the tie SURVIVES everywhere tested
(max spread 0.023, at the design-target q=0.794 cell; singular q=1.198: 0.009,
0/669 non-optimal solves). Baseline T=252 / full universe unchanged.

| T | n | seed | q | OOS SR RMT | OOS SR LW | OOS SR raw | OOS spread | avg #signals | non-opt |
|---|---|---|---|---|---|---|---|---|---|
| 504 | 50 | 0 | 0.099 | 0.638 | 0.639 | 0.639 | 0.001 | 19 | 0/615 |
| 504 | 50 | 1 | 0.099 | 0.514 | 0.511 | 0.511 | 0.003 | 19 | 0/615 |
| 252 | 50 | 0 | 0.198 | 0.663 | 0.667 | 0.666 | 0.004 | 13 | 0/651 |
| 252 | 50 | 1 | 0.198 | 0.529 | 0.531 | 0.535 | 0.006 | 13 | 0/651 |
| 504 | 100 | 0 | 0.198 | 0.476 | 0.477 | 0.477 | 0.001 | 28 | 0/615 |
| 504 | 100 | 1 | 0.198 | 0.596 | 0.597 | 0.596 | 0.001 | 31 | 0/615 |
| 504 | 151 | 0 | 0.300 | 0.510 | 0.507 | 0.508 | 0.003 | 39 | 0/615 |
| 126 | 50 | 0 | 0.397 | 0.658 | 0.665 | 0.666 | 0.007 | 9 | 0/669 |
| 126 | 50 | 1 | 0.397 | 0.573 | 0.566 | 0.568 | 0.007 | 9 | 0/669 |
| 252 | 100 | 0 | 0.397 | 0.548 | 0.554 | 0.553 | 0.006 | 21 | 0/651 |
| 252 | 100 | 1 | 0.397 | 0.589 | 0.596 | 0.593 | 0.007 | 22 | 0/651 |
| 252 | 151 | 0 | 0.599 | 0.579 | 0.581 | 0.580 | 0.002 | 29 | 0/651 |
| 126 | 100 | 0 | 0.794 | 0.578 | 0.555 | 0.557 | 0.023 | 16 | 0/669 |
| 126 | 100 | 1 | 0.794 | 0.653 | 0.647 | 0.647 | 0.007 | 17 | 0/669 |
| 126 | 151 | 0 | 1.198 | 0.596 | 0.590 | 0.599 | 0.009 | 26 | 0/669 |

Footnote (3.3 sweep): all Sharpe columns are OOS (2018–2024), shown for the
pre-stated tie question only — no parameter was selected on them; the
baseline window/universe was fixed before the grid ran.
