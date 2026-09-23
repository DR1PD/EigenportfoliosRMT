"""Smoke + invariants for eigenrmt.run_rmt_backtest on synthetic data."""
import numpy as np
import pandas as pd

from eigenrmt import run_rmt_backtest

CFG = {'lookback_days': 60, 'rebal_frequency': 21, 'w_max': 0.5}


def _toy_returns(T=300, n=8, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(0.0002, 0.01, (T, n)),
                     index=pd.date_range('2015-01-01', periods=T, freq='B'))
    return X


def test_backtest_runs_and_reports_statuses():
    bt = run_rmt_backtest(_toy_returns(), CFG, verbose=False)
    assert set(bt) == {'rmt_hard', 'rmt_shrink', 'rmt_const',
                       'ledoit_wolf', 'raw_sample', 'equal_weight'}
    for s in ['rmt_hard', 'ledoit_wolf', 'raw_sample']:
        statuses = bt[s]['solve_status']
        assert len(statuses) > 0                      # P2-5: statuses recorded
        assert all(st in ('optimal', 'optimal_inaccurate') for st in statuses)
        assert np.isfinite(bt[s]['returns']).all()


def test_equal_weight_is_mean_of_simple_returns():
    """EW never trades (zero cost), so its daily return must equal the mean
    of the assets' SIMPLE returns — the G-2 aggregation contract."""
    X = _toy_returns()
    bt = run_rmt_backtest(X, CFG, verbose=False)
    ew = bt['equal_weight']['returns']
    expected = np.expm1(X).mean(axis=1).reindex(ew.index)
    assert np.allclose(ew.values, expected.values, atol=1e-12)
