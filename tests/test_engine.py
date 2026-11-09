"""Tests for the C++ engine (cpp/engine.cpp via backtest/fast.py).

The Python implementation in strategy.py is the reference: the C++ engine must
reproduce it to floating-point precision.
"""

import numpy as np
import pandas as pd
import pytest

from rebalancing_frontrun import _engine
from rebalancing_frontrun.backtest.fast import placebo, run_backtest_fast
from rebalancing_frontrun.backtest.strategy import TRADING_DAYS, run_backtest

COLS = ["signal_prev", "direction", "w_es", "w_zn", "gross", "cost", "net"]


@pytest.fixture
def returns():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2019-01-01", "2021-06-15")  # ends mid-month on purpose
    r = pd.DataFrame(
        {"es_ret": rng.normal(0, 0.01, len(days)), "zn_ret": rng.normal(0, 0.003, len(days))},
        index=days,
    )
    r.iloc[100, 0] = np.nan  # a market closure
    return r


def _sharpe(pnl):
    return pnl.mean() / pnl.std() * np.sqrt(TRADING_DAYS)


@pytest.mark.parametrize("timing", ["holding", "paper"])
@pytest.mark.parametrize("costs", [(0.5, 1.5), (0.0, 0.0), (2.0, 3.0)])
def test_cpp_matches_python(returns, timing, costs):
    kwargs = {"es_cost_bps": costs[0], "zn_cost_bps": costs[1], "timing": timing}
    ref = run_backtest(returns, **kwargs)
    fast = run_backtest_fast(returns, **kwargs)

    assert fast.index.equals(ref.index)
    assert (fast["active"] == ref["active"]).all()
    assert (fast["trade_month"] == ref["trade_month"]).all()
    for col in COLS:
        np.testing.assert_allclose(fast[col], ref[col], rtol=0, atol=1e-12, err_msg=col)


@pytest.mark.parametrize("params", [{"window": 3}, {"vol_lookback": 30}, {"leg_vol": 0.2}])
def test_cpp_matches_python_with_other_parameters(returns, params):
    ref = run_backtest(returns, **params)
    fast = run_backtest_fast(returns, **params)
    np.testing.assert_allclose(fast["net"], ref["net"], rtol=0, atol=1e-12)


def test_invalid_timing_raises(returns):
    with pytest.raises(ValueError):
        run_backtest_fast(returns, timing="other")


def test_placebo_equals_strategy_when_every_month_has_five_days():
    """With exactly 5 trading days per month, the only possible 5-day block is the
    whole month, which is also the real strategy's window, so results must match."""
    rng = np.random.default_rng(2)
    months = pd.date_range("2018-01-01", periods=36, freq="MS")
    days = pd.DatetimeIndex([d for m in months for d in pd.bdate_range(m, periods=5)])
    r = pd.DataFrame(
        {"es_ret": rng.normal(0, 0.01, len(days)), "zn_ret": rng.normal(0, 0.003, len(days))},
        index=days,
    )
    real = run_backtest(r)
    split_date = days[89]
    pl = placebo(r, n_sims=3, seed=0, in_sample_end=split_date)

    np.testing.assert_allclose(pl["full"], _sharpe(real["net"]), atol=1e-12)
    np.testing.assert_allclose(pl["in_sample"], _sharpe(real.loc[:split_date, "net"]), atol=1e-12)
    np.testing.assert_allclose(pl["out_of_sample"], _sharpe(real.loc[days[90]:, "net"]), atol=1e-12)


def test_placebo_is_reproducible(returns):
    a = placebo(returns, n_sims=200, seed=7)
    b = placebo(returns, n_sims=200, seed=7)
    c = placebo(returns, n_sims=200, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_placebo_shape_and_split(returns):
    pl = placebo(returns, n_sims=50, in_sample_end="2020-06-30")
    assert list(pl.columns) == ["full", "in_sample", "out_of_sample"]
    assert len(pl) == 50
    assert np.isfinite(pl.to_numpy()).all()
    assert placebo(returns, n_sims=5)["out_of_sample"].isna().all()  # no split -> no OOS


def test_engine_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        _engine.backtest(np.zeros(10), np.zeros(9), np.zeros(10))
