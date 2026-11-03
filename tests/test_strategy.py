"""Tests for the front-running backtest (backtest/strategy.py)."""

import numpy as np
import pandas as pd
import pytest

from rebalancing_frontrun.backtest.strategy import run_backtest, summarize


@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2020-01-01", "2020-12-31")
    return pd.DataFrame(
        {"es_ret": rng.normal(0, 0.01, len(days)), "zn_ret": rng.normal(0, 0.003, len(days))},
        index=days,
    )


def test_no_look_ahead(returns):
    """Changing day t's returns must not change the position held on day t."""
    bt = run_backtest(returns)
    t = bt.index[bt["active"]][40]
    shocked = returns.copy()
    shocked.loc[t] = [0.5, -0.5]
    bt2 = run_backtest(shocked)
    assert bt.loc[t, "w_es"] == pytest.approx(bt2.loc[t, "w_es"])
    assert bt.loc[t, "w_zn"] == pytest.approx(bt2.loc[t, "w_zn"])


def test_active_only_in_last_five_days(returns):
    bt = run_backtest(returns)
    month = bt.index.to_period("M")
    days_to_end = bt.groupby(month).cumcount(ascending=False)
    assert (days_to_end[bt["active"]] < 5).all()
    assert (bt.loc[~bt["active"], ["w_es", "w_zn"]] == 0).all().all()


def test_direction_opposes_signal(returns):
    """Equities overweight (signal > 0) -> short ES, long ZN."""
    bt = run_backtest(returns)
    a = bt[bt["active"]]
    assert (np.sign(a["direction"]) == -np.sign(a["signal_prev"])).all()
    assert (np.sign(a["w_es"]) == np.sign(a["direction"])).all()
    assert (np.sign(a["w_zn"]) == -np.sign(a["direction"])).all()


def test_zero_costs_give_net_equal_gross(returns):
    bt = run_backtest(returns, es_cost_bps=0, zn_cost_bps=0)
    assert np.allclose(bt["net"], bt["gross"])


def test_costs_only_reduce_pnl(returns):
    bt = run_backtest(returns)
    assert (bt["cost"] >= 0).all()
    assert bt["net"].sum() < bt["gross"].sum()


def test_paper_timing_reverses_on_first_day_of_month(returns):
    bt = run_backtest(returns, timing="paper")
    first_days = bt.groupby(bt.index.to_period("M")).head(1).index[1:]
    for d in first_days:
        if bt.loc[d, "active"] and bt.loc[d, "signal_prev"] != 0:
            # Opposite of the front-running direction, and booked to the prior month's trade
            assert np.sign(bt.loc[d, "direction"]) == np.sign(bt.loc[d, "signal_prev"])
            assert bt.loc[d, "trade_month"] == d.to_period("M") - 1


def test_invalid_timing_raises(returns):
    with pytest.raises(ValueError):
        run_backtest(returns, timing="other")


def test_summary_counts_one_trade_per_month(returns):
    bt = run_backtest(returns)
    traded = bt[bt["active"]].index.to_period("M").nunique()
    assert summarize(bt).loc["trades", "net"] == traded
