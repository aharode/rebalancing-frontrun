"""Tests for the trading calendar and Calendar rebalancing signal (signals/calendar.py)."""

import numpy as np
import pandas as pd
import pytest

from rebalancing_frontrun.signals.calendar import calendar_signal, trading_calendar


def _returns(index, es=0.0, zn=0.0):
    return pd.DataFrame({"es_ret": es, "zn_ret": zn}, index=pd.DatetimeIndex(index))


def test_month_end_uses_last_trading_day_in_the_data():
    # March 2024: Good Friday (Mar 29) is a holiday, so the 28th is the month-end
    days = pd.bdate_range("2024-03-20", "2024-04-05").drop(pd.Timestamp("2024-03-29"))
    cal = trading_calendar(days)
    assert cal.loc["2024-03-28", "is_month_end"]
    assert cal.loc["2024-03-28", "days_to_month_end"] == 0
    assert cal.loc["2024-03-27", "days_to_month_end"] == 1


def test_incomplete_final_month_has_no_month_end():
    days = pd.bdate_range("2024-03-25", "2024-04-10")  # April is unfinished
    cal = trading_calendar(days)
    april = cal.loc["2024-04"]
    assert not april["is_month_end"].any()
    assert april["days_to_month_end"].isna().all()
    assert cal.loc["2024-03-29", "is_month_end"]


def test_signal_matches_hand_calculation():
    # ES +10%, ZN flat on one day: 66 / (66 + 40) - 0.60 = 0.0226...
    r = _returns(pd.bdate_range("2024-03-04", periods=3))
    r.iloc[0, 0] = 0.10
    sig = calendar_signal(r)
    assert sig.iloc[0] == pytest.approx(66 / 106 - 0.60)


def test_signal_resets_after_month_end():
    days = pd.bdate_range("2024-03-25", "2024-04-05")
    r = _returns(days, es=0.01)  # equities outperform every day
    sig = calendar_signal(r)
    # Drift builds through March, then restarts from 60/40 on April 1
    assert sig.loc["2024-03-29"] > sig.loc["2024-04-01"]
    assert sig.loc["2024-04-01"] == pytest.approx(0.606 / (0.606 + 0.40) - 0.60)


def test_signal_sign_follows_relative_performance():
    days = pd.bdate_range("2024-03-04", periods=5)
    assert (calendar_signal(_returns(days, es=0.01)) > 0).all()
    assert (calendar_signal(_returns(days, zn=0.01)) < 0).all()


def test_missing_returns_treated_as_flat():
    days = pd.bdate_range("2024-03-04", periods=3)
    r = _returns(days)
    r.iloc[1, 0] = np.nan
    assert np.isfinite(calendar_signal(r)).all()
