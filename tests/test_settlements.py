"""Tests for the settlement table and roll-aware returns (data/settlements.py).

Each test builds a small synthetic Databento-style statistics frame, so the
rules validated on real data stay enforced if the code changes.
"""

import databento as db
import numpy as np
import pandas as pd
import pytest

from rebalancing_frontrun.data.settlements import (
    MANUAL_FILLS,
    build_returns,
    build_settlements,
)

SETTLE = int(db.StatType.SETTLEMENT_PRICE)
VOLUME = int(db.StatType.CLEARED_VOLUME)
NULL_QTY = 2**63 - 1
COLS = ["ts_recv", "ts_ref", "symbol", "stat_type", "price", "quantity", "stat_flags"]


def _stats(rows):
    df = pd.DataFrame(rows, columns=COLS)
    df["ts_recv"] = pd.to_datetime(df["ts_recv"], utc=True)
    df["ts_ref"] = pd.to_datetime(df["ts_ref"], utc=True)
    return df.set_index("ts_recv")


def _day(date, symbol, price, volume, recv_hour=22):
    """One settlement and one volume record for a contract on a trade date."""
    ts = f"{date} {recv_hour}:00"
    return [
        (ts, date, symbol, SETTLE, price, NULL_QTY, 3),
        (ts, date, symbol, VOLUME, np.nan, volume, 0),
    ]


# ---------- build_settlements ----------

def test_drops_spreads_replays_and_nonpositive_prices():
    rows = _day("2024-03-01", "ESH4", 5000.0, 1000)
    rows += [
        ("2024-03-01 22:00", "2024-03-01", "ESH4-ESM4", SETTLE, -10.0, NULL_QTY, 3),  # spread
        ("2024-03-03 16:00", None, "ESH4", SETTLE, 9999.0, NULL_QTY, 1),             # Sunday replay
        ("2024-03-01 22:00", "2024-03-01", "ESU4", SETTLE, 0.0, NULL_QTY, 2),         # zero price
    ]
    out = build_settlements(_stats(rows))
    es = out[out["product"] == "ES"]
    assert list(es["symbol"]) == ["ESH4"]
    assert es["settlement"].iloc[0] == 5000.0


def test_drops_weekend_trade_dates():
    rows = _day("2024-03-01", "ESH4", 5000.0, 1000) + _day("2024-03-02", "ESH4", 5001.0, 10)
    out = build_settlements(_stats(rows))
    es = out[out["product"] == "ES"]
    assert (es["trade_date"].dt.dayofweek < 5).all()
    assert len(es) == 1


def test_keeps_last_published_settlement():
    rows = [
        ("2024-03-01 21:00", "2024-03-01", "ESH4", SETTLE, 4999.0, NULL_QTY, 2),  # preliminary
        ("2024-03-01 22:00", "2024-03-01", "ESH4", SETTLE, 5000.0, NULL_QTY, 3),  # final
    ]
    out = build_settlements(_stats(rows))
    assert out.loc[out["symbol"] == "ESH4", "settlement"].item() == 5000.0


def test_manual_fills_added_only_when_missing():
    # Databento has the ZNH0 settlement on 2020-02-27 -> the manual fill is not added
    rows = _day("2020-02-27", "ZNH0", 999.0, 1)
    out = build_settlements(_stats(rows))
    znh0 = out[(out["symbol"] == "ZNH0") & (out["trade_date"] == "2020-02-27")]
    assert len(znh0) == 1
    assert znh0["settlement"].item() == 999.0

    # The other fill (ZNU0 on 2020-06-30) is missing -> it is added once
    znu0 = out[(out["symbol"] == "ZNU0") & (out["trade_date"] == "2020-06-30")]
    assert len(znu0) == 1
    expected = MANUAL_FILLS.loc[MANUAL_FILLS["symbol"] == "ZNU0", "settlement"].item()
    assert znu0["settlement"].item() == expected


# ---------- build_returns ----------

def _settlements(records):
    """records: (date, product, symbol, settlement, volume)"""
    df = pd.DataFrame(records, columns=["trade_date", "product", "symbol", "settlement", "volume"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def _two_contract_roll():
    """ESH4 is most traded on days 1-2, ESM4 from day 3; ESM4 trades 10 pts higher."""
    dates = pd.bdate_range("2024-03-04", periods=5)
    rec = []
    for i, d in enumerate(dates):
        rec.append((d, "ES", "ESH4", 5000.0 + i, 1000 if i < 2 else 10))
        rec.append((d, "ES", "ESM4", 5010.0 + i, 100 if i < 2 else 2000))
        rec.append((d, "ZN", "ZNH4", 110.0, 500))
    return _settlements(rec), dates


def test_holds_contract_with_most_volume_the_previous_day():
    s, dates = _two_contract_roll()
    r = build_returns(s)
    # ESM4 becomes most traded on day 3 (index 2), so it is held from day 4 (index 3)
    assert r.loc[dates[2], "es_contract"] == "ESH4"
    assert r.loc[dates[3], "es_contract"] == "ESM4"


def test_roll_creates_no_price_jump():
    s, dates = _two_contract_roll()
    r = build_returns(s)
    # Each day's return compares the held contract with its own prior settlement,
    # so the 10-point gap between contracts never shows up as a return
    roll_day_ret = r.loc[dates[3], "es_ret"]
    assert roll_day_ret == pytest.approx((5013.0 / 5012.0) - 1)
    assert r["es_ret"].abs().max() < 0.001


def test_keeps_previous_contract_when_volume_missing():
    s, dates = _two_contract_roll()
    s.loc[(s["trade_date"] == dates[1]) & (s["product"] == "ES"), "volume"] = np.nan
    r = build_returns(s)
    assert r.loc[dates[2], "es_contract"] == r.loc[dates[1], "es_contract"]
    assert not np.isnan(r.loc[dates[2], "es_ret"])
