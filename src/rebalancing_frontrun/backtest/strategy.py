"""Front-running backtest for the Calendar rebalancing signal.

Timing (no look-ahead): the position held over day t, which earns day t's
close-to-close return, is set at the close of day t-1 using only data
available then: the Calendar signal at t-1 and trailing volatility through t-1.

Rule: during the last `window` trading days of each month,
  signal > 0 (equities overweight)  -> rebalancers will sell ES / buy ZN
                                    -> we go short ES, long ZN ahead of them
  signal < 0 (equities underweight) -> long ES, short ZN
Flat on all other days. Direction uses the sign of the signal only.

Two timing conventions (`timing`):
  "holding" (default): hold during the last `window` trading days of the month.
  "paper": follow Harvey, Mazzoleni & Melone's Section 4 strategy. A signal
           observed on one of the last `window` days sets the NEXT day's
           position: front-running on the last `window - 1` days of the month,
           then the OPPOSITE side on the first trading day of the next month,
           where the paper finds month-end price pressure partly reverses.
           The reversal day uses the sign of the month-end (pre-reset) signal.
           That day counts as part of the prior month's trade.

Sizing: each leg targets the same annualized volatility (`leg_vol`), using a
trailing `vol_lookback`-day estimate, so the trade is a stock-vs-bond bet
rather than mostly an equity bet (ES is ~3x as volatile as ZN).

Costs: each leg pays `cost_bps` per side on notional traded, i.e. on the daily
change in its weight. Defaults are roughly one tick per side incl. commissions:
ES ~0.5 bp, ZN ~1.5 bp.

P&L is additive (weights are notional exposure per unit of capital), so
cumulative P&L is a running sum, not a compounded product.
"""

import numpy as np
import pandas as pd

from rebalancing_frontrun.signals.calendar import calendar_signal, trading_calendar

TRADING_DAYS = 252


def run_backtest(
    returns: pd.DataFrame,
    window: int = 5,
    vol_lookback: int = 60,
    leg_vol: float = 0.10,
    es_cost_bps: float = 0.5,
    zn_cost_bps: float = 1.5,
    timing: str = "holding",
) -> pd.DataFrame:
    """Daily positions and P&L.

    Columns: trade_month, signal_prev, active, direction
    (+1 = long ES / short ZN), w_es, w_zn, gross, cost, net.
    """
    r = returns[["es_ret", "zn_ret"]].sort_index()
    r0 = r.fillna(0.0)  # market closures: that leg is flat for the day
    cal = trading_calendar(r.index)

    # Everything below is known at the close of day t-1
    signal_prev = calendar_signal(r).shift(1)
    ann = np.sqrt(TRADING_DAYS)
    vol_es = r0["es_ret"].rolling(vol_lookback, min_periods=20).std().shift(1) * ann
    vol_zn = r0["zn_ret"].rolling(vol_lookback, min_periods=20).std().shift(1) * ann

    dtme = cal["days_to_month_end"]
    flip = pd.Series(1.0, index=r.index)  # -1 on reversal days
    reversal = pd.Series(False, index=r.index)
    if timing == "holding":
        in_window = dtme.lt(window)  # NaN (incomplete month) -> False
    elif timing == "paper":
        dtme_prev = dtme.shift(1)
        frontrun = dtme_prev.ge(1) & dtme_prev.lt(window)  # yesterday a signal day, same month
        reversal = dtme_prev.eq(0)  # first trading day after a month-end
        in_window = frontrun | reversal
        flip = flip.where(~reversal, -1.0)
    else:
        raise ValueError(f"timing must be 'holding' or 'paper', not {timing!r}")
    active = in_window & signal_prev.notna() & vol_es.notna() & vol_zn.notna()

    direction = (-np.sign(signal_prev) * flip).where(active, 0.0)
    w_es = (direction * leg_vol / vol_es).where(active, 0.0)
    w_zn = (-direction * leg_vol / vol_zn).where(active, 0.0)

    gross = w_es * r0["es_ret"] + w_zn * r0["zn_ret"]

    # Trading cost on the change in each leg's weight (entry, resizing, exit)
    turn_es = w_es.diff().abs().fillna(w_es.abs())
    turn_zn = w_zn.diff().abs().fillna(w_zn.abs())
    cost = turn_es * es_cost_bps / 1e4 + turn_zn * zn_cost_bps / 1e4

    # Month each day's P&L belongs to as a trade (reversal day -> prior month)
    period = r.index.to_period("M")
    trade_month = pd.Series(period, index=r.index).where(~reversal, period - 1)

    return pd.DataFrame(
        {
            "trade_month": trade_month,
            "signal_prev": signal_prev,
            "active": active,
            "direction": direction,
            "w_es": w_es,
            "w_zn": w_zn,
            "gross": gross,
            "cost": cost,
            "net": gross - cost,
        }
    )


def summarize(bt: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Performance metrics for gross and net P&L over [start, end].

    Daily stats use every trading day (flat days count as 0), which is the
    right basis for a strategy that sits in no position most of the month.
    Per-trade stats treat each month's window as one trade (`trade_month`).
    """
    b = bt.loc[start:end]
    month = b["trade_month"]
    traded_months = b.groupby(month)["active"].any()
    traded_months = traded_months[traded_months].index

    out = {}
    for col in ("gross", "net"):
        pnl = b[col]
        cum = pnl.cumsum()
        per_trade = pnl.groupby(month).sum().loc[traded_months]
        out[col] = {
            "ann_return_%": pnl.mean() * TRADING_DAYS * 100,
            "ann_vol_%": pnl.std() * np.sqrt(TRADING_DAYS) * 100,
            "sharpe": pnl.mean() / pnl.std() * np.sqrt(TRADING_DAYS),
            "max_drawdown_%": (cum - cum.cummax()).min() * 100,
            "trades": len(per_trade),
            "hit_rate_%": (per_trade > 0).mean() * 100,
            "avg_trade_bps": per_trade.mean() * 1e4,
            "t_stat_trades": per_trade.mean() / per_trade.std() * np.sqrt(len(per_trade)),
        }
    return pd.DataFrame(out).round(2)
