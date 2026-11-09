"""pandas interface to the C++ backtest engine (cpp/engine.cpp).

`run_backtest_fast` gives the same output as `strategy.run_backtest`, computed in
C++. `placebo` runs the same trade on randomly placed 5-day windows inside each
month, thousands of times, to test whether month-end timing matters.

The Python implementation in strategy.py remains the reference: tests check
that both agree (tests/test_engine.py).
"""

import numpy as np
import pandas as pd

from rebalancing_frontrun import _engine
from rebalancing_frontrun.signals.calendar import trading_calendar


def _inputs(returns: pd.DataFrame):
    r = returns[["es_ret", "zn_ret"]].sort_index()
    dtme = trading_calendar(r.index)["days_to_month_end"]
    return (
        r,
        r["es_ret"].to_numpy(dtype=float),
        r["zn_ret"].to_numpy(dtype=float),
        dtme.to_numpy(dtype=float),
    )


def run_backtest_fast(
    returns: pd.DataFrame,
    window: int = 5,
    vol_lookback: int = 60,
    leg_vol: float = 0.10,
    es_cost_bps: float = 0.5,
    zn_cost_bps: float = 1.5,
    timing: str = "holding",
) -> pd.DataFrame:
    """C++ version of strategy.run_backtest; same arguments and columns."""
    r, es, zn, dtme = _inputs(returns)
    out = _engine.backtest(
        es, zn, dtme,
        window=window, vol_lookback=vol_lookback, leg_vol=leg_vol,
        es_cost_bps=es_cost_bps, zn_cost_bps=zn_cost_bps, timing=timing,
    )

    # Month each day's P&L belongs to as a trade (paper reversal day -> prior month)
    period = r.index.to_period("M")
    reversal = np.zeros(len(r), dtype=bool)
    if timing == "paper":
        reversal[1:] = dtme[:-1] == 0
    trade_month = pd.Series(period, index=r.index).where(~reversal, period - 1)

    cols = ["signal_prev", "active", "direction", "w_es", "w_zn", "gross", "cost", "net"]
    return pd.DataFrame({"trade_month": trade_month, **{c: out[c] for c in cols}}, index=r.index)


def placebo(
    returns: pd.DataFrame,
    n_sims: int = 10_000,
    seed: int = 0,
    in_sample_end=None,
    window: int = 5,
    vol_lookback: int = 60,
    leg_vol: float = 0.10,
    es_cost_bps: float = 0.5,
    zn_cost_bps: float = 1.5,
) -> pd.DataFrame:
    """Net Sharpe ratios of `n_sims` backtests on random `window`-day blocks per month.

    Each simulation places one block of consecutive trading days at a random
    position inside every complete month and runs the same trade as the real
    strategy (direction from the Calendar signal at the previous close, same
    sizing and costs). Columns: full, in_sample, out_of_sample. The split is
    after `in_sample_end` (a date); without it the out-of-sample column is NaN.
    """
    r, es, zn, dtme = _inputs(returns)
    split = len(r) if in_sample_end is None else int((r.index <= pd.Timestamp(in_sample_end)).sum())
    out = _engine.placebo(
        es, zn, dtme,
        n_sims=n_sims, seed=seed, split=split, window=window, vol_lookback=vol_lookback,
        leg_vol=leg_vol, es_cost_bps=es_cost_bps, zn_cost_bps=zn_cost_bps,
    )
    return pd.DataFrame(
        {
            "full": out["sharpe_full"],
            "in_sample": out["sharpe_in_sample"],
            "out_of_sample": out["sharpe_out_of_sample"],
        }
    )
