"""Calendar rebalancing signal (Harvey, Mazzoleni & Melone).

A simulated 60/40 ES/ZN portfolio drifts with daily returns and is reset to
target at the close of the last trading day of each month. The signal is the
equity weight's distance from target at each close, BEFORE that day's reset:

    signal > 0  -> equities overweight -> rebalancers will sell ES, buy ZN
    signal < 0  -> equities underweight -> rebalancers will buy ES, sell ZN

Rule of thumb: signal ~= 0.24 * (month-to-date ES return - ZN return),
since 0.6 * 0.4 = 0.24.
"""

import pandas as pd

TARGET = 0.60


def trading_calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Month-end flags built from the actual trading days in the data.

    Columns:
      is_month_end      True on the last trading day of each calendar month
      days_to_month_end 0 on the last trading day, 1 the day before, ...

    The final month is usually incomplete (data stops before month-end), so
    its days_to_month_end is NaN and is_month_end is False.
    """
    idx = pd.DatetimeIndex(index).sort_values()
    month = idx.to_period("M")
    pos = pd.Series(range(len(idx)), index=idx)
    dtme = pos.groupby(month).transform("max") - pos

    last_day = idx[-1]
    last_bday_of_month = last_day + pd.offsets.BMonthEnd(0)
    if last_day < last_bday_of_month:  # data ends mid-month
        dtme[month == last_day.to_period("M")] = float("nan")

    return pd.DataFrame(
        {"is_month_end": dtme.eq(0), "days_to_month_end": dtme},
        index=idx,
    )


def calendar_signal(returns: pd.DataFrame, target: float = TARGET) -> pd.Series:
    """Equity weight minus target at each close, before the month-end reset.

    `returns` needs es_ret and zn_ret. Missing returns (market closures) are
    treated as 0 for that leg, i.e. its value is unchanged that day.
    """
    r = returns[["es_ret", "zn_ret"]].fillna(0.0).sort_index()
    cal = trading_calendar(r.index)

    w = target
    out = []
    for (re, rb), month_end in zip(r.itertuples(index=False), cal["is_month_end"]):
        eq, bd = w * (1 + re), (1 - w) * (1 + rb)
        w = eq / (eq + bd)          # drift with today's returns
        out.append(w - target)      # signal at today's close
        if month_end:
            w = target              # rebalance after the close

    return pd.Series(out, index=r.index, name="calendar_signal")
