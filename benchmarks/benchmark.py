"""Benchmark: Python vs. C++ backtest engine.

Times (1) a single backtest and (2) the placebo test, on the real ES/ZN returns.
The Python placebo baseline is a fair, vectorized NumPy version: signal and
volatility are computed once, and each simulation is vectorized over days.

Run from the repo root (needs data/processed/returns.parquet):
    uv run python benchmarks/benchmark.py
"""

import platform
import time

import numpy as np
import pandas as pd

from rebalancing_frontrun.backtest.fast import placebo, run_backtest_fast
from rebalancing_frontrun.backtest.strategy import TRADING_DAYS, run_backtest
from rebalancing_frontrun.data.databento_loader import REPO_ROOT
from rebalancing_frontrun.signals.calendar import calendar_signal, trading_calendar

IS_END = "2023-03-17"


def timeit(fn, repeat):
    fn()  # warm-up
    t = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t) / repeat


def numpy_placebo(returns, n_sims, seed=0, window=5, vol_lookback=60, leg_vol=0.10,
                  es_cost_bps=0.5, zn_cost_bps=1.5):
    """Vectorized NumPy placebo used as the Python baseline (net Sharpe, full sample)."""
    r = returns[["es_ret", "zn_ret"]].sort_index()
    r0 = r.fillna(0.0)
    es, zn = r0["es_ret"].to_numpy(), r0["zn_ret"].to_numpy()
    dtme = trading_calendar(r.index)["days_to_month_end"].to_numpy()
    sig_prev = calendar_signal(r).shift(1).to_numpy()
    ann = np.sqrt(TRADING_DAYS)
    vol_es = (r0["es_ret"].rolling(vol_lookback, min_periods=20).std().shift(1) * ann).to_numpy()
    vol_zn = (r0["zn_ret"].rolling(vol_lookback, min_periods=20).std().shift(1) * ann).to_numpy()
    valid = ~np.isnan(sig_prev) & ~np.isnan(vol_es) & ~np.isnan(vol_zn)
    direction = -np.sign(np.nan_to_num(sig_prev))

    ends = np.flatnonzero(dtme == 0)
    starts = np.r_[0, ends[:-1] + 1]
    blocks = [(s, e) for s, e in zip(starts, ends) if e - s + 1 >= window]

    rng = np.random.default_rng(seed)
    out = np.empty(n_sims)
    for k in range(n_sims):
        mask = np.zeros(len(r), dtype=bool)
        for s, e in blocks:
            start = rng.integers(s, e - window + 2)
            mask[start:start + window] = True
        active = mask & valid
        w_es = np.where(active, direction * leg_vol / vol_es, 0.0)
        w_zn = np.where(active, -direction * leg_vol / vol_zn, 0.0)
        cost = (np.abs(np.diff(w_es, prepend=0.0)) * es_cost_bps
                + np.abs(np.diff(w_zn, prepend=0.0)) * zn_cost_bps) / 1e4
        net = w_es * es + w_zn * zn - cost
        out[k] = net.mean() / net.std(ddof=1) * ann
    return out


def main():
    returns = pd.read_parquet(REPO_ROOT / "data" / "processed" / "returns.parquet")
    print(f"{len(returns):,} trading days | {platform.machine()} | Python {platform.python_version()}\n")

    t_py = timeit(lambda: run_backtest(returns), repeat=20)
    t_cpp = timeit(lambda: run_backtest_fast(returns), repeat=200)
    print("Single backtest")
    print(f"  Python (pandas)  {t_py * 1e3:8.2f} ms")
    print(f"  C++              {t_cpp * 1e3:8.2f} ms   ({t_py / t_cpp:.0f}x faster)\n")

    n_np, n_cpp = 500, 10_000
    t_np = timeit(lambda: numpy_placebo(returns, n_np), repeat=1) / n_np
    t_pl = timeit(lambda: placebo(returns, n_sims=n_cpp, in_sample_end=IS_END), repeat=3) / n_cpp
    print("Placebo test, per simulation")
    print(f"  Python (NumPy)   {t_np * 1e6:8.1f} us")
    print(f"  C++              {t_pl * 1e6:8.1f} us   ({t_np / t_pl:.0f}x faster)")
    print(f"  10,000 sims: NumPy ~{t_np * 1e4:.1f} s vs C++ {t_pl * 1e4:.2f} s")


if __name__ == "__main__":
    main()
