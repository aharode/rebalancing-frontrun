"""Build clean daily settlement prices and roll-aware futures returns.

Rules (validated in notebooks):
- Settlement = official CME settlement price (stat_type SETTLEMENT_PRICE).
- Outright contracts only (calendar spreads contain "-"), price > 0.
- Rows without ts_ref are Sunday replays of the prior settlement -> dropped.
- Last record per contract per trade date is the final settlement
  (matched the final/actual flag rule 100% where flags are interpretable).
- Roll: hold the contract with the highest cleared volume on the PREVIOUS
  trade date (no look-ahead). Each day's return compares the held contract
  to its own prior settlement, so rolls never create fake jumps.
"""

from pathlib import Path

import databento as db
import pandas as pd

UNDEF_QTY = 2**63 - 1  # Databento's null value for quantity fields
PRODUCTS = ("ES", "ZN")

# Manual fills for settlements missing from Databento's statistics data.
# Databento has no ZN settlement records on these two dates (neighboring days
# are complete). Values come from Barchart's per-contract price history
# ("Latest" = CME settlement; prices in 32nds), viewed 2026-10-28:
#   https://www.barchart.com/futures/quotes/ZNH20/price-history/historical
#   https://www.barchart.com/futures/quotes/ZNU20/price-history/historical
# Cross-check: Barchart matches Databento exactly (settlement and volume) on the
# surrounding days (ZNH20 on 2020-02-26/28; ZNU20 on 18 days from 2020-06-22 to
# 2020-07-17), and Barchart's daily changes link the filled value to both neighbors.
# Only the held contract is filled; other contracts that day are not needed.
MANUAL_FILLS = pd.DataFrame(
    [
        # trade_date, product, symbol, settlement, volume
        ("2020-02-27", "ZN", "ZNH0", 133 + 14.0 / 32, 2_733_280),  # 133-140
        ("2020-06-30", "ZN", "ZNU0", 139 + 5.5 / 32, 1_240_939),   # 139-055
    ],
    columns=["trade_date", "product", "symbol", "settlement", "volume"],
).astype({"trade_date": "datetime64[ns]", "volume": "float64"})


def _to_date(s: pd.Series) -> pd.Series:
    """Timestamps -> tz-naive midnight dates."""
    s = pd.to_datetime(s)
    if s.dt.tz is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.dt.normalize()


def build_settlements(stats: pd.DataFrame) -> pd.DataFrame:
    """One row per outright contract per trade date.

    Columns: trade_date, product, symbol, settlement, volume
    """
    df = stats.sort_index()  # index is ts_recv, so "last" = latest published
    df = df[df["ts_ref"].notna() & ~df["symbol"].str.contains("-")].copy()
    df["trade_date"] = _to_date(df["ts_ref"])
    df = df[df["trade_date"].dt.dayofweek < 5]  # a few stray records carry weekend dates
    df["product"] = df["symbol"].str[:2]
    df = df[df["product"].isin(PRODUCTS)]

    keys = ["trade_date", "product", "symbol"]

    is_settle = (df["stat_type"] == db.StatType.SETTLEMENT_PRICE) & (df["price"] > 0)
    settle = df[is_settle].groupby(keys)["price"].last().rename("settlement")

    is_vol = (df["stat_type"] == db.StatType.CLEARED_VOLUME) & (df["quantity"] < UNDEF_QTY)
    volume = df[is_vol].groupby(keys)["quantity"].last().rename("volume")

    out = pd.concat([settle, volume], axis=1).reset_index()
    out = out[out["settlement"].notna()]

    # Add manual fills only where Databento is still missing that contract-date,
    # so a future data refresh that includes them is never double-counted.
    existing = out.set_index(["trade_date", "symbol"]).index
    fill_idx = MANUAL_FILLS.set_index(["trade_date", "symbol"]).index
    fills = MANUAL_FILLS[~fill_idx.isin(existing)]
    out = pd.concat([out, fills], ignore_index=True)

    return out.sort_values(keys).reset_index(drop=True)


def _product_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Roll-aware daily returns for a single product."""
    px = df.pivot(index="trade_date", columns="symbol", values="settlement").sort_index()
    vol = df.pivot(index="trade_date", columns="symbol", values="volume").reindex(px.index)

    # Contract held today = highest volume yesterday.
    # If yesterday's volume record is missing, keep holding the previous contract.
    vol_prev = vol.shift(1)
    has_vol = vol_prev.notna().any(axis=1)
    held = vol_prev[has_vol].idxmax(axis=1).reindex(px.index).ffill().dropna()

    # Every contract's return vs. its OWN previous settlement
    contract_ret = px / px.shift(1) - 1

    ret = pd.Series(
        [contract_ret.at[d, c] for d, c in held.items()],
        index=held.index,
        dtype=float,
    )
    return pd.DataFrame({"contract": held, "ret": ret})


def build_returns(settlements: pd.DataFrame) -> pd.DataFrame:
    """Columns: es_contract, es_ret, zn_contract, zn_ret (index: trade_date)."""
    parts = []
    for product in PRODUCTS:
        r = _product_returns(settlements[settlements["product"] == product])
        r.columns = [f"{product.lower()}_{c}" for c in r.columns]
        parts.append(r)
    out = pd.concat(parts, axis=1, sort=True)
    out.index.name = "trade_date"
    return out


if __name__ == "__main__":
    from rebalancing_frontrun.data.databento_loader import REPO_ROOT, pull

    stats = pull("statistics", ["ES.FUT", "ZN.FUT"], "parent").to_df()
    settlements = build_settlements(stats)
    returns = build_returns(settlements)

    processed = Path(REPO_ROOT) / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    settlements.to_parquet(processed / "settlements.parquet", index=False)
    returns.to_parquet(processed / "returns.parquet")

    print(f"settlements: {len(settlements):,} rows")
    print(f"returns:     {len(returns):,} days, "
          f"{returns.index.min():%Y-%m-%d} -> {returns.index.max():%Y-%m-%d}")
    print(f"missing ES: {returns['es_ret'].isna().sum()}, "
          f"missing ZN: {returns['zn_ret'].isna().sum()}")