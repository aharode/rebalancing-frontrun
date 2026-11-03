from pathlib import Path

import databento as db
from dotenv import load_dotenv

load_dotenv()

DATASET = "GLBX.MDP3"
START, END = "2010-06-06", "2026-10-27"  # fixed end date so pulls are reproducible
REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"


def _params(schema, symbols, stype_in, start, end):
    return {
        "dataset": DATASET, "schema": schema, "symbols": symbols,
        "stype_in": stype_in, "start": start, "end": end,
    }


def _cache_path(schema, symbols, start, end):
    tag = "_".join(s.replace(".", "-") for s in symbols)
    return RAW_DIR / f"{tag}_{schema}_{start}_{end}.dbn.zst"


def quote(schema, symbols, stype_in, start=START, end=END):
    """Free cost estimate in USD."""
    client = db.Historical()
    return client.metadata.get_cost(**_params(schema, symbols, stype_in, start, end))


def pull(schema, symbols, stype_in, start=START, end=END, max_cost=5.0):
    """Download once, then always read from the local cache."""
    path = _cache_path(schema, symbols, start, end)
    if path.exists():
        return db.DBNStore.from_file(path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    params = _params(schema, symbols, stype_in, start, end)
    client = db.Historical()
    cost = client.metadata.get_cost(**params)
    if cost > max_cost:
        raise RuntimeError(f"{schema} pull would cost ${cost:.2f} (cap ${max_cost:.2f})")
    return client.timeseries.get_range(**params, path=path)


REQUESTS = [
    ("statistics", ["ES.FUT", "ZN.FUT"], "parent"),
    ("definition", ["ES.FUT", "ZN.FUT"], "parent"),
    ("ohlcv-1d", ["ES.v.0", "ES.v.1", "ZN.v.0", "ZN.v.1"], "continuous"),
]

if __name__ == "__main__":
    for schema, symbols, stype_in in REQUESTS:
        store = pull(schema, symbols, stype_in)
        df = store.to_df()
        print(f"{schema:<11} {len(df):>10,} rows  {df.index.min()} → {df.index.max()}")