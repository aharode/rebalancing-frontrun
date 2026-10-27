
import databento as db
from dotenv import load_dotenv

load_dotenv()

DATASET = "GLBX.MDP3"
SYMBOLS = ["ES.v.0", "ES.v.1", "ZN.v.0", "ZN.v.1"]


def get_cost(symbols, start_date, end_date):
    client = db.Historical()  # reads DATABENTO_API_KEY from the environment
    return client.metadata.get_cost(
        dataset=DATASET,
        schema="ohlcv-1d",
        symbols=symbols,
        stype_in="continuous",
        start=start_date,
        end=end_date,
    )


if __name__ == "__main__":
    cost = get_cost(SYMBOLS, "2010-06-06", "2026-10-27")
    print(f"Total cost = ${cost:.2f}")