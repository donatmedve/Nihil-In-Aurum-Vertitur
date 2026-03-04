# scripts/fetch_data.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from decimal import Decimal
import pickle
import time

import MetaTrader5 as mt5

from shared.schemas import OHLCVBar, Pair, BarSource
from data.aggregation import bars_to_dataframe
from data.time_utils import mt5_server_to_utc

BROKER_TZ = "Etc/GMT-2"

SYMBOLS = {
    Pair.EURUSD: "EURUSD",
    Pair.GBPUSD: "GBPUSD",
}


def convert_mt5_bars(rates, pair: Pair, broker_tz: str) -> list[OHLCVBar]:
    result = []
    point = 0.00001 if pair != Pair.USDJPY else 0.001

    for r in rates:
        open_naive   = datetime.fromtimestamp(r['time'])
        utc_open     = mt5_server_to_utc(open_naive, broker_tz)
        utc_close    = utc_open + timedelta(seconds=300)
        spread_price = float(r['spread']) * point
        half         = spread_price / 2.0

        bar = OHLCVBar(
            pair=pair,
            utc_open=utc_open,
            utc_close=utc_close,
            timeframe_sec=300,
            open_bid=Decimal(str(round(float(r['open'])  - half, 6))),
            high_bid=Decimal(str(round(float(r['high'])  - half, 6))),
            low_bid=Decimal(str(round(float(r['low'])    - half, 6))),
            close_bid=Decimal(str(round(float(r['close'])- half, 6))),
            open_ask=Decimal(str(round(float(r['open'])  + half, 6))),
            high_ask=Decimal(str(round(float(r['high'])  + half, 6))),
            low_ask=Decimal(str(round(float(r['low'])    + half, 6))),
            close_ask=Decimal(str(round(float(r['close'])+ half, 6))),
            volume=int(r['tick_volume']),
            is_complete=True,
            source=BarSource.BROKER_HISTORICAL,
        )
        result.append(bar)
    return result


def fetch_all_bars(symbol: str, batch_size: int = 5000) -> list:
    """
    Fetch as many bars as possible by walking backwards in batches.
    MT5 won't accept huge single requests so we fetch in chunks.
    """
    all_rates = []
    offset = 0

    while True:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, offset, batch_size)

        if rates is None or len(rates) == 0:
            break

        all_rates = list(rates) + all_rates  # prepend older bars
        offset += len(rates)

        oldest = datetime.fromtimestamp(rates[0]['time'])
        print(f"    Fetched batch: offset={offset}, oldest bar={oldest.strftime('%Y-%m-%d')}")

        # Stop if we've gone back far enough or hit the limit
        if len(rates) < batch_size:
            print("    Reached end of available history.")
            break

        if offset >= 99000:  # MT5 hard limit is around 99,000 bars
            print("    Hit MT5 maximum history limit.")
            break

        time.sleep(0.2)  # be gentle with the terminal

    return all_rates


def main():
    print("Connecting to MetaTrader 5...")
    if not mt5.initialize():
        print(f"ERROR: {mt5.last_error()}")
        return

    print(f"Connected: {mt5.account_info().server}")

    for symbol in SYMBOLS.values():
        mt5.symbol_select(symbol, True)
    time.sleep(3)

    os.makedirs("data", exist_ok=True)

    for pair, symbol in SYMBOLS.items():
        print(f"\nFetching {symbol}...")

        all_rates = fetch_all_bars(symbol)

        if not all_rates:
            print(f"  ERROR: No data at all for {symbol}")
            continue

        print(f"  Total bars collected: {len(all_rates)}")
        print(f"  Earliest: {datetime.fromtimestamp(all_rates[0]['time']).strftime('%Y-%m-%d')}")
        print(f"  Latest:   {datetime.fromtimestamp(all_rates[-1]['time']).strftime('%Y-%m-%d')}")

        bars = convert_mt5_bars(all_rates, pair, BROKER_TZ)
        df   = bars_to_dataframe(bars)

        out_path = f"data/{symbol.lower()}_5m_bars.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(df, f)

        print(f"  Saved → {out_path}")

    mt5.shutdown()
    print("\nDone.")


if __name__ == "__main__":
    main()
