# data/ingestion.py
# MT5 bar fetching — the concrete implementation of _fetch_recent_bars.
#
# This is the ONLY place that calls mt5.copy_rates_from_pos().
# The live loop calls fetch_recent_bars_mt5() and gets back a list of OHLCVBar.
#
# Rules:
#   - Always fetch N+1 bars so the last (in-progress) bar can be dropped
#   - All timestamps are converted from broker server time to UTC immediately
#   - Never return the currently-open (incomplete) bar as a signal source
#   - Log and raise on data quality issues — do not silently return bad data

import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from shared.schemas import OHLCVBar, Pair, BarSource, assert_utc
from data.time_utils import mt5_server_to_utc, now_utc, format_utc, is_market_open

logger = logging.getLogger(__name__)

# MT5 pair name mapping — must match your broker's exact symbol strings
DEFAULT_SYMBOL_MAP = {
    Pair.EURUSD: "EURUSD",
    Pair.GBPUSD: "GBPUSD",
    Pair.USDJPY: "USDJPY",
}


def fetch_recent_bars_mt5(
    pair: Pair,
    n: int,
    timeframe_mt5: int,          # e.g. mt5.TIMEFRAME_M5
    timeframe_sec: int,          # e.g. 300  (must match timeframe_mt5)
    broker_tz_str: str,          # e.g. "Etc/GMT-2"  — check your broker docs
    symbol_map: dict = None,
) -> list[OHLCVBar]:
    """
    Fetch the N most recent complete bars from MT5 for the given pair.

    Returns a list of OHLCVBar sorted ascending by utc_open.
    The last bar in the list is always marked is_complete=False (currently open).
    The caller should filter to [b for b in bars if b.is_complete] before use.

    MT5's copy_rates_from_pos(symbol, timeframe, start_pos, count):
      - start_pos=0 means the current (incomplete) bar
      - start_pos=1 means the last complete bar
      - Higher start_pos = further back in history

    We fetch from pos=0 so we can determine completeness ourselves,
    and request n+1 bars to guarantee we have n complete bars after
    dropping the currently-open one.

    Raises RuntimeError on MT5 errors.
    Raises ImportError if MetaTrader5 package is not installed.
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise ImportError(
            "MetaTrader5 package not installed. Run: pip install MetaTrader5\n"
            "Note: MT5 Python API only works on Windows."
        )

    sym_map = symbol_map or DEFAULT_SYMBOL_MAP
    symbol = sym_map.get(pair)
    if symbol is None:
        raise ValueError(f"Pair {pair} not found in symbol_map: {sym_map}")

    # Fetch n+2 bars: +1 for the in-progress bar, +1 as insurance against
    # edge cases at bar boundaries where a bar may have just completed
    fetch_count = n + 2

    rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, 0, fetch_count)

    if rates is None or len(rates) == 0:
        error = mt5.last_error()
        raise RuntimeError(
            f"MT5 copy_rates_from_pos({symbol}) returned None. "
            f"MT5 error: {error}. "
            f"Is MT5 terminal running? Is the symbol available?"
        )

    if len(rates) < 2:
        raise RuntimeError(
            f"MT5 returned only {len(rates)} bar(s) for {symbol}. "
            f"Need at least 2. Is historical data loaded in MT5?"
        )

    now = now_utc()
    bars: list[OHLCVBar] = []

    for i, rate in enumerate(rates):
        # MT5 rate fields: time (int unix), open, high, low, close, tick_volume, spread, real_volume
        # 'time' is the bar OPEN time in broker server time (as Unix timestamp)
        bar_open_utc = datetime.fromtimestamp(rate["time"], tz=timezone.utc) - timedelta(hours=2)
        bar_close_utc = bar_open_utc + timedelta(seconds=timeframe_sec)

        # A bar is complete if its close time is in the past.
        # The currently-open bar's close time is in the future.
        import time as _time
        is_complete = bar_close_utc.timestamp() <= _time.time()

        # Derive bid/ask from OHLC + spread.
        # MT5 rates are mid prices. Spread is in points (1 point = 0.00001 for 5-digit brokers).
        # We approximate: bid = rate price, ask = bid + spread * point_size.
        # For live signals we always get fresh prices via adapter.get_price(), so
        # this approximation is only used for feature computation (RSI, ATR, etc.)
        # where mid-price accuracy is sufficient.
        spread_points = int(rate["spread"]) if rate["spread"] else 0

        # Point size: 0.00001 for EURUSD/GBPUSD (5-digit), 0.001 for USDJPY
        if pair == Pair.USDJPY:
            point_size = 0.001
        else:
            point_size = 0.00001

        spread_price = Decimal(str(round(spread_points * point_size, 5)))

        open_bid  = Decimal(str(round(float(rate["open"]),  5)))
        high_bid  = Decimal(str(round(float(rate["high"]),  5)))
        low_bid   = Decimal(str(round(float(rate["low"]),   5)))
        close_bid = Decimal(str(round(float(rate["close"]), 5)))

        open_ask  = open_bid  + spread_price
        high_ask  = high_bid  + spread_price
        low_ask   = low_bid   + spread_price
        close_ask = close_bid + spread_price

        bar = OHLCVBar(
            pair=pair,
            utc_open=bar_open_utc,
            utc_close=bar_close_utc,
            timeframe_sec=timeframe_sec,
            open_bid=open_bid,
            high_bid=high_bid,
            low_bid=low_bid,
            close_bid=close_bid,
            open_ask=open_ask,
            high_ask=high_ask,
            low_ask=low_ask,
            close_ask=close_ask,
            volume=int(rate["tick_volume"]),
            is_complete=is_complete,
            source=BarSource.BROKER_LIVE,
        )
        bars.append(bar)

    # Bars from MT5 come newest-first (pos=0 is current). Reverse to ascending.
    bars.reverse()

    n_complete = sum(1 for b in bars if b.is_complete)
    logger.debug(
        "Fetched bars from MT5",
        extra={
            "pair": pair.value,
            "symbol": symbol,
            "fetched": len(bars),
            "complete": n_complete,
            "latest_bar": format_utc(bars[-1].utc_open) if bars else "none",
        }
    )

    return bars


def fetch_historical_bars_mt5(
    pair: Pair,
    date_from: datetime,
    date_to: datetime,
    timeframe_mt5: int,
    timeframe_sec: int,
    broker_tz_str: str,
    symbol_map: dict = None,
) -> list[OHLCVBar]:
    """
    Fetch historical bars between date_from and date_to (both UTC-aware).
    Used for research/training data collection, not live trading.

    Example:
        from datetime import datetime, timezone
        import MetaTrader5 as mt5

        bars = fetch_historical_bars_mt5(
            pair=Pair.EURUSD,
            date_from=datetime(2021, 1, 1, tzinfo=timezone.utc),
            date_to=datetime(2024, 1, 1, tzinfo=timezone.utc),
            timeframe_mt5=mt5.TIMEFRAME_M5,
            timeframe_sec=300,
            broker_tz_str="Etc/GMT-2",
        )
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise ImportError("MetaTrader5 package not installed. Run: pip install MetaTrader5")

    assert_utc(date_from, "fetch_historical_bars_mt5 date_from")
    assert_utc(date_to,   "fetch_historical_bars_mt5 date_to")

    sym_map = symbol_map or DEFAULT_SYMBOL_MAP
    symbol = sym_map.get(pair)
    if symbol is None:
        raise ValueError(f"Pair {pair} not in symbol_map")

    # MT5 copy_rates_range takes naive datetimes in broker server time.
    # We convert our UTC inputs to broker time first.
    from data.time_utils import utc_to_mt5_server
    dt_from_broker = utc_to_mt5_server(date_from, broker_tz_str)
    dt_to_broker   = utc_to_mt5_server(date_to,   broker_tz_str)

    rates = mt5.copy_rates_range(symbol, timeframe_mt5, dt_from_broker, dt_to_broker)

    if rates is None:
        error = mt5.last_error()
        raise RuntimeError(f"MT5 copy_rates_range failed: {error}")

    if len(rates) == 0:
        logger.warning(
            "No historical bars returned",
            extra={"pair": pair.value, "date_from": str(date_from), "date_to": str(date_to)}
        )
        return []

    bars: list[OHLCVBar] = []
    for rate in rates:
        bar_open_broker = datetime.fromtimestamp(rate["time"])
        bar_open_utc  = mt5_server_to_utc(bar_open_broker, broker_tz_str)
        bar_close_utc = bar_open_utc + timedelta(seconds=timeframe_sec)

        spread_points = int(rate["spread"]) if rate["spread"] else 0
        point_size = 0.001 if pair == Pair.USDJPY else 0.00001
        spread_price = Decimal(str(round(spread_points * point_size, 5)))

        open_bid  = Decimal(str(round(float(rate["open"]),  5)))
        high_bid  = Decimal(str(round(float(rate["high"]),  5)))
        low_bid   = Decimal(str(round(float(rate["low"]),   5)))
        close_bid = Decimal(str(round(float(rate["close"]), 5)))

        bars.append(OHLCVBar(
            pair=pair,
            utc_open=bar_open_utc,
            utc_close=bar_close_utc,
            timeframe_sec=timeframe_sec,
            open_bid=open_bid,
            high_bid=high_bid,
            low_bid=low_bid,
            close_bid=close_bid,
            open_ask=open_bid  + spread_price,
            high_ask=high_bid  + spread_price,
            low_ask=low_bid    + spread_price,
            close_ask=close_bid + spread_price,
            volume=int(rate["tick_volume"]),
            is_complete=True,   # historical bars are always complete
            source=BarSource.BROKER_HISTORICAL,
        ))

    logger.info(
        "Fetched historical bars",
        extra={"pair": pair.value, "count": len(bars),
               "from": format_utc(bars[0].utc_open),
               "to": format_utc(bars[-1].utc_open)}
    )
    return bars
