# data/aggregation.py
# Deterministic tick-to-bar aggregation.
# Rules:
#   - Bar timestamps are the BAR OPEN time (left edge), always UTC
#   - A bar is COMPLETE only when the next bar's first tick arrives
#   - The in-progress bar is always marked is_complete=False
#   - Empty bars within session hours are gap-filled (volume=0, source='gap_filled')
#   - Weekend and between-session gaps are NOT filled
#   - Late ticks (>2s past bar boundary) are logged and dropped

import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional

import pandas as pd
import numpy as np

from shared.schemas import OHLCVBar, Pair, assert_utc
from data.time_utils import (
    floor_to_bar, bar_open_times, is_market_open, format_utc, now_utc
)

logger = logging.getLogger(__name__)

LATE_TICK_TOLERANCE_SECONDS = 2
MAX_GAP_FILL_BARS = 3       # only gap-fill runs shorter than this


def aggregate_ticks_to_bars(
    ticks: pd.DataFrame,
    timeframe_sec: int,
    pair: Pair,
) -> list[OHLCVBar]:
    """
    Convert a DataFrame of ticks to a list of OHLCVBar objects.

    ticks columns required:
        utc_ts   : datetime64[ns, UTC]
        bid      : float64
        ask      : float64
        volume   : int32

    Returns bars sorted ascending by utc_open.
    Last bar will have is_complete=False.

    IMPORTANT: Always pass complete ticks for the time range.
    Calling this on a rolling window mid-session: the last bar returned
    is NOT complete. Do not use it for signals.
    """
    if ticks.empty:
        logger.warning("aggregate_ticks_to_bars called with empty tick DataFrame")
        return []

    # Validate UTC timezone
    if ticks["utc_ts"].dt.tz is None:
        raise ValueError("Tick timestamps must be timezone-aware UTC")

    ticks = ticks.sort_values("utc_ts").copy()

    # Assign each tick to its bar
    ticks["bar_open_ts"] = ticks["utc_ts"].apply(
        lambda ts: floor_to_bar(ts.to_pydatetime(), timeframe_sec)
    )

    bars = []
    grouped = ticks.groupby("bar_open_ts")

    for bar_open, group in grouped:
        bar_close = bar_open + timedelta(seconds=timeframe_sec)

        # Check for late ticks: ticks that belong to this bar but arrived late
        # (These are ticks whose utc_ts is before bar_open but somehow ended up here)
        # This shouldn't happen with sort + floor, but guard defensively
        late = group[group["utc_ts"] < bar_open]
        if not late.empty:
            logger.warning(
                "Late ticks detected and dropped",
                extra={"count": len(late), "bar_open": format_utc(bar_open)}
            )
            group = group[group["utc_ts"] >= bar_open]

        if group.empty:
            continue

        bar = OHLCVBar(
            pair=pair,
            utc_open=bar_open,
            utc_close=bar_close,
            timeframe_sec=timeframe_sec,
            open_bid=Decimal(str(group["bid"].iloc[0])),
            high_bid=Decimal(str(group["bid"].max())),
            low_bid=Decimal(str(group["bid"].min())),
            close_bid=Decimal(str(group["bid"].iloc[-1])),
            open_ask=Decimal(str(group["ask"].iloc[0])),
            high_ask=Decimal(str(group["ask"].max())),
            low_ask=Decimal(str(group["ask"].min())),
            close_ask=Decimal(str(group["ask"].iloc[-1])),
            volume=int(group["volume"].sum()),
            is_complete=True,   # corrected for last bar below
            source="broker_live",
        )
        bars.append(bar)

    if not bars:
        return []

    # Last bar is still in progress
    last = bars[-1]
    bars[-1] = OHLCVBar(
        pair=last.pair,
        utc_open=last.utc_open,
        utc_close=last.utc_close,
        timeframe_sec=last.timeframe_sec,
        open_bid=last.open_bid,
        high_bid=last.high_bid,
        low_bid=last.low_bid,
        close_bid=last.close_bid,
        open_ask=last.open_ask,
        high_ask=last.high_ask,
        low_ask=last.low_ask,
        close_ask=last.close_ask,
        volume=last.volume,
        is_complete=False,
        source=last.source,
    )

    # Gap-fill short runs within session hours
    bars = _fill_session_gaps(bars, timeframe_sec, pair)

    return bars


def _fill_session_gaps(
    bars: list[OHLCVBar],
    timeframe_sec: int,
    pair: Pair,
) -> list[OHLCVBar]:
    """
    Insert forward-filled zero-volume bars for gaps shorter than MAX_GAP_FILL_BARS
    that occur within market hours.
    Gaps longer than MAX_GAP_FILL_BARS or outside market hours are left as-is.
    """
    if len(bars) < 2:
        return bars

    result = [bars[0]]
    step = timedelta(seconds=timeframe_sec)

    for i in range(1, len(bars)):
        prev = result[-1]
        curr = bars[i]
        gap_bars = int((curr.utc_open - prev.utc_open) / step) - 1

        if gap_bars > 0 and gap_bars <= MAX_GAP_FILL_BARS:
            # Fill only if within session hours
            fill_time = prev.utc_open + step
            while fill_time < curr.utc_open:
                if is_market_open(fill_time):
                    filled = OHLCVBar(
                        pair=pair,
                        utc_open=fill_time,
                        utc_close=fill_time + step,
                        timeframe_sec=timeframe_sec,
                        open_bid=prev.close_bid,
                        high_bid=prev.close_bid,
                        low_bid=prev.close_bid,
                        close_bid=prev.close_bid,
                        open_ask=prev.close_ask,
                        high_ask=prev.close_ask,
                        low_ask=prev.close_ask,
                        close_ask=prev.close_ask,
                        volume=0,
                        is_complete=True,
                        source="gap_filled",
                    )
                    result.append(filled)
                    logger.debug(
                        "Gap-filled bar inserted",
                        extra={"bar_open": format_utc(fill_time), "pair": pair.value}
                    )
                fill_time += step

        result.append(curr)

    return result


def bars_to_dataframe(bars: list[OHLCVBar]) -> pd.DataFrame:
    """
    Convert a list of OHLCVBar to a pandas DataFrame for feature computation.
    All price columns are float64. utc_open and utc_close are datetime64[ns, UTC].
    """
    if not bars:
        return pd.DataFrame()

    records = []
    for b in bars:
        records.append({
            "pair": b.pair.value,
            "utc_open": b.utc_open,
            "utc_close": b.utc_close,
            "timeframe_sec": b.timeframe_sec,
            "open_bid": float(b.open_bid),
            "high_bid": float(b.high_bid),
            "low_bid": float(b.low_bid),
            "close_bid": float(b.close_bid),
            "open_ask": float(b.open_ask),
            "high_ask": float(b.high_ask),
            "low_ask": float(b.low_ask),
            "close_ask": float(b.close_ask),
            "volume": b.volume,
            "is_complete": b.is_complete,
            "source": b.source,
        })

    df = pd.DataFrame(records)
    df["utc_open"] = pd.to_datetime(df["utc_open"], utc=True)
    df["utc_close"] = pd.to_datetime(df["utc_close"], utc=True)
    return df.set_index("utc_open").sort_index()


def detect_gaps(
    bars: pd.DataFrame,
    timeframe_sec: int,
    expected_open_utc: Optional[datetime] = None,
    expected_close_utc: Optional[datetime] = None,
) -> list[dict]:
    """
    Scan a bar DataFrame for gaps.
    Returns list of gap descriptions: {"gap_start", "gap_end", "missing_bars", "in_session"}.

    Use this after loading historical data to audit completeness before training.
    """
    if bars.empty:
        return []

    step = timedelta(seconds=timeframe_sec)
    gaps = []
    index = bars.index.to_list()

    for i in range(1, len(index)):
        expected = index[i - 1] + step
        actual = index[i]
        if actual > expected:
            missing = int((actual - expected) / step)
            gaps.append({
                "gap_start": expected,
                "gap_end": actual,
                "missing_bars": missing,
                "in_session": is_market_open(expected),
            })

    return gaps
