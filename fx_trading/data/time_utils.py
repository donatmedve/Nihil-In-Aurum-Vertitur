# data/time_utils.py
# UTC enforcement and broker-time conversion utilities.
# ALL datetimes in this system are UTC with explicit tzinfo.
# Broker-local time is converted HERE, at the ingestion boundary, and nowhere else.

from datetime import datetime, timezone, timedelta
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from shared.schemas import assert_utc


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def mt5_server_to_utc(dt_naive: datetime, broker_tz_str: str) -> datetime:
    """
    Convert a naive datetime from the MT5 API to UTC.

    MT5 returns naive datetimes in broker server local time (often UTC+2/+3).
    You MUST know your broker's timezone string. Check your broker's documentation.

    Common values:
      "Etc/GMT-2"        → UTC+2, no DST (many ECN brokers)
      "Europe/Helsinki"  → UTC+2/+3 with Finnish DST rules
      "America/New_York" → for US-based brokers

    Raises ZoneInfoNotFoundError if the timezone string is invalid.
    On DST ambiguity, zoneinfo uses fold=0 by default (first occurrence).
    """
    assert dt_naive.tzinfo is None, (
        f"Expected naive datetime from MT5, got tz-aware: {dt_naive}. "
        f"MT5 always returns naive datetimes. Check your ingestion code."
    )

    # Use stdlib zoneinfo (no pytz required)
    broker_tz = ZoneInfo(broker_tz_str)
    localized = dt_naive.replace(tzinfo=broker_tz)
    return localized.astimezone(timezone.utc)


def oanda_str_to_utc(ts_str: str) -> datetime:
    """
    Parse an OANDA v20 RFC3339 timestamp string to UTC datetime.

    OANDA returns strings like: "2024-01-15T13:45:00.000000000Z"
    The 'Z' suffix means UTC. Always.
    """
    # Handle nanosecond precision: truncate to microseconds
    ts_str = ts_str.rstrip("Z")
    if "." in ts_str:
        base, frac = ts_str.split(".")
        frac = frac[:6]  # truncate to microseconds
        ts_str = f"{base}.{frac}"
    else:
        ts_str = ts_str

    dt = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
    return assert_utc(dt, context="oanda_str_to_utc")


def now_utc() -> datetime:
    """Always use this instead of datetime.now() or datetime.utcnow()."""
    return datetime.now(timezone.utc)


def utc_from_timestamp(ts: float) -> datetime:
    """Convert a Unix timestamp (float seconds) to UTC datetime."""
    return datetime.fromtimestamp(ts, tz=timezone.utc)


# ---------------------------------------------------------------------------
# Market session helpers
# ---------------------------------------------------------------------------

# Market hours in UTC. These are approximate — broker-specific sessions may differ.
# Sydney:  21:00 – 06:00 UTC (Sun–Fri)
# Tokyo:   23:00 – 08:00 UTC
# London:  07:00 – 16:00 UTC
# New York: 12:00 – 21:00 UTC
# Overlap: 12:00 – 16:00 UTC (London/NY — highest liquidity)

SESSIONS = {
    "sydney":   (21, 6),
    "tokyo":    (23, 8),
    "london":   (7, 16),
    "new_york": (12, 21),
    "overlap":  (12, 16),
}


def get_active_sessions(utc_dt: datetime) -> list[str]:
    """Return list of active session names for a given UTC datetime."""
    assert_utc(utc_dt, context="get_active_sessions")
    h = utc_dt.hour
    active = []
    for name, (open_h, close_h) in SESSIONS.items():
        if open_h < close_h:
            if open_h <= h < close_h:
                active.append(name)
        else:  # wraps midnight
            if h >= open_h or h < close_h:
                active.append(name)
    return active


def is_market_open(utc_dt: datetime) -> bool:
    """
    Returns True if FX market is open for major pairs.
    Market is closed: Friday 21:00 UTC to Sunday 21:00 UTC.
    Note: individual brokers may vary by minutes. This is conservative.
    """
    assert_utc(utc_dt, context="is_market_open")
    weekday = utc_dt.weekday()  # 0=Monday, 6=Sunday
    hour = utc_dt.hour

    # Saturday: always closed
    if weekday == 5:
        return False
    # Friday after 21:00 UTC: closed
    if weekday == 4 and hour >= 21:
        return False
    # Sunday before 21:00 UTC: closed
    if weekday == 6 and hour < 21:
        return False
    return True


def is_weekend_gap_open(utc_dt: datetime) -> bool:
    """
    Returns True if this datetime is within the first bar after weekend open.
    Use to flag gap-open bars for special handling.
    Sunday 21:00–21:05 UTC (for 5m bars).
    """
    assert_utc(utc_dt, context="is_weekend_gap_open")
    return utc_dt.weekday() == 6 and utc_dt.hour == 21 and utc_dt.minute < 5


# ---------------------------------------------------------------------------
# Bar time helpers
# ---------------------------------------------------------------------------

def floor_to_bar(utc_dt: datetime, timeframe_sec: int) -> datetime:
    """
    Floor a datetime to the start of its containing bar.
    E.g., 13:47:23 UTC on 5m bars → 13:45:00 UTC
    """
    assert_utc(utc_dt, context="floor_to_bar")
    ts = utc_dt.timestamp()
    floored_ts = (ts // timeframe_sec) * timeframe_sec
    return datetime.fromtimestamp(floored_ts, tz=timezone.utc)


def bar_open_times(
    start_utc: datetime,
    end_utc: datetime,
    timeframe_sec: int,
) -> list[datetime]:
    """
    Generate all expected bar open times between start and end.
    Used for gap detection: compare against actual bars received.
    Only generates times during market hours.
    """
    assert_utc(start_utc, context="bar_open_times start")
    assert_utc(end_utc, context="bar_open_times end")

    times = []
    current = floor_to_bar(start_utc, timeframe_sec)
    step = timedelta(seconds=timeframe_sec)

    while current < end_utc:
        if is_market_open(current):
            times.append(current)
        current += step

    return times


def format_utc(dt: datetime) -> str:
    """Canonical ISO 8601 UTC string for logging and storage."""
    assert_utc(dt, context="format_utc")
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def parse_utc(s: str) -> datetime:
    """Parse a canonical UTC string back to datetime."""
    dt = datetime.fromisoformat(s.rstrip("Z")).replace(tzinfo=timezone.utc)
    return assert_utc(dt, context="parse_utc")
