# shared/schemas.py
# ============================================================
# CANONICAL DATA CONTRACTS
# Every module imports types from here. Never redefine these elsewhere.
# All datetimes are UTC and timezone-aware without exception.
# ============================================================

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional
import uuid


# ── Time Helpers (here so all modules can import without circular deps) ──────

def assert_utc(dt: datetime, context: str = "") -> datetime:
    """Assert datetime is UTC and timezone-aware. Call at every module boundary."""
    assert dt.tzinfo is not None, \
        f"Expected UTC datetime, got naive{f' [{context}]' if context else ''}: {dt}"
    offset = dt.utcoffset()
    assert offset is not None and offset.total_seconds() == 0, \
        f"Expected UTC (offset=0), got offset={offset}{f' [{context}]' if context else ''}"
    return dt


def now_utc() -> datetime:
    """Always use this instead of datetime.utcnow() which returns naive."""
    return datetime.now(timezone.utc)


# ── Enums ────────────────────────────────────────────────────

class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(Enum):
    PENDING_SUBMIT = "PENDING_SUBMIT"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    DEAD_LETTERED = "DEAD_LETTERED"
    UNCONFIRMED = "UNCONFIRMED"


class PositionStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    ORPHANED = "ORPHANED"
    BROKER_CLOSED = "BROKER_CLOSED"


class Pair(Enum):
    EURUSD = "EUR/USD"
    GBPUSD = "GBP/USD"
    USDJPY = "USD/JPY"


class CloseReason(Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    MANUAL = "manual"
    CIRCUIT_BREAKER = "circuit_breaker"
    RECONCILIATION = "reconciliation"


class BarSource(Enum):
    BROKER_HISTORICAL = "broker_historical"
    BROKER_LIVE = "broker_live"
    GAP_FILLED = "gap_filled"
    RECONSTRUCTED = "reconstructed"


# ── Market Data ───────────────────────────────────────────────

@dataclass(frozen=True)
class Tick:
    pair: Pair
    utc_ts: datetime
    bid: Decimal
    ask: Decimal
    volume: int

    def __post_init__(self):
        assert self.utc_ts.tzinfo is not None, \
            f"Tick.utc_ts must be tz-aware UTC, got naive: {self.utc_ts}"
        assert self.bid > 0, f"Bid must be positive, got {self.bid}"
        assert self.ask >= self.bid, \
            f"Ask must be >= bid, got ask={self.ask} bid={self.bid}"
        assert self.volume >= 0, f"Volume must be non-negative"


@dataclass(frozen=True)
class OHLCVBar:
    pair: Pair
    utc_open: datetime
    utc_close: datetime
    timeframe_sec: int
    open_bid: Decimal
    high_bid: Decimal
    low_bid: Decimal
    close_bid: Decimal
    open_ask: Decimal
    high_ask: Decimal
    low_ask: Decimal
    close_ask: Decimal
    volume: int
    is_complete: bool
    source: BarSource

    def __post_init__(self):
        assert self.utc_open.tzinfo is not None, "utc_open must be tz-aware"
        assert self.utc_close.tzinfo is not None, "utc_close must be tz-aware"
        assert self.utc_close > self.utc_open, "utc_close must be after utc_open"
        assert self.high_bid >= self.low_bid, "high_bid must be >= low_bid"
        assert self.high_ask >= self.low_ask, "high_ask must be >= low_ask"
        assert self.volume >= 0, "Volume must be non-negative"

    @property
    def mid_close(self) -> Decimal:
        """For display/logging ONLY. Never use as a fill price."""
        return (self.close_bid + self.close_ask) / 2

    @property
    def spread_pips(self) -> Decimal:
        pip_size = Decimal("0.01") if "JPY" in self.pair.value else Decimal("0.0001")
        return (self.close_ask - self.close_bid) / pip_size


# ── Orders ────────────────────────────────────────────────────

@dataclass
class Order:
    pair: Pair
    side: Side
    order_type: OrderType
    units: int
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    limit_price: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None
    broker_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING_SUBMIT
    submitted_at: Optional[datetime] = None
    filled_price: Optional[Decimal] = None
    filled_at: Optional[datetime] = None
    slippage_pips: Optional[Decimal] = None
    rejection_reason: Optional[str] = None
    raw_broker_response: Optional[dict] = None

    def __post_init__(self):
        assert self.units > 0, f"Units must be positive, got {self.units}"
        assert self.client_order_id, "client_order_id must not be empty"

    def is_terminal(self) -> bool:
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.DEAD_LETTERED,
        )


@dataclass
class OrderResult:
    client_order_id: str
    broker_order_id: Optional[str]
    status: OrderStatus
    filled_price: Optional[Decimal]
    filled_units: Optional[int]
    rejection_reason: Optional[str]
    raw_broker_response: dict


# ── Positions ─────────────────────────────────────────────────

@dataclass
class Position:
    position_id: str
    client_order_id: str
    pair: Pair
    side: Side
    units: int
    entry_price: Decimal
    entry_utc: datetime
    stop_loss_price: Decimal
    take_profit_price: Optional[Decimal] = None
    unrealized_pnl_usd: Decimal = Decimal("0")
    swap_accumulated_usd: Decimal = Decimal("0")
    status: PositionStatus = PositionStatus.OPEN
    closed_at: Optional[datetime] = None
    close_price: Optional[Decimal] = None
    realized_pnl_usd: Optional[Decimal] = None
    close_reason: Optional[CloseReason] = None
    model_version: Optional[str] = None
    signal_confidence: Optional[float] = None

    def __post_init__(self):
        assert self.units > 0, f"Units must be positive"
        assert self.entry_price > 0, "Entry price must be positive"
        assert self.entry_utc.tzinfo is not None, "entry_utc must be tz-aware"


# ── Account ───────────────────────────────────────────────────

@dataclass
class AccountState:
    equity_usd: Decimal
    balance_usd: Decimal
    margin_used_usd: Decimal
    margin_free_usd: Decimal
    as_of_utc: datetime

    def __post_init__(self):
        assert self.as_of_utc.tzinfo is not None, "as_of_utc must be tz-aware"
        assert self.equity_usd >= 0, "Equity cannot be negative"


# ── Trade Record ──────────────────────────────────────────────

@dataclass
class TradeRecord:
    """Immutable record written when a position closes."""
    position_id: str
    client_order_id: str
    pair: Pair
    side: Side
    units: int
    entry_price: Decimal
    entry_utc: datetime
    exit_price: Decimal
    exit_utc: datetime
    gross_pnl_usd: Decimal
    swap_usd: Decimal
    commission_usd: Decimal
    net_pnl_usd: Decimal
    close_reason: CloseReason
    model_version: str
    signal_confidence: Optional[float]
    slippage_entry_pips: Optional[Decimal]

    def __post_init__(self):
        assert self.exit_utc > self.entry_utc, "Exit must be after entry"


# ── Signal ────────────────────────────────────────────────────

@dataclass
class Signal:
    pair: Pair
    bar_utc: datetime
    direction: int          # -1=SHORT, 0=ABSTAIN, 1=LONG
    confidence: float
    model_version: str
    abstain_reason: Optional[str] = None

    def __post_init__(self):
        assert self.direction in (-1, 0, 1), f"Invalid direction: {self.direction}"
        assert 0.0 <= self.confidence <= 1.0, f"Confidence out of range: {self.confidence}"
        assert self.bar_utc.tzinfo is not None, "bar_utc must be tz-aware"


# ── Risk Decision ─────────────────────────────────────────────

@dataclass
class RiskDecision:
    trade_permitted: bool
    reason: str
    position_size_units: int = 0
    stop_distance_pips: float = 0.0
    stop_price: Decimal = Decimal("0")
    max_loss_usd: float = 0.0


# ── Pip Helpers ───────────────────────────────────────────────

def pip_size(pair: Pair) -> Decimal:
    """Return the pip size. JPY pairs: 0.01. All others: 0.0001."""
    if "JPY" in pair.value:
        return Decimal("0.01")
    return Decimal("0.0001")


def pip_multiplier(pair: Pair) -> float:
    """Return multiplier to convert price difference to pips."""
    if "JPY" in pair.value:
        return 100.0
    return 10000.0
