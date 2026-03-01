# state/store.py
# Persistent state layer. Source of truth. Survives kill -9.
# Uses SQLite with WAL mode and FULL synchronous — every commit is fsynced.
#
# RULES:
#   - All writes go through this module. No direct SQL outside state/.
#   - Write order: write to DB, then take action. Never the reverse.
#   - On read conflicts between broker and local: broker wins on positions/fills.
#   - Kill switch is checked here. If set, callers must halt.

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

from shared.schemas import (
    Order, OrderStatus, Position, PositionStatus, TradeRecord,
    Pair, Side, CloseReason, assert_utc
)
from data.time_utils import now_utc, format_utc, parse_utc

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS system_state (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    client_order_id     TEXT PRIMARY KEY,
    broker_order_id     TEXT,
    pair                TEXT NOT NULL,
    side                TEXT NOT NULL,
    order_type          TEXT NOT NULL,
    units               INTEGER NOT NULL,
    limit_price         REAL,
    stop_loss_price     REAL,
    take_profit_price   REAL,
    status              TEXT NOT NULL,
    submitted_at        TEXT,
    filled_price        REAL,
    filled_at           TEXT,
    slippage_pips       REAL,
    rejection_reason    TEXT,
    model_version       TEXT,
    signal_confidence   REAL,
    raw_broker_response TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    position_id         TEXT PRIMARY KEY,
    client_order_id     TEXT NOT NULL REFERENCES orders(client_order_id),
    pair                TEXT NOT NULL,
    side                TEXT NOT NULL,
    units               INTEGER NOT NULL,
    entry_price         REAL NOT NULL,
    entry_utc           TEXT NOT NULL,
    stop_loss_price     REAL NOT NULL,
    take_profit_price   REAL,
    unrealized_pnl_usd  REAL DEFAULT 0.0,
    swap_accumulated_usd REAL DEFAULT 0.0,
    status              TEXT NOT NULL DEFAULT 'OPEN',
    closed_at           TEXT,
    close_price         REAL,
    realized_pnl_usd    REAL,
    close_reason        TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trade_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id         TEXT NOT NULL,
    client_order_id     TEXT NOT NULL,
    pair                TEXT NOT NULL,
    side                TEXT NOT NULL,
    units               INTEGER NOT NULL,
    entry_price         REAL NOT NULL,
    entry_utc           TEXT NOT NULL,
    exit_price          REAL NOT NULL,
    exit_utc            TEXT NOT NULL,
    realized_pnl_usd    REAL NOT NULL,
    swap_usd            REAL NOT NULL DEFAULT 0.0,
    commission_usd      REAL NOT NULL DEFAULT 0.0,
    net_pnl_usd         REAL NOT NULL,
    close_reason        TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    signal_confidence   REAL
);

CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_trade_log_entry_utc ON trade_log(entry_utc);
"""

# System state keys
KEY_KILL_SWITCH = "kill_switch"
KEY_HIGH_WATER_MARK = "high_water_mark"
KEY_DAILY_PNL_REALIZED = "daily_pnl_realized"
KEY_DAILY_PNL_DATE = "daily_pnl_date"
KEY_CONSECUTIVE_LOSSES = "consecutive_losses"
KEY_BARS_SINCE_LOSS_LIMIT = "bars_since_loss_limit"
KEY_LAST_HEARTBEAT = "last_heartbeat_utc"
KEY_LAST_BAR_UTC = "last_bar_utc"
KEY_DEPLOYED_MODEL_SHA256 = "deployed_model_sha256"
KEY_DEPLOYED_PIPELINE_SHA256 = "deployed_pipeline_sha256"
KEY_DEPLOYED_MODEL_VERSION = "deployed_model_version"


class StateStore:
    """
    All persistent state in one place.
    Thread safety: SQLite WAL handles concurrent reads.
    Writes are single-threaded (single execution loop).
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA_SQL)
        logger.info("StateStore initialized", extra={"db_path": self._db_path})

    @contextmanager
    def _conn(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # System state (key-value)
    # ------------------------------------------------------------------

    def get_state(self, key: str, default=None) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM system_state WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else default

    def set_state(self, key: str, value: str) -> None:
        ts = format_utc(now_utc())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO system_state (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, str(value), ts)
            )

    def get_kill_switch(self) -> bool:
        return self.get_state(KEY_KILL_SWITCH, "false").lower() == "true"

    def set_kill_switch(self, active: bool, reason: str = "") -> None:
        self.set_state(KEY_KILL_SWITCH, "true" if active else "false")
        logger.critical(
            "Kill switch state changed",
            extra={"active": active, "reason": reason}
        )

    def get_high_water_mark(self) -> float:
        val = self.get_state(KEY_HIGH_WATER_MARK)
        return float(val) if val else 0.0

    def update_high_water_mark(self, equity: float) -> None:
        current = self.get_high_water_mark()
        if equity > current:
            self.set_state(KEY_HIGH_WATER_MARK, str(equity))

    def get_drawdown_from_peak(self, current_equity: float) -> float:
        hwm = self.get_high_water_mark()
        if hwm <= 0:
            return 0.0
        return max(0.0, (hwm - current_equity) / hwm)

    def get_consecutive_losses(self) -> int:
        val = self.get_state(KEY_CONSECUTIVE_LOSSES, "0")
        return int(val)

    def increment_consecutive_losses(self) -> int:
        current = self.get_consecutive_losses()
        new_val = current + 1
        self.set_state(KEY_CONSECUTIVE_LOSSES, str(new_val))
        self.set_state(KEY_BARS_SINCE_LOSS_LIMIT, "0")
        return new_val

    def reset_consecutive_losses(self) -> None:
        self.set_state(KEY_CONSECUTIVE_LOSSES, "0")

    def get_bars_since_loss_limit(self) -> int:
        val = self.get_state(KEY_BARS_SINCE_LOSS_LIMIT, "999")
        return int(val)

    def increment_bar_count(self) -> None:
        current = self.get_bars_since_loss_limit()
        self.set_state(KEY_BARS_SINCE_LOSS_LIMIT, str(current + 1))

    def get_daily_pnl(self, current_date: str) -> float:
        """Returns daily P&L for current_date. Resets to 0 on new day."""
        stored_date = self.get_state(KEY_DAILY_PNL_DATE, "")
        if stored_date != current_date:
            # New day: reset
            self.set_state(KEY_DAILY_PNL_REALIZED, "0.0")
            self.set_state(KEY_DAILY_PNL_DATE, current_date)
            return 0.0
        val = self.get_state(KEY_DAILY_PNL_REALIZED, "0.0")
        return float(val)

    def add_daily_pnl(self, amount: float, current_date: str) -> float:
        current = self.get_daily_pnl(current_date)
        new_val = current + amount
        self.set_state(KEY_DAILY_PNL_REALIZED, str(new_val))
        return new_val

    def set_heartbeat(self) -> None:
        self.set_state(KEY_LAST_HEARTBEAT, format_utc(now_utc()))

    def get_last_heartbeat(self) -> Optional[datetime]:
        val = self.get_state(KEY_LAST_HEARTBEAT)
        return parse_utc(val) if val else None

    def set_last_bar_utc(self, utc: datetime) -> None:
        assert_utc(utc, "set_last_bar_utc")
        self.set_state(KEY_LAST_BAR_UTC, format_utc(utc))

    def set_deployed_artifact(self, model_sha256: str, pipeline_sha256: str, version: str) -> None:
        self.set_state(KEY_DEPLOYED_MODEL_SHA256, model_sha256)
        self.set_state(KEY_DEPLOYED_PIPELINE_SHA256, pipeline_sha256)
        self.set_state(KEY_DEPLOYED_MODEL_VERSION, version)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def upsert_order(self, order: Order) -> None:
        """
        Write order to DB. This MUST be called before sending to broker.
        On conflict (same client_order_id): update all mutable fields.
        """
        ts = format_utc(now_utc())
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO orders (
                    client_order_id, broker_order_id, pair, side, order_type,
                    units, limit_price, stop_loss_price, take_profit_price,
                    status, submitted_at, filled_price, filled_at, slippage_pips,
                    rejection_reason, model_version, signal_confidence,
                    raw_broker_response, created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(client_order_id) DO UPDATE SET
                    broker_order_id=excluded.broker_order_id,
                    status=excluded.status,
                    submitted_at=excluded.submitted_at,
                    filled_price=excluded.filled_price,
                    filled_at=excluded.filled_at,
                    slippage_pips=excluded.slippage_pips,
                    rejection_reason=excluded.rejection_reason,
                    raw_broker_response=excluded.raw_broker_response,
                    updated_at=excluded.updated_at
            """, (
                order.client_order_id,
                order.broker_order_id,
                order.pair.value if order.pair else None,
                order.side.value if order.side else None,
                order.order_type.value if order.order_type else None,
                order.units,
                float(order.limit_price) if order.limit_price else None,
                float(order.stop_loss_price) if order.stop_loss_price else None,
                float(order.take_profit_price) if order.take_profit_price else None,
                order.status.value,
                format_utc(order.submitted_at) if order.submitted_at else None,
                float(order.filled_price) if order.filled_price else None,
                format_utc(order.filled_at) if order.filled_at else None,
                float(order.slippage_pips) if order.slippage_pips else None,
                order.rejection_reason,
                None,  # model_version — not on Order; stored on Position
                None,  # signal_confidence — not on Order; stored on Position
                None,   # raw_broker_response set separately if needed
                ts, ts,
            ))

    def get_order(self, client_order_id: str) -> Optional[Order]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM orders WHERE client_order_id = ?",
                (client_order_id,)
            ).fetchone()
        return _row_to_order(row) if row else None

    def get_orders_by_status(self, status: OrderStatus) -> list[Order]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM orders WHERE status = ? ORDER BY created_at",
                (status.value,)
            ).fetchall()
        return [_row_to_order(r) for r in rows]

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def upsert_position(self, pos: Position) -> None:
        ts = format_utc(now_utc())
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO positions (
                    position_id, client_order_id, pair, side, units,
                    entry_price, entry_utc, stop_loss_price, take_profit_price,
                    unrealized_pnl_usd, swap_accumulated_usd, status,
                    closed_at, close_price, realized_pnl_usd, close_reason,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(position_id) DO UPDATE SET
                    unrealized_pnl_usd=excluded.unrealized_pnl_usd,
                    swap_accumulated_usd=excluded.swap_accumulated_usd,
                    stop_loss_price=excluded.stop_loss_price,
                    take_profit_price=excluded.take_profit_price,
                    status=excluded.status,
                    closed_at=excluded.closed_at,
                    close_price=excluded.close_price,
                    realized_pnl_usd=excluded.realized_pnl_usd,
                    close_reason=excluded.close_reason,
                    updated_at=excluded.updated_at
            """, (
                pos.position_id, pos.client_order_id,
                pos.pair.value, pos.side.value, pos.units,
                float(pos.entry_price),
                format_utc(pos.entry_utc),
                float(pos.stop_loss_price),
                float(pos.take_profit_price) if pos.take_profit_price else None,
                float(pos.unrealized_pnl_usd),
                float(pos.swap_accumulated_usd),
                pos.status.value,
                format_utc(pos.closed_at) if pos.closed_at else None,
                float(pos.close_price) if pos.close_price else None,
                float(pos.realized_pnl_usd) if pos.realized_pnl_usd else None,
                pos.close_reason.value if pos.close_reason else None,
                ts, ts,
            ))

    def get_open_positions(self) -> list[Position]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE status = 'OPEN' ORDER BY entry_utc"
            ).fetchall()
        return [_row_to_position(r) for r in rows]

    def get_position(self, position_id: str) -> Optional[Position]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM positions WHERE position_id = ?", (position_id,)
            ).fetchone()
        return _row_to_position(row) if row else None

    # ------------------------------------------------------------------
    # Trade log (append-only)
    # ------------------------------------------------------------------

    def append_trade(self, trade: TradeRecord) -> None:
        """Append a closed trade to the immutable trade log."""
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO trade_log (
                    position_id, client_order_id, pair, side, units,
                    entry_price, entry_utc, exit_price, exit_utc,
                    realized_pnl_usd, swap_usd, commission_usd, net_pnl_usd,
                    close_reason, model_version, signal_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.position_id, trade.client_order_id,
                trade.pair.value, trade.side.value, trade.units,
                float(trade.entry_price), format_utc(trade.entry_utc),
                float(trade.exit_price), format_utc(trade.exit_utc),
                float(trade.realized_pnl_usd),
                float(trade.swap_usd),
                float(trade.commission_usd),
                float(trade.net_pnl_usd),
                trade.close_reason.value,
                trade.model_version,
                trade.signal_confidence,
            ))

    def get_trade_log(self, from_utc: Optional[str] = None) -> list[dict]:
        with self._conn() as conn:
            if from_utc:
                rows = conn.execute(
                    "SELECT * FROM trade_log WHERE entry_utc >= ? ORDER BY entry_utc",
                    (from_utc,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trade_log ORDER BY entry_utc"
                ).fetchall()
        return [dict(r) for r in rows]

    def fsync(self) -> None:
        """
        Force a WAL checkpoint to ensure data is on disk.
        Call after writing critical state (before network operations).
        """
        with self._conn() as conn:
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")


# ------------------------------------------------------------------
# Row deserializers
# ------------------------------------------------------------------

def _row_to_order(row: sqlite3.Row) -> Order:
    from shared.schemas import OrderType
    return Order(
        client_order_id=row["client_order_id"],
        broker_order_id=row["broker_order_id"],
        pair=Pair(row["pair"]) if row["pair"] else None,
        side=Side(row["side"]) if row["side"] else None,
        order_type=OrderType(row["order_type"]) if row["order_type"] else OrderType.MARKET,
        status=OrderStatus(row["status"]),
        units=row["units"],
        stop_loss_price=Decimal(str(row["stop_loss_price"])) if row["stop_loss_price"] else None,
        take_profit_price=Decimal(str(row["take_profit_price"])) if row["take_profit_price"] else None,
        filled_price=Decimal(str(row["filled_price"])) if row["filled_price"] else None,
        slippage_pips=Decimal(str(row["slippage_pips"])) if row["slippage_pips"] else None,
        rejection_reason=row["rejection_reason"],
    )


def _row_to_position(row: sqlite3.Row) -> Position:
    return Position(
        position_id=row["position_id"],
        client_order_id=row["client_order_id"],
        pair=Pair(row["pair"]),
        side=Side(row["side"]),
        units=row["units"],
        entry_price=Decimal(str(row["entry_price"])),
        entry_utc=parse_utc(row["entry_utc"]),
        stop_loss_price=Decimal(str(row["stop_loss_price"])),
        take_profit_price=Decimal(str(row["take_profit_price"])) if row["take_profit_price"] else None,
        unrealized_pnl_usd=Decimal(str(row["unrealized_pnl_usd"])),
        swap_accumulated_usd=Decimal(str(row["swap_accumulated_usd"])),
        status=PositionStatus(row["status"]),
        closed_at=parse_utc(row["closed_at"]) if row["closed_at"] else None,
        close_price=Decimal(str(row["close_price"])) if row["close_price"] else None,
        realized_pnl_usd=Decimal(str(row["realized_pnl_usd"])) if row["realized_pnl_usd"] else None,
        close_reason=CloseReason(row["close_reason"]) if row["close_reason"] else None,
    )
