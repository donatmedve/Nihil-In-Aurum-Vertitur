# broker/adapter.py
# BrokerAdapter ABC and IdempotentOrderSubmitter.
# All broker communication goes through this interface.
# Direct imports of MetaTrader5 or oandapyV20 are ONLY in concrete adapter files.

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from shared.schemas import (
    Order, OrderResult, OrderStatus, Position, AccountState, Pair, Side
)
from data.time_utils import now_utc, format_utc

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_RETRY_DELAY_SECONDS = 1.0


class BrokerAdapter(ABC):
    """
    Abstract interface over all brokers.
    Concrete implementations: MT5Adapter, OANDAAdapter.
    NEVER call MT5 or OANDA APIs directly outside of concrete adapter files.
    """

    @abstractmethod
    def get_open_positions(self) -> list[Position]:
        """Fetch all open positions from broker. Source of truth on reconnect."""
        ...

    @abstractmethod
    def get_pending_orders(self) -> list[Order]:
        """Fetch all pending (not yet filled) orders from broker."""
        ...

    @abstractmethod
    def send_order(self, order: Order) -> OrderResult:
        """
        Submit a single order to the broker.
        Must NOT generate or set client_order_id — that is set before this call.
        Must return raw_broker_response as a dict, always.
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order by broker order ID. Returns True if cancelled."""
        ...

    @abstractmethod
    def get_account_state(self) -> AccountState:
        """Fetch current account equity, balance, margin. Max staleness: 60s."""
        ...

    @abstractmethod
    def get_price(self, pair: Pair) -> tuple[Decimal, Decimal]:
        """Return current (bid, ask) for a pair. Never cache this."""
        ...

    @abstractmethod
    def modify_stop_loss(self, position_id: str, new_sl: Decimal) -> bool:
        """Modify stop loss on an open position. Returns True on success."""
        ...

    @abstractmethod
    def close_position(self, position_id: str, units: Optional[int] = None) -> OrderResult:
        """
        Close an open position. If units=None, close the full position.
        Partial close if units < position.units.
        """
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Returns True if the broker connection is healthy."""
        ...


class IdempotentOrderSubmitter:
    """
    Prevents duplicate orders on reconnect or retry.
    The client_order_id (UUID4) is the idempotency key.

    PROTOCOL — in this exact order:
      1. Check if client_order_id already exists in DB with non-terminal status
         → If yes: log warning, return existing result. Do NOT resubmit.
      2. Write order with status=PENDING_SUBMIT to DB (WAL fsync)
      3. Send to broker
      4. Update order status in DB based on broker response
      5. On all retries exhausted: set DEAD_LETTERED, log CRITICAL, alert

    NEVER auto-resubmit PENDING_SUBMIT orders found on restart.
    Crash recovery must flag them UNCONFIRMED for human review.
    """

    def __init__(self, adapter: BrokerAdapter, state_store, model_version: str):
        self._adapter = adapter
        self._state = state_store
        self._model_version = model_version

    def submit(self, order: Order, signal_confidence: float = 0.0) -> OrderResult:
        """
        Submit an order idempotently.
        Sets model_version and signal_confidence on the order before persisting.
        """
        order.model_version = self._model_version
        order.signal_confidence = signal_confidence

        # ---- Step 1: Idempotency check ----
        existing = self._state.get_order(order.client_order_id)
        if existing is not None:
            terminal = {OrderStatus.FILLED, OrderStatus.CANCELLED,
                        OrderStatus.REJECTED, OrderStatus.DEAD_LETTERED}
            if existing.status not in terminal:
                logger.warning(
                    "Duplicate order submission blocked",
                    extra={
                        "client_order_id": order.client_order_id,
                        "existing_status": existing.status.value,
                    }
                )
                return OrderResult(
                    client_order_id=order.client_order_id,
                    broker_order_id=existing.broker_order_id,
                    status=existing.status,
                    filled_price=existing.filled_price,
                    filled_units=None,
                    rejection_reason="Blocked: duplicate submission",
                    raw_broker_response={},
                )

        # ---- Step 2: Write to DB BEFORE sending ----
        order.status = OrderStatus.PENDING_SUBMIT
        order.submitted_at = now_utc()
        self._state.upsert_order(order)
        self._state.fsync()     # CRITICAL: must be on disk before network call

        logger.info(
            "Order persisted, submitting to broker",
            extra={
                "client_order_id": order.client_order_id,
                "pair": order.pair.value,
                "side": order.side.value,
                "units": order.units,
                "stop_loss": float(order.stop_loss_price) if order.stop_loss_price else None,
            }
        )

        # ---- Step 3: Send with retry ----
        result = self._send_with_retry(order)

        # ---- Step 4: Update DB with result ----
        if result.status == OrderStatus.FILLED:
            order.status = OrderStatus.FILLED
            order.broker_order_id = result.broker_order_id
            order.filled_price = result.filled_price
            order.filled_at = now_utc()
            if result.filled_price and order.limit_price:
                from shared.schemas import pip_multiplier
                slip_raw = abs(float(result.filled_price) - float(order.limit_price))
                order.slippage_pips = Decimal(str(slip_raw * pip_multiplier(order.pair)))
            logger.info(
                "Order filled",
                extra={
                    "client_order_id": order.client_order_id,
                    "broker_order_id": result.broker_order_id,
                    "fill_price": float(result.filled_price),
                    "slippage_pips": float(order.slippage_pips) if order.slippage_pips else 0,
                }
            )

        elif result.status == OrderStatus.SUBMITTED:
            order.status = OrderStatus.SUBMITTED
            order.broker_order_id = result.broker_order_id

        elif result.status == OrderStatus.REJECTED:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = result.rejection_reason
            logger.error(
                "Order rejected by broker",
                extra={
                    "client_order_id": order.client_order_id,
                    "reason": result.rejection_reason,
                    "raw": result.raw_broker_response,
                }
            )

        else:  # DEAD_LETTERED
            order.status = OrderStatus.DEAD_LETTERED
            order.rejection_reason = result.rejection_reason
            logger.critical(
                "Order dead-lettered — human review required",
                extra={
                    "client_order_id": order.client_order_id,
                    "reason": result.rejection_reason,
                }
            )

        self._state.upsert_order(order)
        return result

    def _send_with_retry(self, order: Order) -> OrderResult:
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                result = self._adapter.send_order(order)
                # Broker rejections are terminal — do not retry
                if result.status == OrderStatus.REJECTED:
                    return result
                # Success
                if result.status in (OrderStatus.FILLED, OrderStatus.SUBMITTED):
                    return result
            except Exception as exc:
                last_exc = exc
                delay = BASE_RETRY_DELAY_SECONDS * (2 ** attempt)
                logger.warning(
                    "Order submission exception, will retry",
                    extra={
                        "attempt": attempt + 1,
                        "max_retries": MAX_RETRIES,
                        "delay_s": delay,
                        "error": str(exc),
                        "client_order_id": order.client_order_id,
                    }
                )
                time.sleep(delay)

        return OrderResult(
            client_order_id=order.client_order_id,
            broker_order_id=None,
            status=OrderStatus.DEAD_LETTERED,
            filled_price=None,
            filled_units=None,
            rejection_reason=f"All {MAX_RETRIES} retries exhausted. Last error: {last_exc}",
            raw_broker_response={},
        )
