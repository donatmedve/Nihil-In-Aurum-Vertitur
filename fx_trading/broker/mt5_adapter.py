# broker/mt5_adapter.py
# MetaTrader5 concrete adapter.
# This is the ONLY file that imports MetaTrader5.
# All other modules use BrokerAdapter (the ABC).

import logging
from decimal import Decimal
from typing import Optional

from shared.schemas import (
    Order, OrderResult, OrderStatus, Position, PositionStatus,
    AccountState, Pair, Side, OrderType, assert_utc
)
from data.time_utils import mt5_server_to_utc, now_utc
from broker.adapter import BrokerAdapter

logger = logging.getLogger(__name__)

# MT5 pair name mapping: internal Pair enum → broker symbol string
DEFAULT_SYMBOL_MAP = {
    Pair.EURUSD: "EURUSD",
    Pair.GBPUSD: "GBPUSD",
    Pair.USDJPY: "USDJPY",
}


class MT5Adapter(BrokerAdapter):
    """
    MetaTrader5 Python API adapter.
    broker_tz_str: timezone string for your specific broker's server time.
    Check your broker's documentation — this is NOT always UTC+2.
    """

    def __init__(
        self,
        broker_tz_str: str,
        symbol_map: Optional[dict] = None,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
    ):
        self._broker_tz = broker_tz_str
        self._symbol_map = symbol_map or DEFAULT_SYMBOL_MAP
        self._reverse_symbol_map = {v: k for k, v in self._symbol_map.items()}
        self._login = login
        self._password = password
        self._server = server
        self._connected = False

    def _ensure_connected(self) -> None:
        """Lazy connect. Reconnects if needed."""
        try:
            import MetaTrader5 as mt5
        except ImportError:
            raise ImportError("MetaTrader5 package required: pip install MetaTrader5")

        if not self._connected:
            kwargs = {}
            if self._login:
                kwargs["login"] = self._login
            if self._password:
                kwargs["password"] = self._password
            if self._server:
                kwargs["server"] = self._server

            if not mt5.initialize(**kwargs):
                raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")
            self._connected = True
            logger.info("MT5 connected", extra={"server": self._server})

        # Verify connection is still alive
        info = mt5.account_info()
        if info is None:
            self._connected = False
            raise RuntimeError(f"MT5 connection lost: {mt5.last_error()}")

    def is_connected(self) -> bool:
        try:
            self._ensure_connected()
            return True
        except Exception:
            return False

    def get_account_state(self) -> AccountState:
        import MetaTrader5 as mt5
        self._ensure_connected()

        info = mt5.account_info()
        if info is None:
            raise RuntimeError(f"MT5 account_info() failed: {mt5.last_error()}")

        return AccountState(
            equity_usd=Decimal(str(info.equity)),
            balance_usd=Decimal(str(info.balance)),
            margin_used_usd=Decimal(str(info.margin)),
            margin_free_usd=Decimal(str(info.margin_free)),
            as_of_utc=now_utc(),
        )

    def get_price(self, pair: Pair) -> tuple[Decimal, Decimal]:
        import MetaTrader5 as mt5
        self._ensure_connected()

        symbol = self._symbol_map[pair]
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"MT5 symbol_info_tick({symbol}) failed: {mt5.last_error()}")

        return Decimal(str(tick.bid)), Decimal(str(tick.ask))

    def get_open_positions(self) -> list[Position]:
        import MetaTrader5 as mt5
        self._ensure_connected()

        mt5_positions = mt5.positions_get()
        if mt5_positions is None:
            error = mt5.last_error()
            if error[0] == 1:   # no positions — not an error
                return []
            raise RuntimeError(f"MT5 positions_get() failed: {error}")

        positions = []
        for p in mt5_positions:
            pair = self._reverse_symbol_map.get(p.symbol)
            if pair is None:
                logger.warning("Unknown symbol in MT5 positions", extra={"symbol": p.symbol})
                continue

            side = Side.BUY if p.type == 0 else Side.SELL
            # MT5 open time is in broker server time (naive)
            entry_utc = mt5_server_to_utc(
                __import__("datetime").datetime.fromtimestamp(p.time),
                self._broker_tz
            )

            # client_order_id is stored in the position comment (set during order submission)
            client_order_id = p.comment if p.comment else f"mt5_{p.ticket}"

            positions.append(Position(
                position_id=str(p.ticket),
                client_order_id=client_order_id,
                pair=pair,
                side=side,
                units=int(p.volume * 100_000),  # lots → units
                entry_price=Decimal(str(p.price_open)),
                entry_utc=entry_utc,
                stop_loss_price=Decimal(str(p.sl)) if p.sl else Decimal("0"),
                take_profit_price=Decimal(str(p.tp)) if p.tp else None,
                unrealized_pnl_usd=Decimal(str(p.profit)),
                swap_accumulated_usd=Decimal(str(p.swap)),
                status=PositionStatus.OPEN,
            ))

        return positions

    def get_pending_orders(self) -> list[Order]:
        import MetaTrader5 as mt5
        self._ensure_connected()

        mt5_orders = mt5.orders_get()
        if mt5_orders is None:
            return []

        orders = []
        for o in mt5_orders:
            pair = self._reverse_symbol_map.get(o.symbol)
            if pair is None:
                continue

            orders.append(Order(
                client_order_id=o.comment if o.comment else f"mt5_{o.ticket}",
                broker_order_id=str(o.ticket),
                pair=pair,
                side=Side.BUY if o.type in (0, 2, 4) else Side.SELL,
                order_type=OrderType.LIMIT,
                units=int(o.volume_initial * 100_000),
                limit_price=Decimal(str(o.price_open)),
                stop_loss_price=Decimal(str(o.sl)) if o.sl else None,
                take_profit_price=Decimal(str(o.tp)) if o.tp else None,
                status=OrderStatus.SUBMITTED,
            ))
        return orders

    def send_order(self, order: Order) -> OrderResult:
        import MetaTrader5 as mt5
        self._ensure_connected()

        symbol = self._symbol_map[order.pair]
        lots = order.units / 100_000

        order_type_map = {
            (OrderType.MARKET, Side.BUY):  mt5.ORDER_TYPE_BUY,
            (OrderType.MARKET, Side.SELL): mt5.ORDER_TYPE_SELL,
            (OrderType.LIMIT, Side.BUY):   mt5.ORDER_TYPE_BUY_LIMIT,
            (OrderType.LIMIT, Side.SELL):  mt5.ORDER_TYPE_SELL_LIMIT,
        }
        mt5_type = order_type_map.get((order.order_type, order.side))
        if mt5_type is None:
            return OrderResult(
                client_order_id=order.client_order_id,
                broker_order_id=None,
                status=OrderStatus.REJECTED,
                filled_price=None,
                filled_units=None,
                rejection_reason=f"Unsupported order type/side: {order.order_type}/{order.side}",
                raw_broker_response={},
            )

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": round(lots, 2),
            "type": mt5_type,
            "sl": float(order.stop_loss_price) if order.stop_loss_price else 0.0,
            "tp": float(order.take_profit_price) if order.take_profit_price else 0.0,
            # Store client_order_id in comment for reconciliation after restart
            # MT5 comment field max 31 chars
            "comment": order.client_order_id[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if order.order_type == OrderType.LIMIT and order.limit_price:
            request["price"] = float(order.limit_price)

        logger.debug("MT5 order_send request", extra={"request": request})
        result = mt5.order_send(request)

        if result is None:
            return OrderResult(
                client_order_id=order.client_order_id,
                broker_order_id=None,
                status=OrderStatus.DEAD_LETTERED,
                filled_price=None,
                filled_units=None,
                rejection_reason=f"MT5 order_send returned None: {mt5.last_error()}",
                raw_broker_response={},
            )

        raw = result._asdict()
        logger.debug("MT5 order_send response", extra={"raw": raw})

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return OrderResult(
                client_order_id=order.client_order_id,
                broker_order_id=str(result.order),
                status=OrderStatus.FILLED,
                filled_price=Decimal(str(result.price)),
                filled_units=order.units,
                rejection_reason=None,
                raw_broker_response=raw,
            )
        else:
            return OrderResult(
                client_order_id=order.client_order_id,
                broker_order_id=None,
                status=OrderStatus.REJECTED,
                filled_price=None,
                filled_units=None,
                rejection_reason=f"MT5 retcode={result.retcode}: {result.comment}",
                raw_broker_response=raw,
            )

    def cancel_order(self, order_id: str) -> bool:
        import MetaTrader5 as mt5
        self._ensure_connected()

        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": int(order_id),
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info("Order cancelled", extra={"broker_order_id": order_id})
            return True
        logger.error(
            "Order cancel failed",
            extra={"broker_order_id": order_id, "result": result._asdict() if result else None}
        )
        return False

    def modify_stop_loss(self, position_id: str, new_sl: Decimal) -> bool:
        import MetaTrader5 as mt5
        self._ensure_connected()

        positions = mt5.positions_get(ticket=int(position_id))
        if not positions:
            logger.error("Position not found for SL modification", extra={"position_id": position_id})
            return False

        pos = positions[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(position_id),
            "sl": float(new_sl),
            "tp": pos.tp,
        }
        result = mt5.order_send(request)
        success = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        if not success:
            logger.error(
                "SL modification failed",
                extra={"position_id": position_id, "new_sl": float(new_sl),
                       "result": result._asdict() if result else None}
            )
        return success

    def close_position(self, position_id: str, units: Optional[int] = None) -> OrderResult:
        import MetaTrader5 as mt5
        self._ensure_connected()

        positions = mt5.positions_get(ticket=int(position_id))
        if not positions:
            return OrderResult(
                client_order_id=f"close_{position_id}",
                broker_order_id=None,
                status=OrderStatus.REJECTED,
                filled_price=None,
                filled_units=None,
                rejection_reason=f"Position {position_id} not found",
                raw_broker_response={},
            )

        pos = positions[0]
        symbol = pos.symbol
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        lots = (units / 100_000) if units else pos.volume

        tick = mt5.symbol_info_tick(symbol)
        price = tick.bid if pos.type == 0 else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": round(lots, 2),
            "type": close_type,
            "position": int(position_id),
            "price": price,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "comment": f"close_{position_id[:20]}",
        }

        result = mt5.order_send(request)
        raw = result._asdict() if result else {}

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return OrderResult(
                client_order_id=f"close_{position_id}",
                broker_order_id=str(result.order),
                status=OrderStatus.FILLED,
                filled_price=Decimal(str(result.price)),
                filled_units=int(lots * 100_000),
                rejection_reason=None,
                raw_broker_response=raw,
            )
        return OrderResult(
            client_order_id=f"close_{position_id}",
            broker_order_id=None,
            status=OrderStatus.REJECTED,
            filled_price=None,
            filled_units=None,
            rejection_reason=f"Close failed: retcode={result.retcode if result else 'None'}",
            raw_broker_response=raw,
        )
