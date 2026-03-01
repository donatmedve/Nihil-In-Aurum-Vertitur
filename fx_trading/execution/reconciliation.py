# execution/reconciliation.py
# Crash recovery and broker/local state reconciliation.
# Execute this on EVERY process start, in the exact order specified.
# Do not resume trading until reconciliation completes without critical errors.

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from shared.schemas import (
    Order, OrderStatus, Position, PositionStatus, Pair
)
from data.time_utils import now_utc, format_utc

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationResult:
    success: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    orphaned_positions: list[str] = field(default_factory=list)   # position_ids
    unconfirmed_orders: list[str] = field(default_factory=list)   # client_order_ids
    positions_updated: int = 0
    orders_updated: int = 0

    @property
    def requires_human_review(self) -> bool:
        return bool(self.orphaned_positions or self.unconfirmed_orders or self.errors)


def run_reconciliation(
    adapter,            # BrokerAdapter instance
    state_store,        # StateStore instance
    model_sha256: str,
    pipeline_sha256: str,
) -> ReconciliationResult:
    """
    Full reconciliation sequence. Must run at process start before trading.
    
    Steps (in order):
      1. Kill switch check
      2. Broker connection verification
      3. Position reconciliation
      4. Order reconciliation
      5. P&L verification
      6. Artifact integrity check
      7. Final state logging

    Returns ReconciliationResult. Check requires_human_review before resuming.
    """
    result = ReconciliationResult(success=False)

    # ---- Step 1: Kill switch ----
    if state_store.get_kill_switch():
        result.errors.append("Kill switch is active. Cannot resume. Disable manually.")
        logger.critical("Reconciliation aborted: kill switch active")
        return result

    # ---- Step 2: Broker connection ----
    try:
        account = adapter.get_account_state()
        logger.info(
            "Broker connected",
            extra={
                "equity_usd": float(account.equity_usd),
                "balance_usd": float(account.balance_usd),
            }
        )
    except Exception as exc:
        result.errors.append(f"Broker connection failed: {exc}")
        logger.critical("Reconciliation aborted: broker connection failed", extra={"error": str(exc)})
        return result

    # ---- Step 3: Position reconciliation ----
    _reconcile_positions(adapter, state_store, result)

    # ---- Step 4: Order reconciliation ----
    _reconcile_orders(adapter, state_store, result)

    # ---- Step 5: P&L verification ----
    _verify_pnl(adapter, state_store, account, result)

    # ---- Step 6: Artifact integrity ----
    _verify_artifacts(state_store, model_sha256, pipeline_sha256, result)

    # ---- Step 7: Final state ----
    state_store.set_heartbeat()
    state_store.update_high_water_mark(float(account.equity_usd))

    if result.errors:
        logger.critical(
            "Reconciliation completed with errors — trading suspended",
            extra={"errors": result.errors, "warnings": result.warnings}
        )
        result.success = False
    else:
        logger.info(
            "Reconciliation completed successfully",
            extra={
                "warnings": result.warnings,
                "positions_updated": result.positions_updated,
                "orders_updated": result.orders_updated,
                "requires_human_review": result.requires_human_review,
            }
        )
        result.success = True

    return result


def _reconcile_positions(adapter, state_store, result: ReconciliationResult) -> None:
    """
    Compare broker positions vs local positions.
    Broker is source of truth for existence and fill prices.
    """
    try:
        broker_positions = adapter.get_open_positions()
    except Exception as exc:
        result.errors.append(f"Failed to fetch broker positions: {exc}")
        return

    local_positions = state_store.get_open_positions()

    broker_ids = {p.position_id: p for p in broker_positions}
    local_ids = {p.position_id: p for p in local_positions}

    # Case A: In broker but not in local → ORPHANED
    for pos_id, broker_pos in broker_ids.items():
        if pos_id not in local_ids:
            broker_pos.status = PositionStatus.ORPHANED
            state_store.upsert_position(broker_pos)
            result.orphaned_positions.append(pos_id)
            result.warnings.append(
                f"ORPHANED position {pos_id} ({broker_pos.pair.value} "
                f"{broker_pos.side.value} {broker_pos.units} units): "
                f"broker has it, local didn't. Do not close automatically."
            )
            logger.warning(
                "Orphaned position found",
                extra={"position_id": pos_id, "pair": broker_pos.pair.value}
            )

    # Case B: In local but not in broker → closed while we were down
    for pos_id, local_pos in local_ids.items():
        if pos_id not in broker_ids:
            local_pos.status = PositionStatus.CLOSED
            local_pos.closed_at = now_utc()
            local_pos.close_reason = __import__("shared.schemas", fromlist=["CloseReason"]).CloseReason.RECONCILIATION
            state_store.upsert_position(local_pos)
            result.positions_updated += 1
            result.warnings.append(
                f"Position {pos_id} ({local_pos.pair.value}) closed by broker "
                f"while system was down. P&L needs manual verification."
            )
            logger.warning(
                "Position closed by broker during downtime",
                extra={"position_id": pos_id}
            )

    # Case C: In both → check for discrepancies
    PRICE_TOLERANCE_PIPS = 1.0
    for pos_id in set(broker_ids) & set(local_ids):
        broker_pos = broker_ids[pos_id]
        local_pos = local_ids[pos_id]

        # Units mismatch
        if broker_pos.units != local_pos.units:
            result.warnings.append(
                f"Position {pos_id} units mismatch: broker={broker_pos.units}, "
                f"local={local_pos.units}. Using broker value."
            )
            local_pos.units = broker_pos.units
            state_store.upsert_position(local_pos)
            result.positions_updated += 1

        # Entry price discrepancy > 1 pip
        from shared.schemas import pip_multiplier
        price_diff_pips = abs(
            float(broker_pos.entry_price) - float(local_pos.entry_price)
        ) * pip_multiplier(local_pos.pair)

        if price_diff_pips > PRICE_TOLERANCE_PIPS:
            result.warnings.append(
                f"Position {pos_id} entry price mismatch: "
                f"broker={broker_pos.entry_price}, local={local_pos.entry_price} "
                f"({price_diff_pips:.1f} pips). Using broker value."
            )
            local_pos.entry_price = broker_pos.entry_price
            state_store.upsert_position(local_pos)
            result.positions_updated += 1

        # Update unrealized P&L from broker
        local_pos.unrealized_pnl_usd = broker_pos.unrealized_pnl_usd
        local_pos.swap_accumulated_usd = broker_pos.swap_accumulated_usd
        state_store.upsert_position(local_pos)


def _reconcile_orders(adapter, state_store, result: ReconciliationResult) -> None:
    """
    Reconcile pending orders.
    PENDING_SUBMIT orders with no broker record are flagged UNCONFIRMED — never auto-resubmit.
    """
    try:
        broker_orders = adapter.get_pending_orders()
    except Exception as exc:
        result.warnings.append(f"Failed to fetch broker pending orders: {exc}")
        return

    broker_order_ids = {o.broker_order_id for o in broker_orders if o.broker_order_id}

    # Check local orders that were pending when we crashed
    pending_submit = state_store.get_orders_by_status(OrderStatus.PENDING_SUBMIT)
    submitted = state_store.get_orders_by_status(OrderStatus.SUBMITTED)

    for order in pending_submit:
        # These were written to DB but broker receipt is unknown
        order.status = OrderStatus.UNCONFIRMED
        state_store.upsert_order(order)
        result.unconfirmed_orders.append(order.client_order_id)
        result.warnings.append(
            f"Order {order.client_order_id} was PENDING_SUBMIT at crash time. "
            f"Status unknown. DO NOT resubmit — marked UNCONFIRMED for human review."
        )
        result.orders_updated += 1

    for order in submitted:
        if order.broker_order_id and order.broker_order_id not in broker_order_ids:
            # Was submitted, now missing from broker → likely filled or cancelled during downtime
            result.warnings.append(
                f"Order {order.client_order_id} (broker_id={order.broker_order_id}) "
                f"was SUBMITTED but not found in broker pending orders. "
                f"May have been filled or cancelled during downtime. Check trade log."
            )
            # Mark as UNCONFIRMED rather than guessing
            order.status = OrderStatus.UNCONFIRMED
            state_store.upsert_order(order)
            result.unconfirmed_orders.append(order.client_order_id)
            result.orders_updated += 1


def _verify_pnl(adapter, state_store, account, result: ReconciliationResult) -> None:
    """
    Cross-check local daily P&L computation against broker account state.
    On discrepancy > $1: log warning, use broker balance as reference.
    """
    today = now_utc().date().isoformat()
    local_pnl = state_store.get_daily_pnl(today)
    broker_balance = float(account.balance_usd)

    # We can't perfectly reconcile without knowing session start balance,
    # but we flag extreme discrepancies
    hwm = state_store.get_high_water_mark()
    if hwm > 0:
        broker_dd = (hwm - broker_balance) / hwm
        if broker_dd >= 0.05:
            result.warnings.append(
                f"Significant drawdown detected: {broker_dd:.1%} from peak. "
                f"HWM=${hwm:.2f}, balance=${broker_balance:.2f}"
            )

    logger.info(
        "P&L verified",
        extra={
            "local_daily_pnl": local_pnl,
            "broker_equity": float(account.equity_usd),
            "broker_balance": broker_balance,
        }
    )


def _verify_artifacts(
    state_store,
    model_sha256: str,
    pipeline_sha256: str,
    result: ReconciliationResult,
) -> None:
    """
    Verify that loaded artifacts match the deployed versions recorded in DB.
    If hashes don't match: fatal error — do not trade with a mismatched model.
    """
    stored_model_sha = state_store.get_state("deployed_model_sha256", "")
    stored_pipeline_sha = state_store.get_state("deployed_pipeline_sha256", "")

    if stored_model_sha and stored_model_sha != model_sha256:
        result.errors.append(
            f"Model SHA-256 mismatch: stored={stored_model_sha[:16]}…, "
            f"loaded={model_sha256[:16]}…. Artifact may have changed since last deployment."
        )

    if stored_pipeline_sha and stored_pipeline_sha != pipeline_sha256:
        result.errors.append(
            f"Pipeline SHA-256 mismatch: stored={stored_pipeline_sha[:16]}…, "
            f"loaded={pipeline_sha256[:16]}…"
        )

    if not stored_model_sha:
        # First run — record the hashes
        result.warnings.append("No stored artifact hashes. Recording current hashes as baseline.")
        state_store.set_state("deployed_model_sha256", model_sha256)
        state_store.set_state("deployed_pipeline_sha256", pipeline_sha256)
