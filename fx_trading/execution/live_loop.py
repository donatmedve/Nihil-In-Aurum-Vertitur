# execution/live_loop.py
# Live execution loop.
# This is the top-level process that runs 24/5 on your VPS.
#
# RULES:
#   - Never retrain inside this loop
#   - Never load a new model version without restart + reconciliation
#   - Check kill switch on every bar
#   - Check heartbeat on every bar
#   - All signals go through RiskEngine before reaching BrokerAdapter
#   - NaN in features = ABSTAIN, always
#   - Take-profit is set server-side at the broker — we do not manage it locally

import logging
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional

from shared.schemas import (
    Order, OrderType, Position, Pair, Side, Signal
)
from data.time_utils import now_utc, is_market_open, format_utc
from research.model_training import predict_proba, compute_signal
from execution.reconciliation import run_reconciliation

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_SECONDS = 30
STALE_TICK_THRESHOLD_SECONDS = 30
BAR_COMPLETION_TIMEOUT_SECONDS = 330   # 5m bar + 30s grace

# MT5 timeframe constant — avoids importing MT5 at module level
# If you change timeframe, update this AND timeframe_sec in __init__
MT5_TIMEFRAME_M5 = 16388   # mt5.TIMEFRAME_M5 numeric value


class LiveExecutionLoop:
    """
    Main loop. Instantiate once at startup. Call run() to start.

    Dependencies are injected — never instantiated inside this class.
    This makes testing and swapping brokers straightforward.
    """

    def __init__(
        self,
        adapter,                    # BrokerAdapter
        submitter,                  # IdempotentOrderSubmitter
        pipeline,                   # FeaturePipeline
        model,                      # trained model object
        model_sha256: str,
        pipeline_sha256: str,
        model_version: str,
        risk_engine,                # RiskEngine
        state_store,                # StateStore
        pairs: list[Pair],
        broker_tz_str: str,         # e.g. "Etc/GMT-2" — your broker's server timezone
        timeframe_sec: int = 300,   # 5m bars
        lookback_bars: int = 110,   # pipeline.config.rolling_window_bars + buffer
    ):
        self._adapter        = adapter
        self._submitter      = submitter
        self._pipeline       = pipeline
        self._model          = model
        self._model_sha256   = model_sha256
        self._pipeline_sha256 = pipeline_sha256
        self._model_version  = model_version
        self._risk           = risk_engine
        self._state          = state_store
        self._pairs          = pairs
        self._broker_tz_str  = broker_tz_str
        self._timeframe_sec  = timeframe_sec
        self._lookback_bars  = lookback_bars

        self._last_heartbeat_check = now_utc()
        self._last_signal_bar: dict[Pair, datetime] = {}   # tracks last bar we acted on

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, model_sha256: str, pipeline_sha256: str) -> None:
        """
        Entry point. Runs reconciliation, then enters the bar loop.
        Exits on kill switch or unrecoverable error.
        """
        logger.info(
            "Starting live execution loop",
            extra={"pairs": [p.value for p in self._pairs], "model": self._model_version}
        )

        # ---- Startup reconciliation (mandatory) ----
        recon_result = run_reconciliation(
            self._adapter, self._state, model_sha256, pipeline_sha256
        )

        if not recon_result.success:
            logger.critical(
                "Reconciliation failed — cannot start trading",
                extra={"errors": recon_result.errors}
            )
            return

        if recon_result.requires_human_review:
            logger.critical(
                "Reconciliation requires human review before trading can resume",
                extra={
                    "orphaned_positions": recon_result.orphaned_positions,
                    "unconfirmed_orders": recon_result.unconfirmed_orders,
                    "warnings": recon_result.warnings,
                }
            )
            return

        logger.info("Reconciliation passed. Entering bar loop.")

        # ---- Main loop ----
        while True:
            try:
                self._iteration()
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received. Shutting down cleanly.")
                break
            except SystemExit:
                logger.critical("Kill switch triggered SystemExit. Halting.")
                break
            except Exception as exc:
                logger.critical(
                    "Unhandled exception in execution loop",
                    extra={"error": str(exc), "type": type(exc).__name__}
                )
                if self._state.get_kill_switch():
                    logger.critical("Kill switch active after exception. Exiting.")
                    break
                time.sleep(5)

    # ------------------------------------------------------------------
    # Per-iteration logic
    # ------------------------------------------------------------------

    def _iteration(self) -> None:
        """
        Single iteration of the execution loop.
        Called on a 1-second polling interval.
        """
        now = now_utc()

        # ---- Heartbeat ----
        if (now - self._last_heartbeat_check).total_seconds() >= HEARTBEAT_INTERVAL_SECONDS:
            self._run_heartbeat(now)
            self._last_heartbeat_check = now

        # ---- Kill switch check ----
        if self._state.get_kill_switch():
            logger.critical("Kill switch active. Loop halted.")
            raise SystemExit("Kill switch")

        # ---- Market closed → sleep ----
        if not is_market_open(now):
            time.sleep(60)
            return

        # ---- Process each pair ----
        for pair in self._pairs:
            self._process_pair(pair, now)

        time.sleep(1)   # polling interval: 1 second

    # ------------------------------------------------------------------
    # Per-pair signal + order logic
    # ------------------------------------------------------------------

    def _process_pair(self, pair: Pair, now: datetime) -> None:
        """
        Check if a new complete bar is available for this pair.
        If yes: compute features, get signal, evaluate risk, potentially trade.
        """
        # ---- Fetch bars ----
        try:
            bars = self._fetch_recent_bars(pair, n=self._lookback_bars + 2)
        except Exception as exc:
            logger.error(
                "Failed to fetch bars",
                extra={"pair": pair.value, "error": str(exc)}
            )
            return

        if not bars or len(bars) < self._lookback_bars:
            logger.warning(
                "Insufficient bar history",
                extra={"pair": pair.value, "available": len(bars) if bars else 0}
            )
            return

        # ---- Only use complete bars ----
        complete_bars = [b for b in bars if b.is_complete]
        if not complete_bars:
            return

        last_bar = complete_bars[-1]

        # ---- Deduplicate: have we already processed this bar? ----
        prev_bar_time = self._last_signal_bar.get(pair)
        if prev_bar_time is not None and last_bar.utc_open <= prev_bar_time:
            return   # same bar, nothing new
        self._last_signal_bar[pair] = last_bar.utc_open

        logger.debug(
            "New complete bar",
            extra={"pair": pair.value, "bar_open": format_utc(last_bar.utc_open)}
        )

        # ---- Convert to DataFrame ----
        from data.aggregation import bars_to_dataframe
        df = bars_to_dataframe(complete_bars)

        if df.isnull().all().any():
            logger.error("All-NaN column in bar data — skipping", extra={"pair": pair.value})
            return

        # ---- Feature computation ----
        try:
            features = self._pipeline.transform(df)
        except Exception as exc:
            logger.error(
                "Feature pipeline error",
                extra={"pair": pair.value, "error": str(exc)}
            )
            return

        if features.empty:
            return

        feature_row = features.iloc[[-1]]

        # Guard: NaN in features → ABSTAIN
        if feature_row.isnull().any().any():
            nan_cols = list(feature_row.columns[feature_row.isnull().any()])
            logger.warning(
                "NaN in feature row — abstaining",
                extra={"pair": pair.value, "nan_cols": nan_cols}
            )
            return

        # ---- Model inference ----
        try:
            proba = predict_proba(self._model, feature_row)
        except Exception as exc:
            logger.error(
                "Model inference error",
                extra={"pair": pair.value, "error": str(exc)}
            )
            return

        direction, confidence = compute_signal(proba, min_confidence=0.55, min_margin=0.10)

        if direction == 0:
            logger.info(
                "ABSTAIN",
                extra={
                    "pair": pair.value,
                    "bar_open": format_utc(last_bar.utc_open),
                    "p_long": round(proba[2], 3),
                    "p_short": round(proba[0], 3),
                    "p_abstain": round(proba[1], 3),
                }
            )
            return

        # ---- Get current prices ----
        try:
            bid, ask = self._adapter.get_price(pair)
        except Exception as exc:
            logger.error(
                "Failed to get price",
                extra={"pair": pair.value, "error": str(exc)}
            )
            return

        # ---- Get account state ----
        try:
            account = self._adapter.get_account_state()
        except Exception as exc:
            logger.error("Failed to get account state", extra={"error": str(exc)})
            return

        # ---- ATR from features (pre-shifted, so no look-ahead) ----
        atr_col = [c for c in feature_row.columns if "atr" in c.lower()]
        if atr_col:
            atr_pips = float(feature_row[atr_col[0]].iloc[0])
        else:
            # Fallback: approximate from last bar range
            atr_pips = float((last_bar.high_bid - last_bar.low_bid) / Decimal("0.0001"))

        # ---- Risk evaluation ----
        open_positions = self._state.get_open_positions()
        daily_pnl = self._state.get_daily_pnl(now_utc().strftime("%Y-%m-%d"))

        try:
            decision = self._risk.evaluate(
                signal=direction,
                confidence=confidence,
                pair=pair,
                current_bid=bid,
                current_ask=ask,
                atr_pips=atr_pips,
                equity_usd=float(account.equity_usd),
                open_positions=open_positions,
                daily_pnl_usd=daily_pnl,
            )
        except Exception as exc:
            logger.error("Risk engine error", extra={"error": str(exc)})
            return

        if not decision.trade_permitted:
            logger.info(
                "Trade denied by risk engine",
                extra={"pair": pair.value, "reason": decision.reason}
            )
            return

        # ---- Build and submit order ----
        side = Side.BUY if direction == 1 else Side.SELL

        order = Order(
            pair=pair,
            side=side,
            order_type=OrderType.MARKET,
            units=decision.position_size_units,
            stop_loss_price=decision.stop_price,
            take_profit_price=decision.take_profit_price,  # set server-side at broker
            model_version=self._model_version,
            signal_confidence=confidence,
        )

        logger.info(
            "Submitting order",
            extra={
                "pair": pair.value,
                "side": side.value,
                "units": decision.position_size_units,
                "stop_price": float(decision.stop_price),
                "take_profit_price": float(decision.take_profit_price) if decision.take_profit_price else None,
                "confidence": round(confidence, 3),
                "bar_open": format_utc(last_bar.utc_open),
            }
        )

        try:
            result = self._submitter.submit(order)
        except Exception as exc:
            logger.critical(
                "Order submission failed",
                extra={"pair": pair.value, "error": str(exc), "client_order_id": order.client_order_id}
            )
            return

        if result.status.value in ("FILLED", "PARTIALLY_FILLED"):
            logger.info(
                "Order filled",
                extra={
                    "pair": pair.value,
                    "broker_order_id": result.broker_order_id,
                    "filled_price": float(result.filled_price) if result.filled_price else None,
                    "units": order.units,
                }
            )
        elif result.status.value == "REJECTED":
            logger.error(
                "Order rejected by broker",
                extra={"pair": pair.value, "reason": result.rejection_reason}
            )
        elif result.status.value == "DEAD_LETTERED":
            logger.critical(
                "Order dead-lettered after retries",
                extra={"pair": pair.value, "client_order_id": order.client_order_id}
            )

    # ------------------------------------------------------------------
    # Bar fetching — concrete MT5 implementation
    # ------------------------------------------------------------------

    def _fetch_recent_bars(self, pair: Pair, n: int) -> list:
        """
        Fetch recent bars from MT5 using data/ingestion.py.
        Returns list of OHLCVBar sorted ascending by utc_open.
        """
        from data.ingestion import fetch_recent_bars_mt5

        return fetch_recent_bars_mt5(
            pair=pair,
            n=n,
            timeframe_mt5=MT5_TIMEFRAME_M5,
            timeframe_sec=self._timeframe_sec,
            broker_tz_str=self._broker_tz_str,
        )

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def _run_heartbeat(self, now: datetime) -> None:
        """
        Runs every HEARTBEAT_INTERVAL_SECONDS.
        Triggers kill switch if all checks fail.
        """
        checks_passed = 0
        total_checks = 4

        # Check 1: Broker connection
        if self._adapter.is_connected():
            checks_passed += 1
        else:
            logger.error("Heartbeat: broker not connected")

        # Check 2: Account state fetchable and fresh
        try:
            account = self._adapter.get_account_state()
            age = (now - account.as_of_utc).total_seconds()
            if age < 60:
                checks_passed += 1
            else:
                logger.warning("Heartbeat: account state stale", extra={"age_s": age})
        except Exception as exc:
            logger.error("Heartbeat: account state fetch failed", extra={"error": str(exc)})

        # Check 3: Local DB writable
        try:
            self._state.set_heartbeat()
            checks_passed += 1
        except Exception as exc:
            logger.critical("Heartbeat: DB write failed", extra={"error": str(exc)})

        # Check 4: Position count consistent between broker and local DB
        try:
            broker_positions = self._adapter.get_open_positions()
            local_positions  = self._state.get_open_positions()
            if len(broker_positions) == len(local_positions):
                checks_passed += 1
            else:
                logger.warning(
                    "Heartbeat: position count mismatch",
                    extra={"broker": len(broker_positions), "local": len(local_positions)}
                )
        except Exception as exc:
            logger.error("Heartbeat: position check failed", extra={"error": str(exc)})

        logger.info(
            "heartbeat",
            extra={
                "checks_passed": checks_passed,
                "total_checks": total_checks,
                "open_positions": len(self._state.get_open_positions()),
                "model_version": self._model_version,
            }
        )

        if checks_passed == 0:
            logger.critical("All heartbeat checks failed — triggering kill switch")
            self._state.set_kill_switch(True, reason="All heartbeat checks failed")
