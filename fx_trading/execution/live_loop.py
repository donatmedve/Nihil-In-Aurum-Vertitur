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


class LiveExecutionLoop:
    """
    Main loop. Instantiate once at startup. Call run() to start.

    Dependencies are injected — never instantiated inside this class.
    This makes testing and swapping brokers straightforward.
    """

    def __init__(
        self,
        adapter,            # BrokerAdapter
        submitter,          # IdempotentOrderSubmitter
        pipeline,           # FeaturePipeline
        model,              # trained model object
        model_sha256: str,
        pipeline_sha256: str,
        model_version: str,
        risk_engine,        # RiskEngine
        state_store,        # StateStore
        pairs: list[Pair],
        timeframe_sec: int = 300,   # 5m bars
        lookback_bars: int = 110,   # pipeline.config.rolling_window_bars + buffer
    ):
        self._adapter = adapter
        self._submitter = submitter
        self._pipeline = pipeline
        self._model = model
        self._model_sha256 = model_sha256
        self._pipeline_sha256 = pipeline_sha256
        self._model_version = model_version
        self._risk = risk_engine
        self._state = state_store
        self._pairs = pairs
        self._timeframe_sec = timeframe_sec
        self._lookback_bars = lookback_bars
        self._bar_buffer: dict[Pair, list] = {p: [] for p in pairs}
        self._last_heartbeat_check = now_utc()
        self._last_tick_time: dict[Pair, datetime] = {}

    def run(self, model_sha256: str, pipeline_sha256: str) -> None:
        """
        Entry point. Runs reconciliation, then enters the bar loop.
        Exits on kill switch or unrecoverable error.
        """
        logger.info("Starting live execution loop", extra={"pairs": [p.value for p in self._pairs]})

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
            except Exception as exc:
                logger.critical(
                    "Unhandled exception in execution loop",
                    extra={"error": str(exc), "type": type(exc).__name__}
                )
                # Don't exit on transient errors — log and continue
                # but check if kill switch was set by circuit breaker
                if self._state.get_kill_switch():
                    logger.critical("Kill switch active after exception. Exiting.")
                    break
                time.sleep(5)

    def _iteration(self) -> None:
        """
        Single iteration of the execution loop.
        Called on each tick or polling interval.
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

        # ---- Poll for new complete bars ----
        for pair in self._pairs:
            self._process_pair(pair, now)

        time.sleep(1)   # polling interval: 1 second

    def _process_pair(self, pair: Pair, now: datetime) -> None:
        """
        Check if a new complete bar is available for this pair.
        If yes: compute features, get signal, evaluate risk, potentially trade.
        """
        if not is_market_open(now):
            return

        # Fetch latest bars from broker (last lookback_bars of complete bars)
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

        # Only process complete bars
        complete_bars = [b for b in bars if b.is_complete]
        if not complete_bars:
            return

        # Convert to DataFrame
        from data.aggregation import bars_to_dataframe
        df = bars_to_dataframe(complete_bars)

        # Validate no all-NaN columns
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

        # Get the last complete row (current signal bar)
        feature_row = features.iloc[[-1]]

        # Guard: NaN in features → ABSTAIN
        if feature_row.isnull().any().any():
            logger.warning(
                "NaN in feature row — abstaining",
                extra={"pair": pair.value, "nan_cols": feature_row.columns[feature_row.isnull().any()].tolist()}
            )
            return

        # ---- Model inference ----
        try:
            proba = predict_proba(self._model, feature_row, "lightgbm")
            signal, confidence = compute_signal(proba[0])
        except Exception as exc:
            logger.error(
                "Model inference error",
                extra={"pair": pair.value, "error": str(exc)}
            )
            return

        # Log signal (even ABSTAIN — important for monitoring)
        self._state.set_last_bar_utc(df.index[-1].to_pydatetime())
        logger.info(
            "signal_generated",
            extra={
                "pair": pair.value,
                "signal": signal,
                "confidence": round(confidence, 4),
                "bar_utc": format_utc(df.index[-1].to_pydatetime()),
                "model_version": self._model_version,
            }
        )

        if signal == 0:
            return

        # ---- Risk evaluation ----
        try:
            bid, ask = self._adapter.get_price(pair)
        except Exception as exc:
            logger.error("Failed to get price", extra={"pair": pair.value, "error": str(exc)})
            return

        try:
            account = self._adapter.get_account_state()
        except Exception as exc:
            logger.error("Failed to get account state", extra={"error": str(exc)})
            return

        # Check drawdown circuit breaker
        if self._risk.check_drawdown_circuit_breaker(float(account.equity_usd)):
            return  # kill switch set inside check_drawdown_circuit_breaker

        open_positions = self._state.get_open_positions()
        today = now_utc().date().isoformat()
        daily_pnl = self._state.get_daily_pnl(today)

        # Get ATR in pips from features
        atr_pct = float(feature_row["atr_pct"].iloc[0]) if "atr_pct" in feature_row.columns else 0
        from shared.schemas import pip_multiplier
        atr_pips = atr_pct * float(ask) * pip_multiplier(pair)

        decision = self._risk.evaluate(
            signal=signal,
            confidence=confidence,
            pair=pair,
            current_bid=bid,
            current_ask=ask,
            atr_pips=atr_pips,
            equity_usd=float(account.equity_usd),
            open_positions=open_positions,
            daily_pnl_usd=daily_pnl,
        )

        if not decision.trade_permitted:
            logger.info(
                "Trade blocked by risk engine",
                extra={"pair": pair.value, "reason": decision.reason}
            )
            return

        # ---- Build and submit order ----
        order = Order(
            pair=pair,
            side=Side.BUY if signal == 1 else Side.SELL,
            order_type=OrderType.MARKET,
            units=decision.position_size_units,
            stop_loss_price=decision.stop_price,
        )

        result = self._submitter.submit(order, signal_confidence=confidence)

        logger.info(
            "order_submitted",
            extra={
                "client_order_id": result.client_order_id,
                "broker_order_id": result.broker_order_id,
                "status": result.status.value,
                "pair": pair.value,
                "units": decision.position_size_units,
                "stop_loss": float(decision.stop_price),
            }
        )

        # Increment bar count for cooldown tracking
        self._state.increment_bar_count()
        self._state.update_high_water_mark(float(account.equity_usd))

    def _run_heartbeat(self, now: datetime) -> None:
        """
        Heartbeat checks. Runs every HEARTBEAT_INTERVAL_SECONDS.
        Triggers kill switch if too many consecutive failures.
        """
        checks_passed = 0
        total_checks = 4

        # Check 1: Broker connection
        if self._adapter.is_connected():
            checks_passed += 1
        else:
            logger.error("Heartbeat: broker not connected")

        # Check 2: Account state fetchable
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

        # Check 4: Position count consistent
        try:
            broker_positions = self._adapter.get_open_positions()
            local_positions = self._state.get_open_positions()
            if len(broker_positions) == len(local_positions):
                checks_passed += 1
            else:
                logger.warning(
                    "Heartbeat: position count mismatch",
                    extra={
                        "broker": len(broker_positions),
                        "local": len(local_positions),
                    }
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

        # If all checks fail repeatedly, trigger kill switch
        if checks_passed == 0:
            logger.critical("All heartbeat checks failed — triggering kill switch")
            self._state.set_kill_switch(True, reason="All heartbeat checks failed")

    def _fetch_recent_bars(self, pair: Pair, n: int) -> list:
        """
        Fetch recent bars from broker. Pair-specific implementation.
        In a real system this calls MT5.copy_rates_from_pos() or OANDA candles endpoint.
        Returns list of OHLCVBar.

        This is a stub — implement broker-specific bar fetching in data/ingestion.py
        and call it from here.
        """
        # Placeholder — replace with actual implementation
        raise NotImplementedError(
            "Implement _fetch_recent_bars using data/ingestion.py. "
            "See data layer specification for MT5/OANDA bar fetching."
        )
