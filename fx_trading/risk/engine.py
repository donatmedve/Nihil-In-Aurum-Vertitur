# risk/engine.py
# Risk engine. Last line of defense between a bad signal and capital loss.
# Pure logic + circuit breakers. No broker imports. No model imports.
#
# Call evaluate() before EVERY order submission.
# Call update_after_close() after EVERY position close.
#
# Changes vs original:
#   - take_profit_price added to RiskDecision output (1.5× ATR, matching label construction)
#   - RiskParameters.tp_atr_multiplier added (default 1.5)

import logging
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, timezone

from shared.schemas import (
    RiskDecision, Position, Pair, Side, pip_size, pip_multiplier
)

logger = logging.getLogger(__name__)


@dataclass
class RiskParameters:
    # Position sizing
    risk_per_trade_pct: float = 0.01        # 1% of equity per trade

    # Circuit breakers
    max_daily_drawdown_pct: float = 0.03    # 3%: halt trading for the day
    max_open_positions: int = 3
    max_consecutive_losses: int = 4
    cooldown_bars: int = 12                 # bars to pause after hitting loss limit

    # Market quality filters
    min_atr_pips: float = 5.0              # skip trades in flat markets
    max_spread_to_atr_ratio: float = 0.25  # skip if spread > 25% of ATR

    # Correlated exposure cap (EUR/USD + GBP/USD both long USD)
    max_correlated_usd_exposure: float = 10_000.0   # in USD notional

    # Stop loss: 1.0× ATR + half-spread buffer
    sl_atr_multiplier: float = 1.0
    sl_spread_buffer_ratio: float = 0.5

    # Take profit: 1.5× ATR (matches label construction — same reward:risk ratio)
    # Increase to 2.0 for a wider target; decrease to 1.0 for faster exits.
    tp_atr_multiplier: float = 1.5

    # Sizing floors
    min_units: int = 1_000      # micro lot minimum
    max_units: int = 100_000    # 1 standard lot maximum


class RiskEngine:
    """
    Stateless evaluation with stateful circuit breaker checks via state_store.

    Usage:
        decision = engine.evaluate(signal, confidence, ...)
        if decision.trade_permitted:
            order = build_order(decision)
            submitter.submit(order)
    """

    def __init__(self, params: RiskParameters, state_store):
        self.params = params
        self._state = state_store

    def evaluate(
        self,
        signal: int,                    # -1, 0, 1
        confidence: float,              # model output confidence [0, 1]
        pair: Pair,
        current_bid: Decimal,
        current_ask: Decimal,
        atr_pips: float,                # ATR in pips (pre-shifted, from last complete bar)
        equity_usd: float,
        open_positions: list[Position],
        daily_pnl_usd: float,
    ) -> RiskDecision:
        """
        Evaluate all risk checks and compute position sizing.
        Returns RiskDecision with trade_permitted=True only if ALL checks pass.
        Always returns a reason string explaining the outcome.
        """

        def deny(reason: str) -> RiskDecision:
            logger.info("Trade denied", extra={"reason": reason, "pair": pair.value})
            return RiskDecision(
                trade_permitted=False,
                reason=reason,
                position_size_units=0,
                stop_distance_pips=0.0,
                stop_price=Decimal("0"),
                take_profit_price=None,
                max_loss_usd=0.0,
            )

        # ---- Circuit breaker 1: Kill switch ----
        if self._state.get_kill_switch():
            return deny("Kill switch is active")

        # ---- Circuit breaker 2: Signal is abstain ----
        if signal == 0:
            return deny("Signal is ABSTAIN")

        # ---- Circuit breaker 3: Daily drawdown ----
        daily_dd_pct = -daily_pnl_usd / equity_usd if equity_usd > 0 else 0.0
        if daily_dd_pct >= self.params.max_daily_drawdown_pct:
            return deny(
                f"Daily drawdown circuit breaker: {daily_dd_pct:.2%} >= "
                f"{self.params.max_daily_drawdown_pct:.2%}"
            )

        # ---- Circuit breaker 4: Max open positions ----
        if len(open_positions) >= self.params.max_open_positions:
            return deny(
                f"Max open positions reached: {len(open_positions)}/{self.params.max_open_positions}"
            )

        # ---- Circuit breaker 5: Consecutive loss cooldown ----
        consec = self._state.get_consecutive_losses()
        if consec >= self.params.max_consecutive_losses:
            bars_since = self._state.get_bars_since_loss_limit()
            if bars_since < self.params.cooldown_bars:
                return deny(
                    f"Consecutive loss cooldown: {consec} losses, "
                    f"{self.params.cooldown_bars - bars_since} bars remaining"
                )
            else:
                self._state.reset_consecutive_losses()

        # ---- Circuit breaker 6: ATR too small ----
        if atr_pips < self.params.min_atr_pips:
            return deny(f"ATR too small: {atr_pips:.1f} pips < {self.params.min_atr_pips:.1f} pips")

        # ---- Circuit breaker 7: Spread/ATR ratio too high ----
        spread_pips = float((current_ask - current_bid) / pip_size(pair))
        spread_atr_ratio = spread_pips / atr_pips if atr_pips > 0 else 999.0
        if spread_atr_ratio > self.params.max_spread_to_atr_ratio:
            return deny(
                f"Spread/ATR too high: {spread_atr_ratio:.2f} > {self.params.max_spread_to_atr_ratio:.2f}"
            )

        # ---- Circuit breaker 8: Correlated USD exposure ----
        if pair in (Pair.EURUSD, Pair.GBPUSD):
            desired_side = Side.BUY if signal == 1 else Side.SELL
            correlated_pairs = [Pair.EURUSD, Pair.GBPUSD]
            correlated_usd_exposure = sum(
                p.units * float(p.entry_price) * (1 if p.side == desired_side else -1)
                for p in open_positions
                if p.pair in correlated_pairs and p.side == desired_side
            )
            if correlated_usd_exposure >= self.params.max_correlated_usd_exposure:
                return deny(
                    f"Correlated USD exposure cap: ${correlated_usd_exposure:,.0f} >= "
                    f"${self.params.max_correlated_usd_exposure:,.0f}"
                )

        # ---- Direction conflict check ----
        same_pair_positions = [p for p in open_positions if p.pair == pair]
        desired_side = Side.BUY if signal == 1 else Side.SELL
        for existing in same_pair_positions:
            if existing.side == desired_side:
                return deny(f"Already have {pair.value} {desired_side.value} position open")

        # ---- Position sizing ----
        risk_usd = equity_usd * self.params.risk_per_trade_pct

        # Scale by confidence (floor at 55% of normal size, cap at 100%)
        confidence_scalar = max(0.55, min(1.0, confidence))
        risk_usd_adjusted = risk_usd * confidence_scalar

        # Stop distance = SL_ATR_MULT × ATR + half-spread buffer
        stop_distance_pips = (
            atr_pips * self.params.sl_atr_multiplier
            + spread_pips * self.params.sl_spread_buffer_ratio
        )

        # Pip value in USD per unit
        pip_val_per_unit = _pip_value_per_unit_usd(pair, current_bid)
        if pip_val_per_unit <= 0 or stop_distance_pips <= 0:
            return deny("Invalid pip value or stop distance calculation")

        # Units = risk_usd / (stop_pips × pip_value_per_unit)
        units = int(risk_usd_adjusted / (stop_distance_pips * pip_val_per_unit))
        units = max(self.params.min_units, min(self.params.max_units, units))

        # ---- Stop price ----
        pip_sz = float(pip_size(pair))
        stop_distance_price = stop_distance_pips * pip_sz

        if signal == 1:   # LONG: entry at ask, stop below bid
            entry_price = current_ask
            stop_price  = entry_price - Decimal(str(round(stop_distance_price, 5)))
        else:             # SHORT: entry at bid, stop above ask
            entry_price = current_bid
            stop_price  = entry_price + Decimal(str(round(stop_distance_price, 5)))

        # ---- Take-profit price ----
        # TP distance = TP_ATR_MULT × ATR (no spread buffer — TP benefits from spread narrowing)
        # This matches the label construction (tp_atr_multiplier=1.5, sl_atr_multiplier=1.0)
        # so the live system's exits mirror what the model was trained to predict.
        tp_distance_pips  = atr_pips * self.params.tp_atr_multiplier
        tp_distance_price = tp_distance_pips * pip_sz

        if signal == 1:   # LONG: TP above entry
            take_profit_price = entry_price + Decimal(str(round(tp_distance_price, 5)))
        else:             # SHORT: TP below entry
            take_profit_price = entry_price - Decimal(str(round(tp_distance_price, 5)))

        # ---- Final decision ----
        max_loss_usd = units * stop_distance_pips * pip_val_per_unit
        rr_ratio = tp_distance_pips / stop_distance_pips

        reason = (
            f"Approved: {units:,} units | "
            f"SL {stop_distance_pips:.1f} pips @ {float(stop_price):.5f} | "
            f"TP {tp_distance_pips:.1f} pips @ {float(take_profit_price):.5f} | "
            f"R:R {rr_ratio:.2f} | "
            f"max_loss ${max_loss_usd:.2f} | "
            f"confidence {confidence:.1%}"
        )

        logger.info(
            "Trade approved",
            extra={
                "pair": pair.value,
                "signal": signal,
                "units": units,
                "stop_distance_pips": stop_distance_pips,
                "stop_price": float(stop_price),
                "tp_distance_pips": tp_distance_pips,
                "take_profit_price": float(take_profit_price),
                "rr_ratio": round(rr_ratio, 2),
                "max_loss_usd": max_loss_usd,
                "confidence": confidence,
                "equity_usd": equity_usd,
            }
        )

        return RiskDecision(
            trade_permitted=True,
            reason=reason,
            position_size_units=units,
            stop_distance_pips=stop_distance_pips,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            max_loss_usd=max_loss_usd,
        )

    def update_after_close(self, realized_pnl_usd: float, current_date: str) -> None:
        """
        Must be called after every position close.
        Updates consecutive loss counter and daily P&L.
        """
        self._state.add_daily_pnl(realized_pnl_usd, current_date)

        if realized_pnl_usd < 0:
            count = self._state.increment_consecutive_losses()
            logger.warning(
                "Losing trade recorded",
                extra={"pnl_usd": realized_pnl_usd, "consecutive_losses": count}
            )
            if count >= self.params.max_consecutive_losses:
                logger.warning(
                    "Consecutive loss limit reached — cooldown activated",
                    extra={"count": count, "cooldown_bars": self.params.cooldown_bars}
                )
        else:
            self._state.reset_consecutive_losses()

    def check_drawdown_circuit_breaker(self, equity_usd: float) -> bool:
        """
        Returns True if the peak-to-trough drawdown exceeds the daily limit.
        Call on every bar during the execution loop.
        If True, the caller must set the kill switch and halt.
        """
        self._state.update_high_water_mark(equity_usd)
        drawdown = self._state.get_drawdown_from_peak(equity_usd)
        if drawdown >= self.params.max_daily_drawdown_pct:
            logger.critical(
                "Drawdown circuit breaker triggered",
                extra={"drawdown": f"{drawdown:.2%}", "equity_usd": equity_usd}
            )
            self._state.set_kill_switch(True, reason=f"Drawdown circuit breaker: {drawdown:.2%}")
            return True
        return False


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _pip_value_per_unit_usd(pair: Pair, current_price: Decimal) -> float:
    """
    USD value of 1 pip for 1 unit of the base currency.

    EUR/USD: 1 pip = 0.0001 USD per unit → $0.0001
    GBP/USD: 1 pip = 0.0001 USD per unit → $0.0001
    USD/JPY: 1 pip = 0.01 JPY per unit → 0.01 / current_price USD per unit

    This is used for position sizing: units = risk_usd / (stop_pips × pip_value).
    """
    if pair == Pair.USDJPY:
        # USD/JPY quote currency is JPY; convert pip value to USD
        if current_price == 0:
            return 0.0
        return float(Decimal("0.01") / current_price)
    else:
        # EUR/USD and GBP/USD quote currency is already USD
        return 0.0001
