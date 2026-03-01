# risk/engine.py
# Risk engine. Last line of defense between a bad signal and capital loss.
# Pure logic + circuit breakers. No broker imports. No model imports.
#
# Call evaluate() before EVERY order submission.
# Call update_after_close() after EVERY position close.

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

    # SL buffer: add half-spread to stop to avoid triggering on noise
    sl_spread_buffer_ratio: float = 0.5

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
                    f"waiting {self.params.cooldown_bars - bars_since} more bars"
                )
            else:
                self._state.reset_consecutive_losses()
                logger.info("Consecutive loss cooldown expired, resuming")

        # ---- Market quality filter 1: ATR ----
        if atr_pips < self.params.min_atr_pips:
            return deny(f"ATR too small: {atr_pips:.1f} pips < {self.params.min_atr_pips}")

        # ---- Market quality filter 2: Spread/ATR ratio ----
        spread_pips = float(current_ask - current_bid) * pip_multiplier(pair)
        if atr_pips > 0 and (spread_pips / atr_pips) > self.params.max_spread_to_atr_ratio:
            return deny(
                f"Spread/ATR ratio too high: {spread_pips:.1f}/{atr_pips:.1f} = "
                f"{spread_pips/atr_pips:.2f} > {self.params.max_spread_to_atr_ratio}"
            )

        # ---- Correlated exposure check ----
        if pair in (Pair.EURUSD, Pair.GBPUSD):
            exposure = self._compute_usd_exposure(open_positions, [Pair.EURUSD, Pair.GBPUSD])
            if exposure >= self.params.max_correlated_usd_exposure:
                return deny(
                    f"Correlated USD exposure cap: ${exposure:,.0f} >= "
                    f"${self.params.max_correlated_usd_exposure:,.0f}"
                )

        # ---- Direction conflict check ----
        # Don't open same pair in same direction if already open
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

        # Stop distance = 1× ATR + half-spread buffer (keeps stop outside spread)
        sl_atr_mult = 1.0
        stop_distance_pips = (atr_pips * sl_atr_mult) + (spread_pips * self.params.sl_spread_buffer_ratio)

        # Pip value in USD per unit
        pip_val_per_unit = _pip_value_per_unit_usd(pair, current_bid)
        if pip_val_per_unit <= 0 or stop_distance_pips <= 0:
            return deny("Invalid pip value or stop distance calculation")

        # Units = risk_usd / (stop_pips × pip_value_per_unit)
        units = int(risk_usd_adjusted / (stop_distance_pips * pip_val_per_unit))
        units = max(self.params.min_units, min(self.params.max_units, units))

        # Stop price (placed outside spread, on the correct side)
        pip_sz = float(pip_size(pair))
        stop_distance_price = stop_distance_pips * pip_sz

        if signal == 1:   # LONG: entry at ask, stop below bid
            entry_price = current_ask
            stop_price = entry_price - Decimal(str(round(stop_distance_price, 5)))
        else:             # SHORT: entry at bid, stop above ask
            entry_price = current_bid
            stop_price = entry_price + Decimal(str(round(stop_distance_price, 5)))

        max_loss_usd = units * stop_distance_pips * pip_val_per_unit

        reason = (
            f"Approved: {units:,} units, SL {stop_distance_pips:.1f} pips, "
            f"max_loss ${max_loss_usd:.2f}, confidence {confidence:.1%}"
        )

        logger.info(
            "Trade approved",
            extra={
                "pair": pair.value,
                "signal": signal,
                "units": units,
                "stop_distance_pips": stop_distance_pips,
                "stop_price": float(stop_price),
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
        """
        dd = self._state.get_drawdown_from_peak(equity_usd)
        if dd >= self.params.max_daily_drawdown_pct:
            logger.critical(
                "Drawdown circuit breaker triggered",
                extra={
                    "drawdown_pct": f"{dd:.2%}",
                    "threshold_pct": f"{self.params.max_daily_drawdown_pct:.2%}",
                    "equity_usd": equity_usd,
                    "high_water_mark": self._state.get_high_water_mark(),
                }
            )
            self._state.set_kill_switch(True, reason=f"Drawdown {dd:.2%} exceeded threshold")
            return True
        return False

    def _compute_usd_exposure(
        self, positions: list[Position], pairs: list[Pair]
    ) -> float:
        """Compute total USD notional exposure for a set of pairs."""
        total = 0.0
        for pos in positions:
            if pos.pair in pairs:
                pip_val = _pip_value_per_unit_usd(pos.pair, pos.entry_price)
                # Notional = units × price (for USD-quoted pairs)
                notional = pos.units * float(pos.entry_price)
                total += notional
        return total


def _pip_value_per_unit_usd(pair: Pair, price: Decimal) -> float:
    """
    USD value of a 1-pip move for 1 unit of the base currency.

    EUR/USD: 1 unit = 1 EUR. 1 pip = 0.0001 USD. pip_value = $0.0001
    GBP/USD: same structure. pip_value = $0.0001
    USD/JPY: 1 unit = 1 USD. 1 pip = 0.01 JPY. pip_value = 0.01 / price_USDJPY USD
    """
    price_f = float(price)
    if pair == Pair.USDJPY:
        return 0.01 / price_f if price_f > 0 else 0.0
    return 0.0001   # EUR/USD, GBP/USD
