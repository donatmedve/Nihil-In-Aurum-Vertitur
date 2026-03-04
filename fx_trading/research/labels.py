# research/labels.py
# Triple barrier label construction.
# Labels are constructed OFFLINE in the research environment.
# Never call this in the live execution loop.
#
# Barrier logic:
#   For each signal bar i, simulate both a LONG and SHORT trade from bar i+1 open.
#   The first barrier hit (TP, SL, or time) determines the label.
#   If only LONG hits TP → label = LONG (1)
#   If only SHORT hits TP → label = SHORT (-1)
#   Otherwise → ABSTAIN (0)

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from shared.schemas import Signal

logger = logging.getLogger(__name__)


@dataclass
class LabelConfig:
    atr_period: int = 14
    tp_atr_multiplier: float = 1.5      # TP distance = ATR × this
    sl_atr_multiplier: float = 1.0      # SL distance = ATR × this
    max_holding_bars: int = 12          # time barrier: max bars to hold
    min_atr_pips: float = 3.0           # skip label if market too quiet
    max_spread_pips: float = 5.0        # skip label if spread too wide


def construct_labels(
    bars: pd.DataFrame,
    config: LabelConfig,
    pair_is_jpy: bool = False,
) -> pd.Series:
    """
    Construct triple barrier labels for a bar DataFrame.

    bars: DataFrame from bars_to_dataframe() — must include bid + ask OHLCV.
          Only pass complete bars (is_complete=True, source != 'gap_filled').
          Must be sorted ascending by utc_open.

    Returns a pd.Series aligned with bars.index.
    Values: 1 (LONG), -1 (SHORT), 0 (ABSTAIN).
    NaN = excluded from training (insufficient data, gap, or label undefined).

    EXCLUSION RULES:
      1. First atr_period bars: ATR undefined
      2. Last max_holding_bars bars: no room to evaluate barriers
      3. source == 'gap_filled': no real price action
      4. spread > max_spread_pips: fill assumptions invalid
      5. ATR < min_atr_pips: market too quiet, barriers too tight
      6. Any bar in the look-forward window has source == 'gap_filled'
    """
    try:
        import pandas_ta as ta
    except ImportError:
        raise ImportError("pandas_ta required: pip install pandas-ta")

    pip_mult = 100.0 if pair_is_jpy else 10000.0
    pip_size = 0.01 if pair_is_jpy else 0.0001

    # Compute ATR on bid prices
    atr_series = ta.atr(
        bars["high_bid"], bars["low_bid"], bars["close_bid"],
        length=config.atr_period
    )

    labels = pd.Series(np.nan, index=bars.index, dtype=float)
    n = len(bars)

    excluded_gap = 0
    excluded_spread = 0
    excluded_atr = 0
    excluded_fwd_gap = 0
    labeled_long = 0
    labeled_short = 0
    labeled_abstain = 0

    for i in range(config.atr_period, n - config.max_holding_bars - 1):
        # ---- Exclusion filters ----
        if bars["source"].iloc[i] == "gap_filled":
            excluded_gap += 1
            continue

        current_atr = atr_series.iloc[i]
        if pd.isna(current_atr):
            continue

        atr_pips = current_atr * pip_mult
        if atr_pips < config.min_atr_pips:
            excluded_atr += 1
            continue

        spread_pips = (bars["close_ask"].iloc[i] - bars["close_bid"].iloc[i]) * pip_mult
        if spread_pips > config.max_spread_pips:
            excluded_spread += 1
            continue

        # Check if any look-forward bar is gap-filled (path is unreliable)
        fwd_slice = bars["source"].iloc[i + 1: i + 1 + config.max_holding_bars]
        if (fwd_slice == "gap_filled").any():
            excluded_fwd_gap += 1
            continue

        # ---- Simulate LONG from bar i+1 open ASK ----
        long_entry = bars["open_ask"].iloc[i + 1]
        long_tp = long_entry + config.tp_atr_multiplier * current_atr
        long_sl = long_entry - config.sl_atr_multiplier * current_atr

        # ---- Simulate SHORT from bar i+1 open BID ----
        short_entry = bars["open_bid"].iloc[i + 1]
        short_tp = short_entry - config.tp_atr_multiplier * current_atr
        short_sl = short_entry + config.sl_atr_multiplier * current_atr

        long_outcome = _check_barriers(
            bars, i + 1, config.max_holding_bars,
            long_tp, long_sl, direction="long"
        )
        short_outcome = _check_barriers(
            bars, i + 1, config.max_holding_bars,
            short_tp, short_sl, direction="short"
        )

        # Label: only assign directional label when one side wins clearly
        if long_outcome == 1 and short_outcome != 1:
            labels.iloc[i] = 1          # LONG
            labeled_long += 1
        elif short_outcome == 1 and long_outcome != 1:
            labels.iloc[i] = -1         # SHORT
            labeled_short += 1
        else:
            labels.iloc[i] = 0          # ABSTAIN
            labeled_abstain += 1

    # Last max_holding_bars rows: labels undefined
    labels.iloc[-(config.max_holding_bars + 1):] = np.nan

    total_labeled = labeled_long + labeled_short + labeled_abstain
    logger.info(
        "Labels constructed",
        extra={
            "total_bars": n,
            "total_labeled": total_labeled,
            "long": labeled_long,
            "short": labeled_short,
            "abstain": labeled_abstain,
            "excluded_gap": excluded_gap,
            "excluded_spread": excluded_spread,
            "excluded_atr": excluded_atr,
            "excluded_fwd_gap": excluded_fwd_gap,
            "long_pct": f"{labeled_long/max(total_labeled,1):.1%}",
            "short_pct": f"{labeled_short/max(total_labeled,1):.1%}",
        }
    )

    return labels


def _check_barriers(
    bars: pd.DataFrame,
    start_idx: int,
    max_bars: int,
    tp_price: float,
    sl_price: float,
    direction: str,
) -> int:
    """
    Simulate price path through subsequent bars.

    Returns:
       1 → TP hit first
      -1 → SL hit first
       0 → time barrier (neither hit within max_bars)

    For LONG:
      SL is below entry → check low_bid <= sl_price (worst case for longs)
      TP is above entry → check high_bid >= tp_price
      SL is checked first within each bar (conservative assumption)

    For SHORT:
      SL is above entry → check high_ask >= sl_price
      TP is below entry → check low_ask <= tp_price
      SL is checked first within each bar
    """
    end_idx = min(start_idx + max_bars, len(bars))

    for j in range(start_idx, end_idx):
        bar = bars.iloc[j]

        if direction == "long":
            # Check SL first (conservative — real SL may trigger before TP on same bar)
            if float(bar["low_bid"]) <= sl_price:
                return -1
            if float(bar["high_bid"]) >= tp_price:
                return 1
        else:  # short
            if float(bar["high_ask"]) >= sl_price:
                return -1
            if float(bar["low_ask"]) <= tp_price:
                return 1

    return 0  # time barrier


def get_label_distribution(labels: pd.Series) -> dict:
    """Summary statistics for label quality review. Call before training."""
    valid = labels.dropna()
    total = len(valid)
    if total == 0:
        return {"error": "No valid labels"}

    counts = valid.value_counts()
    return {
        "total_labeled": total,
        "total_nan": labels.isna().sum(),
        "long": int(counts.get(1.0, 0)),
        "short": int(counts.get(-1.0, 0)),
        "abstain": int(counts.get(0.0, 0)),
        "long_pct": f"{counts.get(1.0, 0) / total:.1%}",
        "short_pct": f"{counts.get(-1.0, 0) / total:.1%}",
        "abstain_pct": f"{counts.get(0.0, 0) / total:.1%}",
        "is_balanced": abs(counts.get(1.0, 0) - counts.get(-1.0, 0)) / total < 0.10,
    }
