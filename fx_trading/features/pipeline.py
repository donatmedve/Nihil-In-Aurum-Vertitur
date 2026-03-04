# features/pipeline.py
# Feature pipeline. Stateless transform. Look-ahead safe.
#
# Key design rules enforced here:
#   1. fit() is called ONLY on training data
#   2. transform() uses ONLY data available at prediction time (all rolling windows use shift(1))
#   3. transform() requires sufficient lookback context — caller must provide it
#   4. No normalization leaks test-period statistics into training
#   5. Pipeline artifact is SHA-256 verified at load time

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeaturePipelineConfig:
    version: str                    # semver e.g. "1.0.0"
    rolling_window_bars: int        # lookback for rolling stats (e.g. 100)
    atr_period: int                 # ATR smoothing period (e.g. 14)
    rsi_period: int                 # RSI period (e.g. 14)
    min_variance_threshold: float   # features below this variance are dropped


class FeaturePipeline:
    """
    fit() determines which features survive the variance filter.
    transform() recomputes everything from raw bar data — no stored rolling state.

    CRITICAL: The caller MUST pass at least config.rolling_window_bars of
    lookback bars into transform() before the rows they want features for.
    The pipeline will return NaN for rows with insufficient lookback,
    and those rows should be dropped before model inference.
    """

    def __init__(self, config: FeaturePipelineConfig):
        self.config = config
        self._fitted = False
        self._feature_columns: list[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, df_train: pd.DataFrame) -> None:
        """
        df_train: complete bars only (is_complete=True, source != 'gap_filled').
        Determines which features pass the variance threshold.
        Does NOT store rolling statistics — transform() is stateless.
        """
        _validate_bar_df(df_train)

        features = self._compute_features(df_train)
        variances = features.var()
        surviving = variances[variances > self.config.min_variance_threshold].index.tolist()

        if len(surviving) == 0:
            raise ValueError("All features have near-zero variance. Check input data.")

        self._feature_columns = surviving
        self._fitted = True

        logger.info(
            "FeaturePipeline fitted",
            extra={
                "total_features": len(variances),
                "surviving_features": len(surviving),
                "dropped": [c for c in variances.index if c not in surviving],
                "train_rows": len(df_train),
                "version": self.config.version,
            }
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df: raw bar DataFrame including lookback context.
        Returns features for ALL rows (with NaN in warm-up rows).
        Caller should slice to the rows they want after calling this.

        Enforces that only fitted feature columns are returned.
        Raises if any fitted column is missing from computed features.
        """
        if not self._fitted:
            raise RuntimeError("FeaturePipeline.fit() must be called before transform()")

        _validate_bar_df(df)
        features = self._compute_features(df)

        missing = set(self._feature_columns) - set(features.columns)
        if missing:
            raise ValueError(
                f"Fitted features missing from transform output: {missing}. "
                f"Ensure input data has the same columns as training data."
            )

        return features[self._feature_columns]

    def save(self, path: str) -> str:
        """
        Save pipeline to directory. Returns SHA-256 of the artifact.
        Creates a manifest file alongside for integrity verification at load time.
        """
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted pipeline")

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": asdict(self.config),
            "feature_columns": self._feature_columns,
        }

        pkl_path = p / "pipeline.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(payload, f, protocol=5)

        sha256 = hashlib.sha256(pkl_path.read_bytes()).hexdigest()
        manifest = {
            "sha256": sha256,
            "version": self.config.version,
            "feature_count": len(self._feature_columns),
            "feature_columns": self._feature_columns,
        }

        with open(p / "pipeline.manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            "FeaturePipeline saved",
            extra={"path": str(p), "sha256": sha256, "version": self.config.version}
        )
        return sha256

    @classmethod
    def load(cls, path: str) -> "FeaturePipeline":
        """
        Load pipeline and verify SHA-256 integrity before returning.
        Raises RuntimeError if integrity check fails — do not catch this in live code.
        """
        p = Path(path)
        pkl_path = p / "pipeline.pkl"
        manifest_path = p / "pipeline.manifest.json"

        if not pkl_path.exists():
            raise FileNotFoundError(f"Pipeline artifact not found: {pkl_path}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Pipeline manifest not found: {manifest_path}")

        manifest = json.loads(manifest_path.read_text())
        actual_sha256 = hashlib.sha256(pkl_path.read_bytes()).hexdigest()

        if actual_sha256 != manifest["sha256"]:
            raise RuntimeError(
                f"Pipeline integrity check FAILED. "
                f"Expected {manifest['sha256']}, got {actual_sha256}. "
                f"Do not proceed. The artifact may be corrupted or tampered with."
            )

        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)

        config = FeaturePipelineConfig(**payload["config"])
        pipeline = cls(config)
        pipeline._feature_columns = payload["feature_columns"]
        pipeline._fitted = True

        logger.info(
            "FeaturePipeline loaded",
            extra={"path": str(p), "sha256": actual_sha256, "version": config.version}
        )
        return pipeline

    # ------------------------------------------------------------------
    # Feature computation (private)
    # ------------------------------------------------------------------

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features. Every feature uses shift(1) to prevent look-ahead.
        At prediction time for bar t, we only know bars up to t-1.

        All rolling windows are applied to shift(1) data, which means at row t
        the window uses rows [t-window-1, t-2], i.e., fully historical.
        """
        out = pd.DataFrame(index=df.index)
        w = self.config.rolling_window_bars

        # ---- Momentum features ----
        out["log_ret_1"] = np.log(df["close_bid"] / df["close_bid"].shift(1)).shift(1)
        out["log_ret_5"] = np.log(df["close_bid"] / df["close_bid"].shift(5)).shift(1)
        out["log_ret_12"] = np.log(df["close_bid"] / df["close_bid"].shift(12)).shift(1)

        # RSI (pure numpy — no pandas_ta required)
        out["rsi"] = _rsi(df["close_bid"], self.config.rsi_period).shift(1)

        # MACD histogram
        out["macd_hist"] = _macd_hist(df["close_bid"]).shift(1)

        # ---- Volatility features ----
        # ATR normalized by price (pure numpy)
        atr = _atr(df["high_bid"], df["low_bid"], df["close_bid"], self.config.atr_period)
        out["atr_pct"] = (atr / df["close_bid"]).shift(1)

        # Bollinger band width (pure pandas rolling)
        mid_20 = df["close_bid"].rolling(20, min_periods=10).mean()
        std_20 = df["close_bid"].rolling(20, min_periods=10).std()
        bb_width = (2 * std_20) / mid_20   # width = 2σ / mid
        out["bb_width"] = bb_width.shift(1)

        # Rolling realized volatility: std of log returns over window
        log_ret = np.log(df["close_bid"] / df["close_bid"].shift(1))
        out["realized_vol"] = log_ret.shift(1).rolling(20, min_periods=10).std()

        # ---- Microstructure features ----
        # Spread in pips: proxy for liquidity
        spread_raw = df["close_ask"] - df["close_bid"]
        spread_pips = spread_raw * _pip_mult_series(df)
        out["spread_pips"] = spread_pips.shift(1)

        # Spread Z-score: how unusual is current spread vs recent history?
        out["spread_zscore"] = _rolling_zscore(spread_pips, w).shift(1)

        # Volume Z-score: relative tick activity
        out["volume_zscore"] = _rolling_zscore(df["volume"].astype(float), w).shift(1)

        # Bid-ask ratio: asymmetry in bid/ask relative to recent range
        mid = (df["close_bid"] + df["close_ask"]) / 2
        out["ba_ratio"] = (spread_raw / mid).shift(1)

        # ---- Session features ----
        utc_dt = _get_utc_index(df)
        hour = utc_dt.hour
        out["session_london"] = ((hour >= 7) & (hour < 16)).astype(int)
        out["session_ny"] = ((hour >= 12) & (hour < 21)).astype(int)
        out["session_overlap"] = ((hour >= 12) & (hour < 16)).astype(int)
        out["session_asian"] = ((hour >= 23) | (hour < 8)).astype(int)

        # ---- Calendar features (cyclical encoding) ----
        dow = utc_dt.dayofweek.astype(float)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 5.0)

        hour_f = utc_dt.hour.astype(float) + utc_dt.minute.astype(float) / 60.0
        out["hour_sin"] = np.sin(2 * np.pi * hour_f / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour_f / 24.0)

        # Month-end proximity: last 3 trading days of month tend to have different flow
        day = utc_dt.day.astype(float)
        days_in_month = utc_dt.days_in_month.astype(float)
        out["days_to_month_end"] = (days_in_month - day) / days_in_month
        if isinstance(df.index, pd.DatetimeIndex):
            hour = df.index.hour
        else:
            hour = pd.DatetimeIndex(df["utc_open"]).hour
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        out["is_london"]  = ((hour >= 7)  & (hour < 16)).astype(float)
        out["is_ny"]      = ((hour >= 13) & (hour < 21)).astype(float)
        out["is_overlap"] = ((hour >= 13) & (hour < 16)).astype(float)

        # ---- Trend alignment (price vs longer MAs) ----
        ma50  = df["close_bid"].rolling(50,  min_periods=25).mean()
        ma200 = df["close_bid"].rolling(200, min_periods=100).mean()
        out["price_vs_ma50"]  = ((df["close_bid"] - ma50)  / ma50).shift(1)
        out["price_vs_ma200"] = ((df["close_bid"] - ma200) / ma200).shift(1)
        out["ma50_vs_ma200"]  = ((ma50 - ma200) / ma200).shift(1)

        # ---- Bar structure (order flow proxy) ----
        hi_lo = (df["high_bid"] - df["low_bid"]).replace(0, np.nan)
        out["close_position"] = ((df["close_bid"] - df["low_bid"]) / hi_lo).shift(1)
        out["upper_wick"]     = ((df["high_bid"] - df[["close_bid","open_bid"]].max(axis=1)) / hi_lo).shift(1)
        out["lower_wick"]     = ((df[["close_bid","open_bid"]].min(axis=1) - df["low_bid"]) / hi_lo).shift(1)

        # ---- Momentum across more timeframes ----
        out["log_ret_24"]  = np.log(df["close_bid"] / df["close_bid"].shift(24)).shift(1)
        out["log_ret_48"]  = np.log(df["close_bid"] / df["close_bid"].shift(48)).shift(1)
        out["log_ret_288"] = np.log(df["close_bid"] / df["close_bid"].shift(288)).shift(1)  # ~1 day

        # ---- Volume ----
        vol_ma = df["volume"].rolling(20, min_periods=10).mean().replace(0, np.nan)
        out["volume_ratio"] = (df["volume"] / vol_ma).shift(1)
        return out


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling Z-score using only past data.
    At time t, uses values from [t-window, t-1] via shift(1).
    Division by zero (flat series) returns NaN — handled by model as missing.
    """
    shifted = series.shift(1)
    mean = shifted.rolling(window, min_periods=window // 2).mean()
    std = shifted.rolling(window, min_periods=window // 2).std()
    std = std.replace(0, np.nan)
    return (series - mean) / std


def _pip_mult_series(df: pd.DataFrame) -> float:
    """Infer pip multiplier from price level. JPY pairs have different pip size."""
    median_bid = df["close_bid"].median()
    if median_bid > 50:     # JPY pairs (USD/JPY ~130–155)
        return 100.0
    return 10000.0


def _get_utc_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Extract UTC DatetimeIndex from bar DataFrame."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index
    if "utc_open" in df.columns:
        return pd.DatetimeIndex(df["utc_open"])
    raise ValueError("DataFrame must have a DatetimeIndex or 'utc_open' column")


def _validate_bar_df(df: pd.DataFrame) -> None:
    """Raise early with a clear message if the input DataFrame is malformed."""
    required = ["close_bid", "close_ask", "high_bid", "low_bid",
                "open_bid", "volume", "is_complete"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Bar DataFrame missing required columns: {missing}")
    if df.empty:
        raise ValueError("Bar DataFrame is empty")
    if df.isnull().all().any():
        all_null_cols = df.columns[df.isnull().all()].tolist()
        raise ValueError(f"Columns are entirely null: {all_null_cols}")


# ------------------------------------------------------------------
# Pure numpy/pandas indicator implementations (no pandas_ta required)
# ------------------------------------------------------------------

def _rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Wilder RSI. Returns series in [0, 100].
    Uses Wilder smoothing (EWM with alpha=1/period, adjust=False).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram = MACD line - signal line."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    Average True Range using Wilder smoothing.
    True Range = max(H-L, |H-Cprev|, |L-Cprev|)
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
