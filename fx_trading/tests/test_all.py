# tests/test_all.py
# Full test suite using stdlib unittest only (no pytest dependency).
# Run with: python -m unittest tests.test_all -v

import sys
import os
import unittest
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from decimal import Decimal


# ── Helpers ───────────────────────────────────────────────────

def _make_bar_df(n: int = 200, pair: str = "EUR/USD", start_price: float = 1.09) -> pd.DataFrame:
    dates = pd.date_range(start="2024-01-01 00:00", periods=n, freq="5min", tz="UTC")
    np.random.seed(42)
    prices = start_price + np.cumsum(np.random.randn(n) * 0.0001)
    spread = 0.0002
    df = pd.DataFrame({
        "utc_open": dates,
        "utc_close": dates + timedelta(minutes=5),
        "open_bid":  prices,
        "high_bid":  prices + np.abs(np.random.randn(n)) * 0.0003,
        "low_bid":   prices - np.abs(np.random.randn(n)) * 0.0003,
        "close_bid": prices + np.random.randn(n) * 0.0001,
        "open_ask":  prices + spread,
        "high_ask":  prices + spread + np.abs(np.random.randn(n)) * 0.0003,
        "low_ask":   prices + spread - np.abs(np.random.randn(n)) * 0.0003,
        "close_ask": prices + spread + np.random.randn(n) * 0.0001,
        "volume":    np.random.randint(100, 1000, n),
        "is_complete": True,
        "source":    "broker_historical",
        "pair":      pair,
    }, index=dates)
    return df


class MockStateStore:
    def __init__(self):
        self._kill = False
        self._consec = 0
        self._bars = 999
        self._hwm = 0.0
        self._daily_pnl = 0.0

    def get_kill_switch(self): return self._kill
    def set_kill_switch(self, v, reason=""): self._kill = v
    def get_consecutive_losses(self): return self._consec
    def get_bars_since_loss_limit(self): return self._bars
    def reset_consecutive_losses(self): self._consec = 0
    def increment_consecutive_losses(self):
        self._consec += 1; self._bars = 0; return self._consec
    def increment_bar_count(self): self._bars += 1
    def add_daily_pnl(self, amount, date): self._daily_pnl += amount
    def get_daily_pnl(self, date): return self._daily_pnl
    def get_high_water_mark(self): return self._hwm
    def update_high_water_mark(self, eq): self._hwm = max(self._hwm, eq)
    def get_drawdown_from_peak(self, eq):
        if self._hwm <= 0: return 0.0
        return max(0.0, (self._hwm - eq) / self._hwm)
    def set_state(self, k, v): pass
    def set_heartbeat(self): pass


# ===========================================================================
# shared/schemas.py
# ===========================================================================

class TestSchemas(unittest.TestCase):

    def test_assert_utc_raises_on_naive(self):
        """Prevents timezone misalignment bugs caused by naive datetimes."""
        from shared.schemas import assert_utc
        naive = datetime(2024, 1, 15, 12, 0, 0)
        with self.assertRaises(AssertionError):
            assert_utc(naive, "test")

    def test_assert_utc_raises_on_non_utc(self):
        """Prevents broker local time being treated as UTC."""
        from shared.schemas import assert_utc
        from datetime import timezone as tz
        # Create offset-aware datetime that is NOT UTC
        non_utc = datetime(2024, 1, 15, 12, 0, 0,
                           tzinfo=tz(timedelta(hours=2)))  # UTC+2
        with self.assertRaises(AssertionError):
            assert_utc(non_utc, "test")

    def test_assert_utc_passes_on_utc(self):
        """UTC-aware datetimes must pass the guard cleanly."""
        from shared.schemas import assert_utc
        utc_dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(assert_utc(utc_dt, "test"), utc_dt)

    def test_pip_size_eurusd(self):
        from shared.schemas import pip_size, Pair
        self.assertEqual(pip_size(Pair.EURUSD), Decimal("0.0001"))

    def test_pip_size_usdjpy(self):
        from shared.schemas import pip_size, Pair
        self.assertEqual(pip_size(Pair.USDJPY), Decimal("0.01"))

    def test_pip_multiplier_eurusd(self):
        from shared.schemas import pip_multiplier, Pair
        self.assertEqual(pip_multiplier(Pair.EURUSD), 10000.0)

    def test_pip_multiplier_usdjpy(self):
        from shared.schemas import pip_multiplier, Pair
        self.assertEqual(pip_multiplier(Pair.USDJPY), 100.0)

    def test_order_gets_unique_client_id(self):
        """Every order must have a unique idempotency key."""
        from shared.schemas import Order, Pair, Side, OrderType
        orders = [Order(pair=Pair.EURUSD, side=Side.BUY,
                        order_type=OrderType.MARKET, units=1000)
                  for _ in range(100)]
        ids = [o.client_order_id for o in orders]
        self.assertEqual(len(set(ids)), 100, "Duplicate client_order_ids detected")

    def test_ohlcvbar_is_immutable(self):
        """Bars must be immutable — frozen dataclass."""
        from shared.schemas import OHLCVBar, Pair, BarSource
        bar = OHLCVBar(
            pair=Pair.EURUSD,
            utc_open=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
            utc_close=datetime(2024, 1, 15, 12, 5, tzinfo=timezone.utc),
            timeframe_sec=300,
            open_bid=Decimal("1.0900"), high_bid=Decimal("1.0910"),
            low_bid=Decimal("1.0895"), close_bid=Decimal("1.0905"),
            open_ask=Decimal("1.0901"), high_ask=Decimal("1.0911"),
            low_ask=Decimal("1.0896"), close_ask=Decimal("1.0906"),
            volume=100, is_complete=True, source=BarSource.BROKER_HISTORICAL,
        )
        with self.assertRaises((AttributeError, TypeError)):
            bar.volume = 999

    def test_tick_rejects_negative_bid(self):
        """Ticks with invalid prices must be rejected at construction."""
        from shared.schemas import Tick, Pair
        with self.assertRaises(AssertionError):
            Tick(pair=Pair.EURUSD,
                 utc_ts=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
                 bid=Decimal("-1.0"), ask=Decimal("1.0901"), volume=1)

    def test_tick_rejects_ask_less_than_bid(self):
        """Ask < bid is a data integrity error."""
        from shared.schemas import Tick, Pair
        with self.assertRaises(AssertionError):
            Tick(pair=Pair.EURUSD,
                 utc_ts=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
                 bid=Decimal("1.0910"), ask=Decimal("1.0900"), volume=1)


# ===========================================================================
# data/time_utils.py
# ===========================================================================

class TestTimeUtils(unittest.TestCase):

    def test_mt5_server_to_utc_converts_correctly(self):
        """MT5 naive datetimes must be converted correctly to UTC."""
        from data.time_utils import mt5_server_to_utc
        naive = datetime(2024, 1, 15, 14, 0, 0)  # 14:00 broker time (UTC+2)
        utc = mt5_server_to_utc(naive, "Etc/GMT-2")
        self.assertEqual(utc.hour, 12)   # 14:00 UTC+2 = 12:00 UTC
        self.assertIsNotNone(utc.tzinfo)

    def test_mt5_server_to_utc_raises_on_aware(self):
        """MT5 always returns naive datetimes — raise if tz-aware is passed."""
        from data.time_utils import mt5_server_to_utc
        aware = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        with self.assertRaises(AssertionError):
            mt5_server_to_utc(aware, "Etc/GMT-2")

    def test_oanda_str_to_utc(self):
        """OANDA RFC3339 strings must parse correctly."""
        from data.time_utils import oanda_str_to_utc
        ts = "2024-01-15T13:45:00.000000000Z"
        dt = oanda_str_to_utc(ts)
        self.assertEqual(dt.year, 2024)
        self.assertEqual(dt.month, 1)
        self.assertEqual(dt.hour, 13)
        self.assertEqual(dt.tzinfo, timezone.utc)

    def test_is_market_open_saturday_closed(self):
        """Saturday FX market is always closed."""
        from data.time_utils import is_market_open
        saturday = datetime(2024, 1, 13, 12, 0, 0, tzinfo=timezone.utc)
        self.assertFalse(is_market_open(saturday))

    def test_is_market_open_sunday_morning_closed(self):
        """Sunday morning (before 21:00 UTC) is closed."""
        from data.time_utils import is_market_open
        sunday_morning = datetime(2024, 1, 14, 10, 0, 0, tzinfo=timezone.utc)
        self.assertFalse(is_market_open(sunday_morning))

    def test_is_market_open_sunday_evening_open(self):
        """Sunday after 21:00 UTC is open."""
        from data.time_utils import is_market_open
        sunday_evening = datetime(2024, 1, 14, 21, 30, 0, tzinfo=timezone.utc)
        self.assertTrue(is_market_open(sunday_evening))

    def test_is_market_open_friday_after_21_closed(self):
        """Friday after 21:00 UTC is closed."""
        from data.time_utils import is_market_open
        friday_late = datetime(2024, 1, 12, 21, 30, 0, tzinfo=timezone.utc)
        self.assertFalse(is_market_open(friday_late))

    def test_floor_to_bar(self):
        """Bar flooring must align to exact bar boundaries."""
        from data.time_utils import floor_to_bar
        dt = datetime(2024, 1, 15, 13, 47, 23, tzinfo=timezone.utc)
        floored = floor_to_bar(dt, 300)  # 5m bars
        self.assertEqual(floored.minute, 45)
        self.assertEqual(floored.second, 0)

    def test_now_utc_is_tz_aware(self):
        """now_utc() must always return a tz-aware datetime."""
        from data.time_utils import now_utc
        dt = now_utc()
        self.assertIsNotNone(dt.tzinfo)


# ===========================================================================
# features/pipeline.py
# ===========================================================================

class TestFeaturePipeline(unittest.TestCase):

    def _make_pipeline(self, window=50):
        from features.pipeline import FeaturePipeline, FeaturePipelineConfig
        config = FeaturePipelineConfig(
            version="test_v1",
            rolling_window_bars=window,
            atr_period=14,
            rsi_period=14,
            min_variance_threshold=1e-10,
        )
        return FeaturePipeline(config)

    def test_fit_transform_no_nan_after_warmup(self):
        """After warmup rows, features must not contain NaN."""
        pipeline = self._make_pipeline()
        df = _make_bar_df(200)
        pipeline.fit(df)
        features = pipeline.transform(df)
        valid = features.iloc[60:]
        nan_cols = valid.columns[valid.isnull().any()].tolist()
        self.assertEqual(nan_cols, [], f"NaN in features after warmup: {nan_cols}")

    def test_no_lookahead_modification_test(self):
        """Modifying bar t must not change features at bar t-1 (look-ahead check)."""
        pipeline = self._make_pipeline()
        df = _make_bar_df(150)
        pipeline.fit(df)
        features_original = pipeline.transform(df.copy()).copy()
        df_modified = df.copy()
        df_modified.iloc[-1, df_modified.columns.get_loc("close_bid")] = 999.0
        features_modified = pipeline.transform(df_modified)
        if len(features_original) >= 2:
            orig_row = features_original.iloc[-2]
            mod_row = features_modified.iloc[-2]
            for col in orig_row.index:
                if pd.notna(orig_row[col]) and pd.notna(mod_row[col]):
                    self.assertLess(abs(orig_row[col] - mod_row[col]), 0.01,
                        f"Look-ahead in column '{col}': {orig_row[col]:.6f} vs {mod_row[col]:.6f}")

    def test_save_load_integrity(self):
        """Pipeline must survive save/load with SHA-256 integrity check."""
        tmp = tempfile.mkdtemp()
        try:
            pipeline = self._make_pipeline()
            df = _make_bar_df(200)
            pipeline.fit(df)
            original = pipeline.transform(df)
            save_path = os.path.join(tmp, "pipeline")
            pipeline.save(save_path)
            loaded = type(pipeline).load(save_path)
            loaded_features = loaded.transform(df)
            pd.testing.assert_frame_equal(original, loaded_features)
        finally:
            shutil.rmtree(tmp)

    def test_load_fails_on_tampered_artifact(self):
        """Tampered artifacts must raise RuntimeError."""
        tmp = tempfile.mkdtemp()
        try:
            pipeline = self._make_pipeline()
            pipeline.fit(_make_bar_df(200))
            save_path = os.path.join(tmp, "pipeline")
            pipeline.save(save_path)
            pkl_path = os.path.join(save_path, "pipeline.pkl")
            with open(pkl_path, "ab") as f:
                f.write(b"tampered")
            with self.assertRaises(RuntimeError):
                type(pipeline).load(save_path)
        finally:
            shutil.rmtree(tmp)

    def test_transform_raises_if_not_fitted(self):
        """Unfitted pipeline must not silently return garbage."""
        pipeline = self._make_pipeline()
        with self.assertRaises(RuntimeError):
            pipeline.transform(_make_bar_df(100))


# ===========================================================================
# research/walk_forward.py
# ===========================================================================

class TestWalkForward(unittest.TestCase):

    def _make_config(self, **kwargs):
        from research.walk_forward import WalkForwardConfig
        defaults = dict(train_bars=1000, purge_bars=10, val_bars=200,
                        embargo_bars=10, test_bars=100, step_bars=100, min_folds=3)
        defaults.update(kwargs)
        return WalkForwardConfig(**defaults)

    def test_no_overlap_between_splits(self):
        """Any overlap between splits = look-ahead bias. Must be zero."""
        from research.walk_forward import generate_folds
        folds = generate_folds(5000, self._make_config())
        for fold in folds:
            train = set(range(fold.train_start, fold.train_end))
            val   = set(range(fold.val_start, fold.val_end))
            test  = set(range(fold.test_start, fold.test_end))
            self.assertFalse(train & val,  f"Fold {fold.fold_id}: train/val overlap")
            self.assertFalse(train & test, f"Fold {fold.fold_id}: train/test overlap")
            self.assertFalse(val & test,   f"Fold {fold.fold_id}: val/test overlap")

    def test_purge_zone_is_sufficient(self):
        """Purge zone must be >= purge_bars to prevent label boundary leakage."""
        from research.walk_forward import generate_folds
        config = self._make_config(purge_bars=12)
        folds = generate_folds(5000, config)
        for fold in folds:
            purge_zone = fold.val_start - fold.train_end
            self.assertGreaterEqual(purge_zone, config.purge_bars,
                f"Fold {fold.fold_id}: purge zone {purge_zone} < {config.purge_bars}")

    def test_embargo_zone_is_sufficient(self):
        """Embargo zone must prevent feature leakage from val into test."""
        from research.walk_forward import generate_folds
        config = self._make_config(embargo_bars=12)
        folds = generate_folds(5000, config)
        for fold in folds:
            embargo_zone = fold.test_start - fold.val_end
            self.assertGreaterEqual(embargo_zone, config.embargo_bars)

    def test_insufficient_data_raises(self):
        """Must raise clearly when data is too short for the fold config."""
        from research.walk_forward import generate_folds, WalkForwardConfig
        config = WalkForwardConfig(
            train_bars=100_000, purge_bars=12, val_bars=20_000,
            embargo_bars=12, test_bars=10_000, step_bars=10_000, min_folds=6)
        with self.assertRaises(ValueError):
            generate_folds(1000, config)

    def test_folds_are_chronologically_ordered(self):
        """Each fold must start after the previous one."""
        from research.walk_forward import generate_folds
        folds = generate_folds(5000, self._make_config())
        for i in range(1, len(folds)):
            self.assertGreater(folds[i].train_start, folds[i-1].train_start)
            self.assertGreater(folds[i].test_start,  folds[i-1].test_start)


# ===========================================================================
# risk/engine.py
# ===========================================================================

class TestRiskEngine(unittest.TestCase):

    def _make_engine(self, **overrides):
        from risk.engine import RiskEngine, RiskParameters
        params = RiskParameters(**overrides)
        return RiskEngine(params, MockStateStore())

    def test_kill_switch_blocks_all_trades(self):
        """Kill switch must absolutely prevent any order submission."""
        from risk.engine import RiskEngine, RiskParameters
        from shared.schemas import Pair
        state = MockStateStore(); state._kill = True
        engine = RiskEngine(RiskParameters(), state)
        d = engine.evaluate(signal=1, confidence=0.8, pair=Pair.EURUSD,
                            current_bid=Decimal("1.0900"), current_ask=Decimal("1.0901"),
                            atr_pips=10.0, equity_usd=10000.0,
                            open_positions=[], daily_pnl_usd=0.0)
        self.assertFalse(d.trade_permitted)
        self.assertIn("kill", d.reason.lower())

    def test_abstain_signal_denied(self):
        """ABSTAIN signal must never reach order submission."""
        from shared.schemas import Pair
        engine = self._make_engine()
        d = engine.evaluate(signal=0, confidence=0.9, pair=Pair.EURUSD,
                            current_bid=Decimal("1.0900"), current_ask=Decimal("1.0901"),
                            atr_pips=10.0, equity_usd=10000.0,
                            open_positions=[], daily_pnl_usd=0.0)
        self.assertFalse(d.trade_permitted)

    def test_daily_drawdown_circuit_breaker(self):
        """3% daily drawdown must halt all trading for the day."""
        from shared.schemas import Pair
        engine = self._make_engine(max_daily_drawdown_pct=0.03)
        d = engine.evaluate(signal=1, confidence=0.8, pair=Pair.EURUSD,
                            current_bid=Decimal("1.0900"), current_ask=Decimal("1.0901"),
                            atr_pips=10.0, equity_usd=9700.0,
                            open_positions=[], daily_pnl_usd=-300.0)
        self.assertFalse(d.trade_permitted)
        self.assertIn("drawdown", d.reason.lower())

    def test_position_size_within_risk_limit(self):
        """Max loss must not exceed risk_per_trade_pct of equity."""
        from shared.schemas import Pair
        engine = self._make_engine(risk_per_trade_pct=0.01)
        d = engine.evaluate(signal=1, confidence=0.8, pair=Pair.EURUSD,
                            current_bid=Decimal("1.0900"), current_ask=Decimal("1.0901"),
                            atr_pips=10.0, equity_usd=10000.0,
                            open_positions=[], daily_pnl_usd=0.0)
        self.assertTrue(d.trade_permitted)
        self.assertLessEqual(d.max_loss_usd, 110.0,
            f"Max loss ${d.max_loss_usd:.2f} exceeds 1% risk limit")

    def test_low_atr_blocked(self):
        """Trades in flat markets (low ATR) must be blocked."""
        from shared.schemas import Pair
        engine = self._make_engine(min_atr_pips=5.0)
        d = engine.evaluate(signal=1, confidence=0.8, pair=Pair.EURUSD,
                            current_bid=Decimal("1.0900"), current_ask=Decimal("1.0901"),
                            atr_pips=2.0, equity_usd=10000.0,
                            open_positions=[], daily_pnl_usd=0.0)
        self.assertFalse(d.trade_permitted)
        self.assertIn("atr", d.reason.lower())

    def test_consecutive_loss_cooldown(self):
        """After max consecutive losses, trading must pause for cooldown_bars."""
        from risk.engine import RiskEngine, RiskParameters
        from shared.schemas import Pair
        state = MockStateStore(); state._consec = 5; state._bars = 3
        engine = RiskEngine(RiskParameters(max_consecutive_losses=4, cooldown_bars=12), state)
        d = engine.evaluate(signal=1, confidence=0.8, pair=Pair.EURUSD,
                            current_bid=Decimal("1.0900"), current_ask=Decimal("1.0901"),
                            atr_pips=10.0, equity_usd=10000.0,
                            open_positions=[], daily_pnl_usd=0.0)
        self.assertFalse(d.trade_permitted)
        self.assertIn("cooldown", d.reason.lower())

    def test_stop_price_outside_spread_for_long(self):
        """LONG stop must be below bid (not inside the spread)."""
        from shared.schemas import Pair
        engine = self._make_engine()
        bid = Decimal("1.0900"); ask = Decimal("1.0902")
        d = engine.evaluate(signal=1, confidence=0.8, pair=Pair.EURUSD,
                            current_bid=bid, current_ask=ask,
                            atr_pips=10.0, equity_usd=10000.0,
                            open_positions=[], daily_pnl_usd=0.0)
        self.assertTrue(d.trade_permitted)
        self.assertLess(d.stop_price, bid,
            f"Stop {d.stop_price} must be below bid {bid}")


# ===========================================================================
# state/store.py
# ===========================================================================

class TestStateStore(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _make_store(self):
        from state.store import StateStore
        return StateStore(os.path.join(self.tmp, "test.db"))

    def test_kill_switch_persists_across_restart(self):
        """Kill switch must survive process restart (write to disk, re-read)."""
        db_path = os.path.join(self.tmp, "kill_test.db")
        from state.store import StateStore
        store = StateStore(db_path)
        self.assertFalse(store.get_kill_switch())
        store.set_kill_switch(True, "test")
        # Simulate restart: new instance, same DB file
        store2 = StateStore(db_path)
        self.assertTrue(store2.get_kill_switch())

    def test_order_idempotent_upsert(self):
        """Upserting the same order twice must not create duplicates."""
        from shared.schemas import Order, OrderStatus, Pair, Side, OrderType
        store = self._make_store()
        order = Order(pair=Pair.EURUSD, side=Side.BUY,
                      order_type=OrderType.MARKET, units=1000,
                      stop_loss_price=Decimal("1.0850"))
        store.upsert_order(order)
        order.status = OrderStatus.SUBMITTED
        order.broker_order_id = "broker_123"
        store.upsert_order(order)
        retrieved = store.get_order(order.client_order_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.status, OrderStatus.SUBMITTED)
        self.assertEqual(retrieved.broker_order_id, "broker_123")

    def test_daily_pnl_resets_on_new_day(self):
        """Daily P&L must reset to zero at the start of a new day."""
        store = self._make_store()
        store.add_daily_pnl(100.0, "2024-01-15")
        self.assertEqual(store.get_daily_pnl("2024-01-15"), 100.0)
        self.assertEqual(store.get_daily_pnl("2024-01-16"), 0.0)

    def test_high_water_mark_only_increases(self):
        """HWM must never decrease — only peak equity is recorded."""
        store = self._make_store()
        store.update_high_water_mark(10000.0)
        store.update_high_water_mark(9500.0)  # drawdown — must not update HWM
        self.assertEqual(store.get_high_water_mark(), 10000.0)

    def test_drawdown_from_peak_calculation(self):
        """Drawdown calculation must be correct to avoid false circuit triggers."""
        store = self._make_store()
        store.update_high_water_mark(10000.0)
        dd = store.get_drawdown_from_peak(9700.0)
        self.assertAlmostEqual(dd, 0.03, places=3,
            msg=f"Expected 3% drawdown, got {dd:.3%}")

    def test_position_round_trip(self):
        """Position data must survive write/read without data loss."""
        from shared.schemas import Position, Pair, Side
        store = self._make_store()
        pos = Position(
            position_id="test_pos_1",
            client_order_id="test_order_1",
            pair=Pair.EURUSD,
            side=Side.BUY,
            units=10000,
            entry_price=Decimal("1.09001"),
            entry_utc=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            stop_loss_price=Decimal("1.08900"),
        )
        store.upsert_position(pos)
        retrieved = store.get_position("test_pos_1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.units, 10000)
        self.assertAlmostEqual(float(retrieved.entry_price), 1.09001, places=5)


# ===========================================================================
# research/model_training.py
# ===========================================================================

class TestModelSignal(unittest.TestCase):

    def test_abstain_when_low_confidence(self):
        """Low-confidence predictions must produce ABSTAIN."""
        from research.model_training import compute_signal
        proba = np.array([0.35, 0.35, 0.30])
        signal, conf = compute_signal(proba, min_confidence=0.55)
        self.assertEqual(signal, 0, f"Expected ABSTAIN, got {signal}")

    def test_abstain_when_abstain_class_wins(self):
        """Dominant abstain class must not trigger a trade."""
        from research.model_training import compute_signal
        proba = np.array([0.20, 0.60, 0.20])
        signal, conf = compute_signal(proba, min_confidence=0.55)
        self.assertEqual(signal, 0)

    def test_long_signal_when_confident(self):
        """Clear LONG confidence must produce LONG signal."""
        from research.model_training import compute_signal
        proba = np.array([0.10, 0.15, 0.75])
        signal, conf = compute_signal(proba, min_confidence=0.55, min_margin=0.10)
        self.assertEqual(signal, 1)
        self.assertAlmostEqual(conf, 0.75, places=5)

    def test_short_signal_when_confident(self):
        """Clear SHORT confidence must produce SHORT signal."""
        from research.model_training import compute_signal
        proba = np.array([0.75, 0.15, 0.10])
        signal, conf = compute_signal(proba, min_confidence=0.55, min_margin=0.10)
        self.assertEqual(signal, -1)

    def test_abstain_when_margin_too_small(self):
        """Ambiguous top-two predictions must abstain."""
        from research.model_training import compute_signal
        proba = np.array([0.05, 0.47, 0.48])
        signal, conf = compute_signal(proba, min_confidence=0.40, min_margin=0.10)
        self.assertEqual(signal, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
