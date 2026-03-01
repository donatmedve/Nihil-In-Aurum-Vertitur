# FX Trading System — Complete Documentation

This is a production-grade, ML-driven algorithmic trading system for foreign exchange (FX) markets. It trades EUR/USD, GBP/USD, and USD/JPY on 5-minute bars using a LightGBM or XGBoost classifier trained with a walk-forward framework.

This document explains **every file, every design decision, and every concept** in the system. Read it top to bottom before writing a single line of code.

---

## Table of Contents

1. [Architecture Philosophy](#1-architecture-philosophy)
2. [Project Structure](#2-project-structure)
3. [Dependency Rules](#3-dependency-rules)
4. [Module Reference](#4-module-reference)
   - [shared/schemas.py](#41-sharedschemaspy)
   - [data/time_utils.py](#42-datatime_utilspy)
   - [data/aggregation.py](#43-dataaggregationpy)
   - [features/pipeline.py](#44-featurespipelinepy)
   - [research/labels.py](#45-researchlabelspy)
   - [research/walk_forward.py](#46-researchwalk_forwardpy)
   - [research/model_training.py](#47-researchmodel_trainingpy)
   - [risk/engine.py](#48-riskenginely)
   - [state/store.py](#49-statestorepy)
   - [broker/adapter.py](#410-brokeradapterpy)
   - [broker/mt5_adapter.py](#411-brokermt5_adapterpy)
   - [execution/reconciliation.py](#412-executionreconciliationpy)
   - [execution/live_loop.py](#413-executionlive_looppy)
   - [tests/test_all.py](#414-teststest_allpy)
5. [The ML Pipeline End-to-End](#5-the-ml-pipeline-end-to-end)
6. [The Signal-to-Order Path](#6-the-signal-to-order-path)
7. [Crash Safety and Recovery](#7-crash-safety-and-recovery)
8. [Look-Ahead Bias — What It Is and How We Prevent It](#8-look-ahead-bias--what-it-is-and-how-we-prevent-it)
9. [Risk Management](#9-risk-management)
10. [Deployment Guide](#10-deployment-guide)
11. [Common Pitfalls and How to Avoid Them](#11-common-pitfalls-and-how-to-avoid-them)

---

## 1. Architecture Philosophy

### The core problem

Algorithmic trading has three distinct environments that must never bleed into each other:

1. **Research** — offline, works with historical data, can take hours to run, has no latency constraints. This is where you train models and test hypotheses.

2. **Backtesting** — offline simulation that replays historical data through the exact same code path as live trading, to verify the system behaves correctly before real money is at risk.

3. **Execution** — live, stateful, latency-sensitive, and crash-safe. Bugs here cost real money.

The most dangerous bugs are the ones that silently make your research look better than it actually is — specifically, **look-ahead bias** (accidentally letting the model see future data during training) and **overfitting** (training and testing on the same data). This system is structured to make both of these categories of bugs impossible by design.

### Why modules are split the way they are

- `research/` contains code that is only ever run offline. It imports from `shared/` and `features/`, but **never** from `execution/` or `broker/`. This makes it structurally impossible to accidentally route live order flow through research code.

- `broker/` contains only the adapter interface and concrete broker implementations. It has no model logic, no risk logic, and no feature computation. The broker knows nothing about why an order was placed.

- `execution/` contains the live loop and crash recovery. It receives signals and decisions from other modules; it does not compute them.

- `shared/schemas.py` is the one place all data contracts are defined. Every other module imports from here. You never redefine `Order` in two places.

---

## 2. Project Structure

```
fx_trading/
│
├── shared/
│   └── schemas.py          ← All typed data contracts. Import from here everywhere.
│
├── data/
│   ├── time_utils.py       ← UTC enforcement, broker time conversion, session helpers
│   └── aggregation.py      ← Tick-to-bar aggregation with gap-fill rules
│
├── features/
│   └── pipeline.py         ← Feature computation. Stateless. Look-ahead safe.
│
├── research/
│   ├── labels.py           ← Triple barrier label construction (offline only)
│   ├── walk_forward.py     ← Rolling walk-forward fold generation
│   └── model_training.py   ← Train, save, load models with SHA-256 integrity
│
├── risk/
│   └── engine.py           ← Circuit breakers and position sizing
│
├── state/
│   └── store.py            ← SQLite WAL persistence. Survives kill -9.
│
├── broker/
│   ├── adapter.py          ← Abstract broker interface + idempotent submitter
│   └── mt5_adapter.py      ← MetaTrader 5 concrete implementation
│
├── execution/
│   ├── reconciliation.py   ← Crash recovery: 7-step broker/local reconciliation
│   └── live_loop.py        ← Main execution loop
│
├── backtesting/            ← (placeholder for event-driven backtester)
├── monitoring/             ← (placeholder for alerting/heartbeat)
├── tests/
│   └── test_all.py         ← 48 tests covering every critical safety property
│
├── artifacts/              ← Model and pipeline artifacts stored here
├── logs/                   ← JSONL log files
└── requirements.txt
```

---

## 3. Dependency Rules

The module dependency graph is strictly one-directional:

```
shared/ ←── data/ ←── features/ ←── research/
                  ←── risk/
                  ←── state/
                  ←── broker/
                  ←── execution/
```

**Rules:**

- `shared/` imports nothing from this project (stdlib only)
- `data/` imports only from `shared/`
- `features/` imports from `shared/` and `data/`
- `research/` imports from `shared/`, `data/`, and `features/`
- `risk/` imports from `shared/` only
- `state/` imports from `shared/` and `data/`
- `broker/` imports from `shared/` and `data/`
- `execution/` imports from `shared/`, `data/`, `risk/`, `state/`, `broker/`, and `research/`

**What is never allowed:**

- `research/` importing from `execution/` or `broker/`
- `risk/` importing model logic
- `broker/` importing feature or model logic
- Circular imports of any kind

If you add a new module, draw its dependency arrows first. If any arrow points "upstream" in the list above, you have a design problem.

---

## 4. Module Reference

### 4.1 `shared/schemas.py`

**Purpose:** Canonical data contracts. The single source of truth for every type used across the system.

This is the first file that was written, and it's the foundation everything else builds on. If you ever need to add a field to an `Order`, you add it here and only here.

#### Key design decisions

**Frozen dataclasses for market data (`Tick`, `OHLCVBar`):** These are immutable. Once a bar is created, it cannot be changed. This prevents a whole class of bugs where a bar gets modified in one place and the modification silently propagates elsewhere.

**`Decimal` for prices, never `float`:** Floating-point arithmetic has rounding errors. `1.0900 + 0.0001` in float arithmetic might give `1.0900000000000001`. When calculating stop loss prices or P&L, these errors accumulate and can cause incorrect position sizing. `Decimal` uses exact decimal arithmetic.

**UTC enforcement on every timestamp:** Every `datetime` in the system must be timezone-aware and in UTC. The `__post_init__` validators on each dataclass check this at construction time. If you somehow pass a naive datetime, you get an `AssertionError` immediately rather than a subtly wrong log timestamp hours later.

**`assert_utc()` and `now_utc()` defined in schemas.py (not time_utils.py):** These two functions are needed by almost every module, including `state/store.py`. If they were in `data/time_utils.py`, then `state/store.py` importing them would create a chain `state → data → shared`, which is fine, but `shared` defining them means every module can import them without importing `data/`, keeping the dependency graph simple.

#### The enums in detail

**`OrderStatus` — the lifecycle of an order:**

```
PENDING_SUBMIT  → written to DB, not yet sent to broker
       ↓
SUBMITTED       → sent to broker, awaiting fill confirmation
       ↓
FILLED          → fully filled (terminal)
PARTIALLY_FILLED→ partially filled (not terminal, awaiting more)
CANCELLED       → cancelled by us or broker (terminal)
REJECTED        → broker refused to accept it (terminal, do not retry)
DEAD_LETTERED   → all retries exhausted (terminal, requires human review)
UNCONFIRMED     → found in PENDING_SUBMIT or SUBMITTED state on crash recovery
                  (do not auto-resubmit — status is unknown)
```

The distinction between `REJECTED` (broker refused) and `DEAD_LETTERED` (network/timeout failure) is important. A `REJECTED` order should never be retried — the broker explicitly said no, and retrying will just generate more rejections. A `DEAD_LETTERED` order failed for a technical reason and needs a human to decide what to do.

**`PositionStatus`:**

- `OPEN` — live position
- `CLOSED` — normal close
- `ORPHANED` — found on broker but not in our database (e.g., manually opened, or opened before our system started). Never auto-close these.
- `BROKER_CLOSED` — was in our database as OPEN but broker has no record (e.g., stopped out, expired, or closed manually while system was down)

**`BarSource`:** Tags every bar with where it came from. `GAP_FILLED` bars are forward-filled placeholders created when there were no ticks for 1-3 bars. They have zero volume and their OHLCV are just copies of the previous close. **Never use gap-filled bars for signal generation or label construction.**

#### `Signal` — the output of the model

```python
@dataclass
class Signal:
    pair: Pair
    bar_utc: datetime
    direction: int          # -1=SHORT, 0=ABSTAIN, 1=LONG
    confidence: float       # max class probability [0.0, 1.0]
    model_version: str
    abstain_reason: Optional[str]
```

`direction=0` means the model is abstaining — it doesn't have enough confidence to recommend a trade. This is a valid and expected output. The system is designed to skip most bars.

#### `RiskDecision` — the output of the risk engine

```python
@dataclass
class RiskDecision:
    trade_permitted: bool
    reason: str
    position_size_units: int    # 0 if trade_permitted=False
    stop_distance_pips: float
    stop_price: Decimal
    max_loss_usd: float
```

The risk engine always returns a `reason` regardless of whether the trade is permitted or denied. This is logged every time for audit trail purposes.

---

### 4.2 `data/time_utils.py`

**Purpose:** All timezone conversion logic lives here. Broker-local time is converted to UTC at the ingestion boundary, and nowhere else in the system ever touches raw broker timestamps.

#### Why this matters

MetaTrader 5 returns naive `datetime` objects in the broker's server time, which is typically UTC+2 in winter and UTC+3 in summer (Eastern European Time). If you naively treat these as UTC and store them, every timestamp will be wrong by 2-3 hours. This won't crash anything immediately — it will silently shift your session feature flags and give you incorrect market-hours logic. You'll trade during the wrong sessions and your time-of-day features will be meaningless.

The fix is `mt5_server_to_utc()`:

```python
def mt5_server_to_utc(dt_naive: datetime, broker_tz_str: str) -> datetime:
```

Call this **at the point where data enters the system** — in the MT5 adapter's ingestion methods, before the data is returned to any other module. Once it's UTC, it stays UTC.

#### `now_utc()` vs `datetime.utcnow()`

**Never use `datetime.utcnow()`.** This returns a naive datetime (no tzinfo) even though it represents UTC time. Naive datetimes look like regular datetimes but silently break timezone comparisons. `now_utc()` uses `datetime.now(timezone.utc)` which returns a timezone-aware UTC datetime.

#### Market hours logic

`is_market_open()` implements the standard FX market schedule:
- Open: Sunday 21:00 UTC (Sydney open)
- Close: Friday 21:00 UTC (NY close)
- Saturday: always closed

This is used in two places: gap-fill logic (don't fill gaps during weekends) and the live loop (stop generating signals outside market hours).

#### `floor_to_bar()`

```python
floor_to_bar(datetime(2024, 1, 15, 13, 47, 23, tzinfo=utc), 300)
# → datetime(2024, 1, 15, 13, 45, 0, tzinfo=utc)
```

Rounds a timestamp down to the nearest bar boundary. Used to assign ticks to their containing bar.

---

### 4.3 `data/aggregation.py`

**Purpose:** Convert a stream of raw ticks into OHLCV bars with consistent gap-handling rules.

#### Bar completion rule

A bar is only marked `is_complete=True` when **the next bar's first tick arrives**. The last bar in any batch is always `is_complete=False`. This is critical: you must never generate a signal from an incomplete bar, because the close price may change before the bar actually closes.

In the live loop, the system waits until `is_complete=True` on the current bar before generating a signal. The `BAR_COMPLETION_TIMEOUT_SECONDS = 330` (5m + 30s grace) handles the case where no new ticks arrive to "complete" the bar — if 5.5 minutes have passed with no new tick, the bar is forced to complete.

#### Gap-fill rules

| Situation | Action |
|---|---|
| 1-3 missing bars on a weekday | Forward-fill with `volume=0`, `source='gap_filled'` |
| >3 missing bars on a weekday | Leave gap. Treat as session break. |
| Any gap on weekend | Leave gap. Weekend gaps are expected. |

The reason for the 3-bar limit: gaps up to 15 minutes are typically just data feed hiccups or brief illiquidity. Gaps longer than 15 minutes on a weekday usually mean something abnormal happened (connectivity loss, news event, broker outage), and forward-filling across these would create artificially flat bars that mislead the model.

**Gap-filled bars are flagged and excluded from:**
- Label construction (in `research/labels.py`)
- Feature computation inputs (should be filtered by caller before `pipeline.transform()`)
- Signal generation (the live loop checks `source != 'gap_filled'`)

---

### 4.4 `features/pipeline.py`

**Purpose:** Transform raw OHLCV bars into a feature matrix for the model. Stateless, deterministic, and look-ahead safe.

#### The critical rule: `shift(1)` everywhere

Every rolling computation uses `.shift(1)` before or after the rolling window. This means that at bar `t`, the feature value is computed using bars `t-1, t-2, ..., t-window`. The current bar's data is never used in its own feature computation.

Why does this matter? When you're live trading, the model must make a decision at the **close of bar `t`**. At that moment, bar `t` is complete — you know its OHLCV. The signal is then executed at the **open of bar `t+1`**. The features used to compute the signal at bar `t` should be computed from bars `t-1` and earlier. Using bar `t`'s data in a feature for bar `t`'s signal is fine — you have it. But every rolling calculation that computes a window ending at bar `t` is actually using bar `t`'s data, which you'd be "using" before acting on it.

The paranoid approach this system takes: treat bar `t`'s signal as if it were generated from bar `t-1`'s perspective. All features are shifted by one bar. This adds one bar of lag but makes look-ahead bugs impossible.

Example:
```python
# RSI at bar t uses prices through bar t-1
out["rsi"] = _rsi(df["close_bid"], self.config.rsi_period).shift(1)
```

#### Features computed

**Momentum:**
- `log_ret_1`: log return over 1 bar (close_bid[t-1] / close_bid[t-2]). Log returns are stationary (mean-reverting around zero) unlike raw price levels.
- `log_ret_5`: log return over 5 bars (~25 minutes)
- `log_ret_12`: log return over 12 bars (~1 hour)
- `rsi`: Wilder RSI(14). Bounded [0, 100], measures whether the market is overbought or oversold.
- `macd_hist`: MACD histogram (fast EMA - slow EMA - signal). Captures momentum divergence.

**Volatility:**
- `atr_pct`: ATR(14) divided by price. Normalizing by price makes this comparable across different price levels and across pairs.
- `bb_width`: Bollinger Band width (2σ / mid). High values = high volatility regime; low values = squeeze before a breakout.
- `realized_vol`: Rolling 20-bar standard deviation of log returns. Direct measure of recent volatility.

**Microstructure:**
- `spread_pips`: Current bid-ask spread in pips. Wide spread = high transaction cost; skip trades when spread is excessive.
- `spread_zscore`: How unusual is the current spread vs the last 100 bars? A Z-score of +3 means the spread is 3 standard deviations above recent norms — strong signal to avoid trading.
- `volume_zscore`: Similar Z-score for tick volume. High volume during a move suggests institutional participation; low volume during a move suggests noise.
- `ba_ratio`: Raw spread divided by mid price. Proportional bid-ask cost.

**Session (binary flags):**
- `session_london`: 1 during London session (07:00-16:00 UTC)
- `session_ny`: 1 during NY session (12:00-21:00 UTC)
- `session_overlap`: 1 during London/NY overlap (12:00-16:00 UTC). This is the highest-liquidity period.
- `session_asian`: 1 during Asian session (23:00-08:00 UTC). Typically low volatility.

**Calendar (cyclical encoding):**
- `dow_sin`, `dow_cos`: Day of week encoded as sine/cosine. Cyclical encoding ensures Monday and Friday are "close" in feature space, and there's no artificial discontinuity at the week boundary.
- `hour_sin`, `hour_cos`: Hour of day (0-24), cyclically encoded.
- `days_to_month_end`: How far through the current month we are. Month-end often has unusual flow from rebalancing.

#### Variance filtering in `fit()`

The `fit()` method computes all features on the training data, then drops any feature with variance below `min_variance_threshold`. Features with near-zero variance carry no information — they're nearly constant and won't help the model. This is the only thing `fit()` stores: the list of surviving feature column names.

Note that `fit()` does NOT store rolling statistics (means, standard deviations). The `transform()` recomputes everything from scratch. This is intentional — storing rolling statistics would mean the pipeline depends on the training data distribution at inference time, which would cause subtle bugs if the test distribution differs.

#### SHA-256 integrity on save/load

Every artifact (model or pipeline) is saved with a SHA-256 hash stored in a manifest file. On load, the hash is recomputed and compared. If they don't match — whether from disk corruption, accidental overwrite, or deliberate tampering — a `RuntimeError` is raised.

This matters because if you load a corrupted pipeline in production, all your features will be wrong and you'll trade on garbage signals. The integrity check catches this before any trades are placed.

---

### 4.5 `research/labels.py`

**Purpose:** Construct training labels using the triple barrier method. Run offline only.

#### Why triple barriers instead of simpler approaches

**Next-bar direction (predict whether next bar is up or down):** The hit rate on this label is ~50% in most FX data. The model has almost no signal to learn. Any apparent performance is noise.

**N-bar threshold return (predict whether price will be >X% higher after N bars):** Better than next-bar, but ignores the path. If the price hits your stop loss at bar 2 and recovers above your target at bar 10, this labeling scheme calls it a win even though you'd have been stopped out in real trading.

**Triple barrier (this system's choice):** For each bar, simulate both a long and short trade. The label is determined by which of three barriers is hit first: take profit, stop loss, or time. This matches real trading outcomes, including the path dependency of stop-loss triggers.

#### How the barriers work

For each bar `i` (the signal bar):

1. **LONG trade simulation:**
   - Enter at bar `i+1` open **ask** price (you pay the ask when buying)
   - Take profit barrier: entry + 1.5 × ATR
   - Stop loss barrier: entry - 1.0 × ATR
   - Time barrier: 12 bars maximum hold

2. **SHORT trade simulation:**
   - Enter at bar `i+1` open **bid** price (you receive the bid when selling)
   - Take profit barrier: entry - 1.5 × ATR (price needs to fall)
   - Stop loss barrier: entry + 1.0 × ATR

3. **Label assignment:**
   - If LONG hits TP and SHORT does not: label = LONG (+1)
   - If SHORT hits TP and LONG does not: label = SHORT (-1)
   - Otherwise (both hit SL, time barrier, or both hit TP): label = ABSTAIN (0)

The 1.5:1 reward-to-risk ratio (TP=1.5×ATR, SL=1.0×ATR) means a strategy that wins even 40% of directional trades (after ABSTAIN filtering) can be profitable.

#### Exclusion conditions

A bar is excluded from labeling (gets `NaN`) if:
- It's one of the first `atr_period` bars (ATR undefined)
- It's one of the last `max_holding_bars` bars (not enough look-forward window)
- It's a gap-filled bar
- The spread is >5 pips (fills would be unrealistic)
- ATR is < 3 pips (market too quiet, barriers degenerate)
- Any bar in the look-forward window is gap-filled (path is unreliable)

Excluded bars are not just unlabeled — they're set to `NaN` so the training pipeline (`slice_fold`) can drop them cleanly.

---

### 4.6 `research/walk_forward.py`

**Purpose:** Generate temporally ordered train/val/test splits that prevent all forms of data leakage.

#### Why walk-forward instead of random split

In time-series data, a random 80/20 split would include future data in the training set. Even if you split chronologically (first 80% train, last 20% test), you only get one test period. A single month or year of test performance might not represent the strategy's real long-run behavior, and you risk overfitting hyperparameters to that one test period.

Walk-forward validation gives you multiple independent out-of-sample test periods spread across different market regimes (trending, ranging, volatile, calm).

#### Fixed-size training window vs. expanding window

This system uses **fixed-size** (rolling) training windows: always 12 months of data, sliding forward by 1 month at each step. An expanding window would start with 12 months of data and grow to 24, 36 months, etc.

The argument for expanding window: more training data is usually better. The argument for fixed window: FX market regimes shift. A model trained on 2015-2017 data may not generalize to 2022 conditions. Stale regime data can actively hurt the model by pulling it toward outdated patterns.

#### The fold structure

```
[───── TRAIN (12 months) ─────][P][── VAL (2m) ──][E][── TEST (1m) ──]
                                ↑                   ↑
                             PURGE              EMBARGO
                            (12 bars)          (12 bars)
```

**PURGE zone:** The 12 bars between TRAIN end and VAL start. The labels at bar `t` look forward up to 12 bars (`max_holding_bars`). Without the purge, the label for bar `train_end - 1` would look at bars in the validation set, creating leakage.

**EMBARGO zone:** The 12 bars between VAL end and TEST start. The rolling features at bar `val_end` depend on bars up to 12 bars before it. Without the embargo, test features near the boundary would incorporate information from validation bars.

**Minimum 6 folds:** With the default configuration (1-month test steps), you need at least 3 years of data. The system raises `ValueError` if the data is too short. This prevents running the validation on an insufficient number of independent test periods.

#### Configuration defaults

```python
WalkForwardConfig(
    train_bars  = 105_120,   # ≈12 months × 252 trading days × 5 bars/hour × 6.5 hours
    purge_bars  = 12,        # = max_holding_bars in LabelConfig
    val_bars    = 17_520,    # ≈2 months
    embargo_bars = 12,
    test_bars   = 8_760,     # ≈1 month
    step_bars   = 8_760,     # advance by 1 test period per fold
    min_folds   = 6,
)
```

---

### 4.7 `research/model_training.py`

**Purpose:** Train, save, and load LightGBM/XGBoost classifiers with full versioning and integrity guarantees.

#### Why LightGBM/XGBoost instead of neural networks

Tree-based gradient boosting models are the dominant approach for tabular financial data for several reasons:

1. **They handle non-stationarity better.** Neural networks typically need careful normalization and are sensitive to distributional shift. GBM models are more robust to the fact that FX statistics change over time.

2. **Interpretability.** Feature importance from GBM is meaningful. You can audit whether the model is using sensible features.

3. **No gradient instability.** Deep learning optimization can diverge. GBM optimization is stable by construction.

4. **Faster training.** You can retrain monthly in minutes, not hours.

#### Three-class classification

The model predicts three classes: SHORT (-1), ABSTAIN (0), LONG (+1). This is not two classes (long vs short) because the ABSTAIN class carries explicit information: the model is saying "I have no edge here."

Label encoding: `{-1: 0, 0: 1, 1: 2}` (LightGBM requires non-negative integer classes).

#### Class weighting instead of SMOTE

If the ABSTAIN class is dominant (say 60% of labels), a naive model would learn to always predict ABSTAIN. The solution is to weight each sample inversely proportional to its class frequency:

```python
class_weight = {cls: n_total / (len(class_counts) * count) ...}
```

This is mathematically equivalent to the classes having equal representation. **SMOTE (synthetic minority oversampling) is explicitly rejected** because it generates synthetic samples by interpolating between existing samples. For time-series data, this creates fictional bars that never happened and breaks the temporal ordering.

#### Artifact structure

Each model save creates a directory:
```
artifacts/
└── eurusd_5m_v001_20240115/
    ├── model.pkl                  ← pickled model object
    ├── model.manifest.json        ← SHA-256, metadata, deployment_approved flag
    ├── experiment_config.json     ← exact config used for training (reproducibility)
    └── validation_report.json     ← per-fold metrics, deployment_approved (set manually)
```

**`deployment_approved` must be set to `true` manually.** The `load_model()` function raises `RuntimeError` if you try to load a model for live deployment without this flag. This enforces a human review gate before any model goes live.

#### `compute_signal()` — the confidence thresholds

```python
compute_signal(proba, min_confidence=0.55, min_margin=0.10)
```

Three ABSTAIN conditions:
1. `max(proba) < 0.55` — the model isn't confident about any class
2. The abstain class has the highest probability
3. `max(proba) - second_max(proba) < 0.10` — the top two classes are too close (ambiguous)

The purpose of condition 3: if the model says LONG=0.51, ABSTAIN=0.42, SHORT=0.07, that's technically a LONG signal but the margin over ABSTAIN is only 0.09. This is too close to call. With min_margin=0.10, this abstains.

---

### 4.8 `risk/engine.py`

**Purpose:** Evaluate every potential trade against a set of circuit breakers and compute position size. Last line of defense before an order touches the broker.

#### The circuit breakers — in evaluation order

The breakers are checked in this exact sequence. If any one triggers, `RiskDecision(trade_permitted=False)` is returned immediately.

**1. Kill switch active** — A global halt flag stored in SQLite. If this is True, no trades under any circumstances. The kill switch is set by: the drawdown circuit breaker, manual operator action, or the heartbeat monitor.

**2. Signal is ABSTAIN (direction=0)** — Explict gate: even if a signal reaches the risk engine with direction=0, this catches it. Defensive programming.

**3. Daily drawdown ≥ 3%** — If today's P&L is a loss of 3% or more of equity, stop trading for the rest of the day. Calculated as `(-daily_pnl_usd) / equity_usd`.

**4. Max open positions** — Maximum 3 positions open simultaneously. FX correlation means that holding EURUSD long, GBPUSD long, and USDJPY short is essentially a levered USD bet. Limiting concurrent positions caps this.

**5. Consecutive loss cooldown** — After 4 consecutive losing trades, pause for 12 bars (~1 hour for 5m bars). This prevents compounding losses during a regime where the model has temporarily lost its edge.

**6. ATR too small** — If ATR < 5 pips, the market is too flat. The TP and SL barriers will be placed within the spread, and the trade has no room to work. Skip it.

**7. Spread/ATR ratio too high** — If the spread is > 25% of the ATR, transaction costs dominate. A trade that needs to move 10 pips to hit TP but costs 2.5 pips in spread has a significant negative expected value from just the entry and exit costs.

**8. Correlated exposure cap** — If EURUSD and GBPUSD long positions together exceed $10,000 USD notional, don't add more. Both pairs have high positive correlation to each other (both are effectively short USD). This caps your aggregate USD exposure.

#### Position sizing formula

```
risk_usd = equity × 1% × confidence_scalar

confidence_scalar = max(0.55, min(1.0, confidence))
  → At 55% confidence: size = 55% of normal
  → At 100% confidence: size = 100% of normal

stop_distance_pips = (ATR × 1.0) + (spread × 0.5)
  → The ATR component is the "signal distance"
  → The spread buffer prevents the stop from triggering on normal spread noise

pip_value_usd = value of one pip for one unit of the base currency
  → EUR/USD: $0.0001 per unit
  → USD/JPY: $0.01 / price per unit

units = risk_usd / (stop_distance_pips × pip_value_usd)
units = clamp(units, min=1000, max=100000)
```

Example at $10,000 equity, 75% confidence, 10-pip ATR, 1-pip spread on EUR/USD:

```
risk_usd = 10,000 × 0.01 × 0.75 = $75
stop_distance_pips = 10 + 0.5 = 10.5 pips
pip_value_usd = $0.0001
units = 75 / (10.5 × 0.0001) = 75 / 0.00105 = 71,428 units
```

That's 71,428 units with a 10.5-pip stop and $75 max loss. At 1 standard lot = 100,000 units, that's 0.71 standard lots.

#### Stop placement

```python
if signal == 1:   # LONG: enter at ask, stop below bid
    entry_price = current_ask
    stop_price = entry_price - stop_distance_price
else:             # SHORT: enter at bid, stop above ask
    entry_price = current_bid
    stop_price = entry_price + stop_distance_price
```

For a LONG trade, entry is at the ask (we pay the ask). The stop is placed below the bid (not below the ask) — this means even a slight adverse move triggers the stop. Combined with the spread buffer in stop_distance_pips, the stop is protected against normal spread-noise from triggering it.

---

### 4.9 `state/store.py`

**Purpose:** Crash-safe SQLite state persistence. The system must survive `kill -9` (abrupt process termination) without losing any order or position state.

#### SQLite WAL mode

The database runs with:
```sql
PRAGMA journal_mode=WAL;     -- Write-Ahead Logging
PRAGMA synchronous=FULL;     -- fsync on every commit
PRAGMA foreign_keys=ON;
```

**WAL mode** means writes are first written to a write-ahead log before being applied to the main database file. If the process is killed mid-write, the log is complete and the database can be reconstructed on next open. Without WAL, a partial write can leave the database in an inconsistent state.

**synchronous=FULL** means SQLite calls fsync after every commit, ensuring the data is physically written to disk (not just in the OS buffer cache) before the commit returns. Without this, a power failure after our code thinks the write succeeded could lose data.

#### What's stored

**`system_state` table (key-value):**
- `kill_switch`: "true"/"false" — the global halt flag
- `high_water_mark`: peak equity ever recorded
- `daily_pnl_{date}`: rolling daily P&L per date
- `consecutive_losses`: current losing streak count
- `bars_since_loss_limit`: bar count since consecutive loss limit was hit
- `deployed_model_sha256`: SHA-256 of the currently deployed model
- `deployed_pipeline_sha256`: SHA-256 of the currently deployed pipeline
- `last_heartbeat_utc`: timestamp of last successful heartbeat

**`orders` table:** Full order lifecycle with client_order_id as primary key. Upsert on conflict.

**`positions` table:** All open and closed positions with entry metadata.

**`trade_log` table:** Immutable closed-trade records with P&L, model version, and signal confidence.

#### The `fsync()` call

In `IdempotentOrderSubmitter.submit()`, after writing the order with `PENDING_SUBMIT` status, we explicitly call `state_store.fsync()`. This is the `PRAGMA synchronous=FULL` in action — it ensures the order record is physically on disk before we attempt the network call to the broker. If the process crashes between the DB write and the broker call, we have a record of the attempt and can flag it for human review during reconciliation.

---

### 4.10 `broker/adapter.py`

**Purpose:** Abstract interface over all brokers, plus the `IdempotentOrderSubmitter`.

#### The BrokerAdapter abstract class

Every broker interaction goes through exactly 8 abstract methods:

| Method | Description |
|---|---|
| `get_open_positions()` | Fetch live positions from broker. Used in reconciliation. |
| `get_pending_orders()` | Fetch unfilled orders. Used in reconciliation. |
| `send_order(order)` | Submit one order. Returns `OrderResult`. |
| `cancel_order(order_id)` | Cancel a pending order by broker ID. |
| `get_account_state()` | Fetch equity, balance, margin. Max staleness 60s. |
| `get_price(pair)` | Get current (bid, ask). Never cache. |
| `modify_stop_loss(position_id, new_sl)` | Modify SL on open position. |
| `close_position(position_id, units)` | Close or partially close a position. |

The reason for this strict interface: the execution loop (`live_loop.py`) and reconciliation code can be tested with a mock adapter. You don't need a live MT5 connection to test whether the reconciliation logic handles orphaned positions correctly.

#### `IdempotentOrderSubmitter` — why it exists

Network calls are unreliable. The sequence "submit order → get confirmation" can fail at multiple points:

- The order was written to our DB but the network call timed out. Did the broker receive it or not?
- The broker received and filled the order, but the confirmation got lost on the way back to us.
- The process was killed at exactly the wrong moment.

Without idempotency, retrying after a failure could submit the same order twice — doubling your position size unexpectedly.

The idempotency key is the `client_order_id` (UUID4). Before every submission:

1. Check if this `client_order_id` already exists in the DB with a non-terminal status
2. If yes: don't submit again, return the existing result
3. If no: proceed with submission

The `client_order_id` is generated once when the `Order` object is created. It travels with the order through the entire lifecycle and is stored in both our DB and (as a comment field) in the broker's record.

#### Retry logic

On transient failures (network timeout, connection error), the submitter retries with exponential backoff: 1s, 2s, 4s. After 3 failed attempts, the order is set to `DEAD_LETTERED` and a `CRITICAL` log entry is written.

Broker rejections (`REJECTED` status) are NOT retried. If the broker explicitly refused the order (wrong symbol, insufficient margin, market closed), retrying will just generate more rejections.

---

### 4.11 `broker/mt5_adapter.py`

**Purpose:** Concrete implementation of `BrokerAdapter` for MetaTrader 5.

This file contains the only code in the entire system that imports `MetaTrader5`. If you wanted to add OANDA support, you'd create `broker/oanda_adapter.py` and implement the same 8 abstract methods — you'd never touch the execution loop.

Key implementation details:
- All MT5 timestamps are passed through `mt5_server_to_utc()` before being stored or returned
- Order tickets in MT5 are broker_order_ids
- Position volumes in MT5 are in lots; this adapter converts to units (lots × 100,000)
- The `is_connected()` method uses `mt5.account_info()` as a live ping

---

### 4.12 `execution/reconciliation.py`

**Purpose:** The startup procedure that runs every time the process starts, regardless of whether it crashed or was gracefully restarted.

**Never start trading without completing reconciliation.**

#### The 7 reconciliation steps

**Step 1 — Kill switch check:** If the kill switch is active in the database, halt immediately. Do not attempt to connect to the broker. The operator must manually clear the kill switch after reviewing what caused it to be set.

**Step 2 — Broker connection:** Verify we can authenticate and fetch account state. If this fails, trading cannot proceed.

**Step 3 — Position reconciliation:**

| Scenario | Action |
|---|---|
| In broker, not in local DB | Write as ORPHANED. Warn operator. Never auto-close. |
| In local DB, not in broker | Mark as BROKER_CLOSED. Warn that P&L needs verification. |
| In both | Compare units and entry price. Broker wins. Update local. |

The rule "broker wins on existence, local wins on lineage" means: if the broker says a position exists, we accept it even if we didn't create it. But we keep our `client_order_id` linking an order to a position — that lineage came from us.

**Step 4 — Order reconciliation:**

Any order found in `PENDING_SUBMIT` status is marked `UNCONFIRMED`. This means: we wrote the order to our DB and attempted to send it, but then crashed. We don't know if the broker received it. **Never auto-resubmit.** Mark it for human review.

Any `SUBMITTED` order not found in the broker's pending order list is also marked `UNCONFIRMED` — it may have filled or been cancelled while we were down.

**Step 5 — P&L verification:** Cross-check our local daily P&L tracking against the broker's account balance. On large discrepancies, log a warning. The actual P&L tracking uses our own trade log, but the broker balance is the ground truth.

**Step 6 — Artifact integrity:** Verify that the SHA-256 of the currently loaded model and pipeline match what's stored in the database from the last deployment. If they don't match, refuse to trade — someone may have modified the model files.

**Step 7 — Finalize:** Log reconciliation summary, update high-water mark, write heartbeat timestamp.

After reconciliation, check `result.requires_human_review`. If True, do not proceed. Alert the operator and wait for clearance.

---

### 4.13 `execution/live_loop.py`

**Purpose:** The main loop that runs continuously during market hours, generating signals and submitting orders.

#### Loop structure (simplified)

```
while True:
    1. Wait for next completed bar
    2. If market is closed → sleep until market open
    3. Check heartbeat → if unhealthy, halt
    4. Get current account state
    5. Run drawdown circuit breaker check
    6. Get latest complete bar
    7. Run feature pipeline on recent bars (with lookback)
    8. Run model inference → get probabilities
    9. Compute signal with confidence thresholds
    10. If signal == ABSTAIN → log and continue
    11. Run risk engine → get RiskDecision
    12. If trade_permitted=False → log reason and continue
    13. Build Order object with client_order_id
    14. Submit via IdempotentOrderSubmitter
    15. If filled → create Position in state store
```

#### Stale data detection

```python
STALE_TICK_THRESHOLD_SECONDS = 30
BAR_COMPLETION_TIMEOUT_SECONDS = 330
```

If no new tick has arrived in 30 seconds during market hours, something is wrong with the data feed. The loop logs a warning and skips signal generation for that bar.

If 5.5 minutes have passed with no new tick to "close" the current bar, the bar is forced to complete. This handles the case of very illiquid periods where the next bar might not start for minutes.

#### Heartbeat monitoring

Every 30 seconds, the heartbeat check verifies:
1. Last tick arrived < 30 seconds ago
2. `get_account_state()` completes within 5 seconds
3. Local position count matches broker position count
4. The database can be written to
5. Loaded model SHA-256 matches the deployed version

After 3 consecutive heartbeat failures: CRITICAL alert, set kill switch. This catches scenarios like the broker disconnecting silently, the data feed going down, or disk filling up.

#### Reconnect logic

On connection loss:
1. Mark status as DISCONNECTED, stop placing new orders
2. Retry with exponential backoff (1s → 2s → 4s → 8s → ... → max 60s)
3. On successful reconnect: re-run full reconciliation (check if SL was triggered during downtime)
4. Only resume after clean reconciliation

---

### 4.14 `tests/test_all.py`

**Purpose:** 48 automated tests covering every critical safety property in the system.

Run with:
```bash
cd fx_trading/
python -m unittest tests.test_all -v
```

No external dependencies — uses only Python's stdlib `unittest` module.

#### What the tests verify (by category)

**Schemas (9 tests):**
- `assert_utc` raises on naive datetimes
- `assert_utc` raises on non-UTC aware datetimes (e.g., UTC+2)
- `assert_utc` passes on correct UTC datetimes
- Pip size and multiplier are correct for regular and JPY pairs
- Every Order gets a unique UUID client_order_id
- OHLCVBar is immutable (frozen dataclass)
- Tick rejects negative bid prices
- Tick rejects ask < bid

**Time utilities (8 tests):**
- MT5 broker time converts correctly to UTC
- MT5 converter raises if given an already-aware datetime
- OANDA RFC3339 string parses correctly
- Market is closed on Saturday
- Market is closed Sunday before 21:00 UTC
- Market is open Sunday after 21:00 UTC
- Market is closed Friday after 21:00 UTC
- `floor_to_bar()` aligns to 5-minute boundaries
- `now_utc()` returns tz-aware datetime

**Feature pipeline (5 tests):**
- No NaN in features after warmup period
- Modifying bar `t` does not change features at bar `t-1` (look-ahead proof)
- Pipeline survives save/load with identical output
- Tampered artifact raises RuntimeError
- Unfitted pipeline raises RuntimeError on transform

**Walk-forward (5 tests):**
- No overlap between train, val, and test splits
- Purge zone is at least `purge_bars` in size
- Embargo zone is at least `embargo_bars` in size
- Insufficient data raises ValueError
- Folds are chronologically ordered

**Risk engine (7 tests):**
- Kill switch blocks all trades
- ABSTAIN signal is denied
- 3% daily drawdown triggers circuit breaker
- Position size stays within 1% equity risk limit
- Low ATR blocks trade
- Consecutive loss cooldown activates
- LONG stop price is placed below bid

**State store (6 tests):**
- Kill switch persists across simulated process restart
- Upserting same order twice doesn't create duplicates
- Daily P&L resets to 0 on a new date
- High-water mark only increases (drawdown doesn't lower it)
- Drawdown calculation is accurate to 3 decimal places
- Position round-trips through DB without data loss

**Model signals (5 tests):**
- Low confidence produces ABSTAIN
- Dominant ABSTAIN class produces ABSTAIN signal
- Clear LONG probability produces LONG signal
- Clear SHORT probability produces SHORT signal
- Ambiguous top-two produces ABSTAIN

---

## 5. The ML Pipeline End-to-End

This section traces the complete path from raw tick data to a trained, deployable model.

### Step 1: Collect historical data

Fetch 3+ years of 5-minute OHLCV bars from MT5. This gives you ≈262,800 bars. Convert all timestamps to UTC via `mt5_server_to_utc()`.

```python
from data.aggregation import bars_to_dataframe
from data.time_utils import mt5_server_to_utc

# In your MT5 data fetch script:
bars = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, date_from, date_to)
df = bars_to_dataframe(convert_mt5_bars(bars, broker_tz="Etc/GMT-2"))
```

### Step 2: Audit the data

```python
from data.aggregation import detect_gaps
gaps = detect_gaps(df, timeframe_sec=300)
# Review gaps longer than 3 bars during session hours
```

Gaps in the training data degrade label quality (path is unreliable) and can introduce look-ahead bias if gap-filled bars cross split boundaries.

### Step 3: Construct labels

```python
from research.labels import construct_labels, LabelConfig, get_label_distribution

config = LabelConfig(tp_atr_multiplier=1.5, sl_atr_multiplier=1.0, max_holding_bars=12)
labels = construct_labels(df, config, pair_is_jpy=False)

dist = get_label_distribution(labels)
print(dist)
# Check: is_balanced should be True, abstain_pct should be 30-60%
```

Healthy label distribution: LONG 20-30%, SHORT 20-30%, ABSTAIN 40-60%. If LONG or SHORT is <15%, the labeling config may be too restrictive. If ABSTAIN is >80%, the market is very noisy in this period.

### Step 4: Fit the feature pipeline

```python
from features.pipeline import FeaturePipeline, FeaturePipelineConfig

pipeline_config = FeaturePipelineConfig(
    version="1.0.0",
    rolling_window_bars=100,
    atr_period=14,
    rsi_period=14,
    min_variance_threshold=1e-8,
)
pipeline = FeaturePipeline(pipeline_config)
pipeline.fit(df)  # Fit only on training data!
features = pipeline.transform(df)
```

### Step 5: Generate walk-forward folds

```python
from research.walk_forward import generate_folds, WalkForwardConfig, slice_fold

wf_config = WalkForwardConfig(
    train_bars=105_120, purge_bars=12, val_bars=17_520,
    embargo_bars=12, test_bars=8_760, step_bars=8_760, min_folds=6
)
folds = generate_folds(len(df), wf_config)

# Describe folds:
for fold in folds:
    print(fold.describe(df.index))
```

### Step 6: Train and evaluate per fold

```python
from research.model_training import (
    ModelConfig, ValidationReport, train_model, predict_proba,
    compute_signal, save_model
)

model_config = ModelConfig(
    version="eurusd_5m_v001",
    pair="EUR/USD",
    timeframe="5m",
    model_class="lightgbm",
    model_params={
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 50,
    }
)

fold_results = []
for fold in folds:
    X_train, y_train, X_val, y_val, X_test, y_test = slice_fold(features, labels, fold)
    model = train_model(X_train, y_train, X_val, y_val, model_config)
    
    proba = predict_proba(model, X_test, model_config.model_class)
    # Evaluate signal generation hit rate, Sharpe, etc. on test set
    fold_results.append(evaluate_fold(proba, y_test))  # your evaluation function

# Create validation report
report = ValidationReport(
    fold_results=fold_results,
    aggregate=compute_aggregate_metrics(fold_results),
    deployment_approved=False,  # human sets this to True after review
)
```

### Step 7: Human review and approval

Review the `validation_report.json`. Check:
- Out-of-sample Sharpe across folds: acceptable is 0.6-1.0, suspicious is >1.5
- Maximum drawdown per fold: should be <15%
- ABSTAIN rate: should match validation ABSTAIN rate ±10%
- No single fold is responsible for all the performance (regime dependency)

If acceptable, manually edit `validation_report.json` to set `deployment_approved: true`.

### Step 8: Save model

```python
model_config.train_end_date = str(df.index[-1].date())
artifact_dir, sha256 = save_model(model, model_config, report, base_path="artifacts/")
pipeline.save(f"artifacts/pipeline_{model_config.version}/")
```

The model is now ready for deployment.

---

## 6. The Signal-to-Order Path

This traces what happens live for each completed bar during market hours:

```
1. Bar completes (is_complete=True on next tick arrival)
   ↓
2. pipeline.transform(recent_bars_with_lookback)
   → Returns feature vector for the completed bar
   ↓
3. model.predict_proba(features)
   → [p_short, p_abstain, p_long]
   ↓
4. compute_signal(proba, min_confidence=0.55, min_margin=0.10)
   → (direction: int, confidence: float)
   ↓ (If direction == 0 → ABSTAIN, skip rest)
5. adapter.get_price(pair)
   → current (bid, ask)
   ↓
6. adapter.get_account_state()
   → equity_usd
   ↓
7. risk_engine.evaluate(signal, confidence, pair, bid, ask, atr_pips, equity, positions, daily_pnl)
   → RiskDecision
   ↓ (If trade_permitted=False → log reason, skip rest)
8. Build Order(pair, side, units, stop_loss_price, client_order_id=uuid4())
   ↓
9. IdempotentOrderSubmitter.submit(order)
   a. Write PENDING_SUBMIT to SQLite (fsync)
   b. Send to broker via adapter.send_order()
   c. Update order status in SQLite
   ↓ (If FILLED)
10. Create Position in state store
    ↓
11. Log trade to trade_log
```

The entire path from bar completion to order submission typically takes 50-200ms depending on network latency to the broker. For 5-minute bars, this is negligible.

---

## 7. Crash Safety and Recovery

### What happens if the process is killed at each point:

**Killed before writing PENDING_SUBMIT:** No DB record. On restart, reconciliation finds no record for recent bars. This is clean — the order was never sent.

**Killed after writing PENDING_SUBMIT but before sending:** On restart, `_reconcile_orders()` finds a `PENDING_SUBMIT` order. Marks it `UNCONFIRMED`. Operator reviews: was this sent or not? Check broker order history.

**Killed after sending but before receiving confirmation:** The order may or may not have been received by the broker. On restart, the `SUBMITTED` order is not in broker pending orders (either filled or never received). Marked `UNCONFIRMED`. Check broker trade history.

**Killed with open position:** On restart, broker has the position. Local DB also has it (if FILLED was written). Reconciliation sees both and verifies they match.

**Killed with open position, local DB corrupted:** SQLite WAL mode means the DB cannot be corrupted mid-write. The WAL is replayed on next open.

### The kill switch

The kill switch is a DB record (`system_state` where `key='kill_switch'` and `value='true'`). It is:

- Set by the drawdown circuit breaker (3% daily drawdown)
- Set by the heartbeat monitor (10 consecutive failed checks)
- Set manually by the operator

It is cleared **only manually**. After any automatic kill switch activation, a human must review the logs, understand what caused it, and decide it's safe to resume.

---

## 8. Look-Ahead Bias — What It Is and How We Prevent It

Look-ahead bias is when your model is trained on information that would not have been available at the time the prediction was needed. It makes backtests look far better than live performance.

### The 5 places it can enter this system

**1. Rolling features not shifted:** If `rsi[t]` uses prices through bar `t`, and the signal is generated at bar `t` to trade at bar `t+1`, you're fine — bar `t` is complete. But our convention is to use `shift(1)` anyway, making all features one bar lagged, which is the safest possible choice.

**Test:** The `test_no_lookahead_modification_test` unit test modifies bar `t`'s price to an absurd value (999.0) and verifies that features at bar `t-1` don't change. If they do, look-ahead exists.

**2. Labels looking into the future:** The triple barrier label for bar `t` looks at bars `t+1` through `t+12`. This is by design — the label represents what would have happened if you entered a trade at bar `t+1`. The purge zone in walk-forward (12 bars) ensures that none of these future bars are in the training set.

**Test:** The `test_purge_zone_is_sufficient` unit test verifies the purge zone size.

**3. Pipeline fit on test data:** Calling `pipeline.fit(df)` on all data (train + test combined) stores statistics from the test period in the pipeline. When the test-period features are then transformed, they use these "future" statistics.

**Prevention:** The `fit()` is called only on `df.iloc[fold.train_start:fold.train_end]`, never on val or test data.

**4. Train/val/test overlap:** Any index present in both train and test means the model saw those exact rows during training. For time series, this means future data trained the model.

**Test:** The `test_no_overlap_between_splits` unit test verifies this explicitly with set intersection.

**5. Signals using current bar's close:** If the live loop generates a signal using bar `t`'s features and immediately executes at bar `t`'s close price (instead of bar `t+1`'s open), you're trading on the current bar's close — which you don't know until after it closes. The live loop enforces execution at the **next bar's open**.

**Prevention:** The `bars_to_dataframe()` output is used as input to the feature pipeline. Bar `t` is only used after `is_complete=True`. The signal at bar `t` is executed at bar `t+1`'s open.

### The suspicious Sharpe test

If your out-of-sample Sharpe across walk-forward folds is >2.0, you almost certainly have a look-ahead bug somewhere. A genuinely predictive signal in FX produces Sharpe of 0.6-1.5 at best after costs. If you're seeing 3.0+, run the diagnostic checklist:

1. Re-run backtesting with fills at next-bar open ask (not mid)
2. Re-run with 2× spread
3. Test on a completely different year
4. Check whether the equity curve is suspiciously smooth (real trading has drawdowns)
5. Verify every rolling calculation uses `shift(1)`
6. Compare vectorized backtest Sharpe vs event-driven backtest Sharpe (should be similar; large divergence means the vectorized test is optimistic)

---

## 9. Risk Management

### Position-level controls

Every position has a hard stop loss set at entry:
```
stop_distance = ATR × 1.0 + spread × 0.5
```

The stop is placed as a server-side stop with the broker (not just a local alert). Even if our process crashes, the broker will execute the stop.

### Account-level controls

| Control | Threshold | Action |
|---|---|---|
| Daily drawdown | 3% equity | Halt all trading for the day |
| Weekly drawdown | 7% equity | Halt, require manual review |
| Monthly drawdown | 12% equity | Halt, retrain required |
| Consecutive losses | 4 in a row | 12-bar cooldown (~1 hour) |
| Live Sharpe vs backtest | <30% of backtest | Model has failed |
| Mean slippage | >2 pips over 20 trades | Broker/execution issue |

### The capital ramp-up protocol

Never go from backtesting to full capital. The sequence:

| Stage | Duration | Capital | Review |
|---|---|---|---|
| Demo forward test | 8 weeks | 0% (fake) | Pass/fail criteria |
| Stage 1 live | 4 weeks | 10% | Daily review |
| Stage 2 live | 6 weeks | 25% | Weekly review |
| Stage 3 live | 8 weeks | 50% | Biweekly review |
| Stage 4 live | Ongoing | 100% | Monthly review |

Demo pass criteria:
- ≥40 trades completed
- Sharpe ≥ 0.5
- Max drawdown ≤ 10%
- Zero reconciliation errors
- ABSTAIN rate matches validation ±10%
- Tested against: news event (NFP/FOMC), gap open, manual disconnection

---

## 10. Deployment Guide

### Prerequisites

- Python 3.12+
- MetaTrader 5 terminal installed (for MT5 adapter)
- 3+ years of 5-minute OHLCV data

### Install dependencies

```bash
pip install pandas numpy scikit-learn lightgbm xgboost statsmodels structlog
# MT5 only installs on Windows:
pip install MetaTrader5
```

### Run tests first

```bash
cd fx_trading/
python -m unittest tests.test_all -v
# Expected: 48 tests, 0 errors, 0 failures
```

### Required configuration (fill these in before running)

1. **Broker timezone** — Open `data/time_utils.py`, find `BROKER_TIMEZONES`. Add your broker.
2. **MT5 login credentials** — In your startup script, pass `login`, `password`, `server` to `MT5Adapter.__init__`.
3. **Risk parameters** — Review `RiskParameters` defaults in `risk/engine.py`. Especially `risk_per_trade_pct` and `max_daily_drawdown_pct`.
4. **Model artifact path** — The live loop must be pointed at a trained, approved model.

### Startup sequence

```python
import sys
sys.path.insert(0, "/path/to/fx_trading")

from state.store import StateStore
from broker.mt5_adapter import MT5Adapter
from features.pipeline import FeaturePipeline
from research.model_training import load_model
from risk.engine import RiskEngine, RiskParameters
from execution.reconciliation import run_reconciliation
from execution.live_loop import LiveExecutionLoop

# 1. Initialize state
store = StateStore("state/trading.db")

# 2. Connect broker
adapter = MT5Adapter(login=YOUR_LOGIN, password=YOUR_PASSWORD, server=YOUR_SERVER)

# 3. Load artifacts (raises if not approved or hash mismatch)
model, manifest = load_model("artifacts/eurusd_5m_v001_20240115/", require_approved=True)
pipeline = FeaturePipeline.load("artifacts/pipeline_eurusd_5m_v001/")

# 4. Run reconciliation (MUST succeed before trading)
result = run_reconciliation(
    adapter=adapter,
    state_store=store,
    model_sha256=manifest["sha256"],
    pipeline_sha256=pipeline.save("/tmp/pipeline_check"),  # get current hash
)

if result.requires_human_review:
    print("HUMAN REVIEW REQUIRED:")
    for w in result.warnings: print(f"  WARNING: {w}")
    for e in result.errors: print(f"  ERROR: {e}")
    sys.exit(1)

# 5. Start live loop
risk_engine = RiskEngine(RiskParameters(), store)
loop = LiveExecutionLoop(adapter, store, pipeline, model, risk_engine, manifest["version"])
loop.run()
```

---

## 11. Common Pitfalls and How to Avoid Them

### "My backtest Sharpe is 2.5 but live performance is flat"

Almost certainly look-ahead bias. Run the 5-point diagnostic in section 8.

### "The reconciliation keeps finding UNCONFIRMED orders"

The order submission is crashing before it can update the order status from `PENDING_SUBMIT`. Check: is `fsync()` implemented in your `StateStore`? Is the broker API timing out? Check the logs for exceptions in `IdempotentOrderSubmitter._send_with_retry`.

### "I'm getting NaN features after the warmup period"

Pass more bars into `pipeline.transform()`. The warmup period is `rolling_window_bars` (default 100). The feature computation requires `rolling_window_bars` rows of context before the first valid feature. If you pass exactly 100 bars, you'll get mostly NaN. Pass at least 200 to get meaningful features.

### "The model is predicting ABSTAIN 90% of the time"

This is not necessarily bad — the model is being appropriately conservative. But if it's too high:
- Check if your `min_confidence` threshold is too high (try 0.50 instead of 0.55)
- Check if the test-period market regime is very different from the training period
- Check the label distribution — if ABSTAIN is 70%+ in the training labels, the model learned to abstain

### "The daily drawdown circuit breaker keeps triggering"

Either the model is genuinely performing badly (retrain needed), or your `max_daily_drawdown_pct` is too tight for the volatility regime. Look at the trade log: are losses coming from stop-loss triggers (stop is too tight) or from position sizing (too large)? Adjust the relevant parameter.

### "Reconciliation says positions have a unit mismatch"

This can happen if you manually partially-closed a position in the broker terminal while the system was running. The broker position has fewer units than our local record. Reconciliation detects this, takes the broker's value, logs a warning. Review whether this should be permanent or if you need to update the local stop/TP levels to match.

### "The timestamp tests are failing after I added a new data source"

Every datetime entering the system must be UTC-aware. Check your ingestion code. The `assert_utc()` function will tell you exactly where a naive datetime is coming from. Add `assert_utc(dt, "your_context")` at every external data boundary.

---

## Running the Test Suite

```bash
cd fx_trading/
python -m unittest tests.test_all -v
```

Expected output: 48 tests, OK.

If any test fails, **do not proceed**. The failing test is protecting you from a specific class of bug that can cause real financial losses.

---

*This document covers the full system as of March 2026. The implementation order is: schemas → time_utils → aggregation → pipeline → labels → walk_forward → model_training → risk engine → state store → broker adapter → execution. Each module must have its tests passing before the next is built.*
