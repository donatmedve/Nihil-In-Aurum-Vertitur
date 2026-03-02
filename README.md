# FX Trading System — Complete Documentation

This is a production-grade, ML-driven algorithmic trading system for foreign exchange (FX) markets. It trades EUR/USD, GBP/USD, and USD/JPY on 5-minute bars using a LightGBM or XGBoost classifier trained with a walk-forward framework.

Read this document top to bottom before touching any code. Bugs here cost real money.

---

## Table of Contents

1. [Architecture Philosophy](#1-architecture-philosophy)
2. [Project Structure](#2-project-structure)
3. [Dependency Rules](#3-dependency-rules)
4. [Module Reference](#4-module-reference)
5. [The ML Pipeline End-to-End](#5-the-ml-pipeline-end-to-end)
6. [The Signal-to-Order Path](#6-the-signal-to-order-path)
7. [Take-Profit System](#7-take-profit-system)
8. [Crash Safety and Recovery](#8-crash-safety-and-recovery)
9. [Look-Ahead Bias — What It Is and How We Prevent It](#9-look-ahead-bias)
10. [Risk Management](#10-risk-management)
11. [Deployment Guide](#11-deployment-guide)
12. [Free Hosting Options](#12-free-hosting-options)
13. [How the System Connects to the Market](#13-how-the-system-connects-to-the-market)
14. [Performance Tuning (Including Aggressive Options)](#14-performance-tuning)
15. [Common Pitfalls](#15-common-pitfalls)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Architecture Philosophy

### The core problem

Algorithmic trading has three distinct environments that must never bleed into each other:

1. **Research** — offline, works with historical data, can take hours to run, no latency constraints. This is where you train models and test hypotheses.
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
│   └── schemas.py              ← All typed data contracts. Import from here everywhere.
│
├── data/
│   ├── time_utils.py           ← UTC enforcement, broker time conversion, session helpers
│   ├── aggregation.py          ← Tick-to-bar aggregation with gap-fill rules
│   └── ingestion.py            ← MT5 bar fetching (fetch_recent_bars_mt5)
│
├── features/
│   └── pipeline.py             ← Feature computation. Stateless. Look-ahead safe.
│
├── research/
│   ├── labels.py               ← Triple barrier label construction (offline only)
│   ├── walk_forward.py         ← Rolling walk-forward fold generation
│   └── model_training.py       ← Train, save, load models with SHA-256 integrity
│
├── risk/
│   └── engine.py               ← Circuit breakers, position sizing, take-profit computation
│
├── state/
│   └── store.py                ← SQLite WAL persistence. Survives kill -9.
│
├── broker/
│   ├── adapter.py              ← Abstract broker interface + idempotent submitter
│   └── mt5_adapter.py          ← MetaTrader 5 concrete implementation
│
├── execution/
│   ├── reconciliation.py       ← Crash recovery: 7-step broker/local reconciliation
│   └── live_loop.py            ← Main execution loop (fully implemented, no stubs)
│
├── backtesting/                ← (placeholder for event-driven backtester)
├── monitoring/                 ← (placeholder for alerting/heartbeat)
├── tests/
│   └── test_all.py             ← 48 tests covering every critical safety property
│
├── artifacts/                  ← Model and pipeline artifacts stored here
├── logs/                       ← JSONL log files
├── start.py                    ← COPY-PASTE READY startup script
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

What is never allowed:
- `research/` importing from `execution/` or `broker/`
- `risk/` importing model logic
- `broker/` importing feature or model logic
- Circular imports of any kind

---

## 4. Module Reference

### 4.1 `shared/schemas.py`

Canonical data contracts. The single source of truth for every type used across the system. Key types: `OHLCVBar`, `Tick`, `Order`, `Position`, `RiskDecision`, `TradeRecord`.

Key design decisions:
- **Frozen dataclasses for market data** — immutable once created. Prevents silent mutation bugs.
- **`Decimal` for prices, never `float`** — floating-point arithmetic accumulates rounding errors in P&L calculations.
- **UTC enforcement on every timestamp** — `__post_init__` validators reject naive datetimes at construction time.

`RiskDecision` now includes `take_profit_price` — the TP is computed by the risk engine at the same time as the stop loss, using the same ATR, ensuring the TP:SL ratio always matches what the model was trained to predict.

### 4.2 `data/time_utils.py`

UTC enforcement and broker timezone conversion. The only place `mt5_server_to_utc()` is implemented. Every timestamp that comes from an external system (MT5, OANDA, Reuters) must pass through this module before entering the rest of the system.

### 4.3 `data/aggregation.py`

Deterministic tick-to-bar aggregation with gap-filling for short runs within market hours. Gaps longer than 3 bars or outside session hours are left as-is.

### 4.4 `data/ingestion.py`

**NEW** — Concrete MT5 bar fetching. This is the implementation of what was previously a `NotImplementedError` stub in the live loop.

Key function: `fetch_recent_bars_mt5(pair, n, timeframe_mt5, timeframe_sec, broker_tz_str)`.

How it works:
- Calls `mt5.copy_rates_from_pos(symbol, timeframe, 0, n+2)` — fetches from the current (in-progress) bar backwards
- Converts all broker server timestamps to UTC via `mt5_server_to_utc()`
- Marks the currently-open bar as `is_complete=False` based on whether its close time is in the past
- Reverses the array (MT5 returns newest-first; we want ascending)
- Approximates bid/ask from OHLC + spread points (sufficient for feature computation; live order prices always come from `adapter.get_price()`)

Also includes `fetch_historical_bars_mt5()` for research/data collection.

### 4.5 `features/pipeline.py`

Feature computation pipeline. Stateless — takes a DataFrame of OHLCV bars and returns a DataFrame of features. Inputs are `shift(1)`'d by default, making all features one bar lagged (the safest possible choice for avoiding look-ahead bias).

### 4.6 `research/labels.py`

Triple barrier label construction (offline only). For each bar, simulates both a LONG and SHORT trade and assigns a label based on which barrier is hit first: take profit, stop loss, or time.

Label barriers:
- TP: entry + 1.5 × ATR (LONG) or entry - 1.5 × ATR (SHORT)
- SL: entry - 1.0 × ATR (LONG) or entry + 1.0 × ATR (SHORT)
- Time: 12 bars maximum hold

**Important:** The live system's TP (1.5 × ATR) matches the label construction exactly. This is intentional — the model learned to predict which barrier would be hit first, so the live exits must use the same barriers.

### 4.7 `research/walk_forward.py`

Rolling walk-forward fold generation. Always 12 months training window, 1 month step, with purge and embargo zones to prevent data leakage at fold boundaries.

### 4.8 `research/model_training.py`

Train, save, and load models with SHA-256 integrity verification. A model artifact is only valid for deployment if `deployment_approved=True` in its manifest. The live loop and reconciliation both verify the SHA-256 on startup.

### 4.9 `risk/engine.py`

**UPDATED** — Now computes and returns `take_profit_price` in `RiskDecision`.

Circuit breakers (checked in order):
1. Kill switch active
2. Signal is ABSTAIN (direction=0)
3. Daily drawdown ≥ 3%
4. Max open positions (3)
5. Consecutive loss cooldown (4 losses → 12-bar pause)
6. ATR too small (< 5 pips)
7. Spread/ATR ratio too high (> 25%)
8. Correlated USD exposure cap ($10,000)
9. Same pair/direction already open

Position sizing formula:
```
risk_usd = equity × 1% × confidence_scalar
stop_distance_pips = (ATR × 1.0) + (spread × 0.5)
tp_distance_pips   = ATR × 1.5                         ← new
units = risk_usd / (stop_distance_pips × pip_value_per_unit)
units = clamp(units, min=1000, max=100000)
```

Stop and TP placement:
```python
# LONG trade (signal=1):
entry_price       = current_ask          # we pay the ask
stop_price        = entry_price - stop_distance_price
take_profit_price = entry_price + tp_distance_price   ← new

# SHORT trade (signal=-1):
entry_price       = current_bid          # we receive the bid
stop_price        = entry_price + stop_distance_price
take_profit_price = entry_price - tp_distance_price   ← new
```

### 4.10 `state/store.py`

Crash-safe SQLite persistence with WAL mode and `synchronous=FULL`. Every commit is fsynced to disk before the commit call returns. Survives `kill -9`.

### 4.11 `broker/adapter.py`

Abstract `BrokerAdapter` interface with 8 abstract methods. The `IdempotentOrderSubmitter` wraps order submission with UUID-based deduplication and exponential-backoff retry (1s, 2s, 4s). After 3 failures: `DEAD_LETTERED` + CRITICAL log.

### 4.12 `broker/mt5_adapter.py`

MetaTrader 5 concrete adapter. The only file that imports `MetaTrader5`. Converts all MT5 timestamps through `mt5_server_to_utc()`. Passes `take_profit_price` from the `Order` to `mt5.order_send()` as the `tp` field — so the TP is managed server-side at the broker, surviving crashes.

### 4.13 `execution/reconciliation.py`

7-step startup procedure. Must succeed before any trading begins. Never start trading without completing reconciliation.

The 7 steps:
1. Kill switch check — halt immediately if active
2. Broker connection — verify auth and account state
3. Position reconciliation — broker wins on existence, local wins on lineage
4. Order reconciliation — mark PENDING_SUBMIT as UNCONFIRMED; never auto-resubmit
5. P&L verification — cross-check local P&L against broker balance
6. Artifact integrity — verify SHA-256 of model and pipeline
7. Finalize — log summary, update high-water mark, write heartbeat

### 4.14 `execution/live_loop.py`

**FULLY IMPLEMENTED** — No stubs. The `_fetch_recent_bars()` stub is replaced with a real call to `data/ingestion.py`.

Changes vs original:
- `_fetch_recent_bars()` calls `fetch_recent_bars_mt5()` from `data/ingestion.py`
- `broker_tz_str` passed in at construction (required for bar timestamp conversion)
- `take_profit_price` from `RiskDecision` is passed to `Order` and on to the broker
- Per-pair deduplication via `_last_signal_bar` dict prevents double-processing the same bar
- ATR extracted from feature row (falls back to bar range if no ATR column found)

### 4.15 `start.py`

**NEW** — Copy-paste ready startup script. Fill in 5 values (MT5 credentials, broker timezone, artifact paths) and run `python start.py`. It:
1. Runs the full test suite — aborts if any test fails
2. Connects to MT5 and logs account equity
3. Safety-prompts if large equity detected on first run
4. Loads and verifies the model artifact
5. Starts the live loop with reconciliation

---

## 5. The ML Pipeline End-to-End

### Step 1: Collect historical data

```python
import MetaTrader5 as mt5
from data.ingestion import fetch_historical_bars_mt5
from shared.schemas import Pair
from datetime import datetime, timezone

mt5.initialize(login=YOUR_LOGIN, password=YOUR_PASSWORD, server=YOUR_SERVER)

bars = fetch_historical_bars_mt5(
    pair=Pair.EURUSD,
    date_from=datetime(2021, 1, 1, tzinfo=timezone.utc),
    date_to=datetime(2024, 1, 1, tzinfo=timezone.utc),
    timeframe_mt5=mt5.TIMEFRAME_M5,
    timeframe_sec=300,
    broker_tz_str="Etc/GMT-2",
)
```

### Step 2: Convert to DataFrame and audit

```python
from data.aggregation import bars_to_dataframe, detect_gaps
df = bars_to_dataframe(bars)
gaps = detect_gaps(df, timeframe_sec=300)
# Review gaps longer than 3 bars during session hours
```

### Step 3: Construct labels

```python
from research.labels import construct_labels, LabelConfig, get_label_distribution

config = LabelConfig(tp_atr_multiplier=1.5, sl_atr_multiplier=1.0, max_holding_bars=12)
labels = construct_labels(df, config, pair_is_jpy=False)

dist = get_label_distribution(labels)
print(dist)
# Healthy: LONG 20-30%, SHORT 20-30%, ABSTAIN 40-60%
```

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
pipeline.fit(df)   # NEVER fit on val or test data
```

### Step 5: Generate walk-forward folds

```python
from research.walk_forward import generate_folds, WalkForwardConfig

wf_config = WalkForwardConfig(
    train_months=12,
    val_months=2,
    test_months=1,
    purge_bars=12,
    embargo_bars=0,
    step_months=1,
)
folds = generate_folds(df, wf_config)
print(f"{len(folds)} folds generated")
```

### Step 6: Train and validate

```python
from research.model_training import train_model, evaluate_model, ModelConfig

model_config = ModelConfig(
    version="eurusd_5m_v001",
    model_class="lightgbm",
    pair="EUR/USD",
    timeframe="5m",
    train_end_date="",
)

all_results = []
for fold in folds:
    model, report = train_model(df, labels, pipeline, fold, model_config)
    all_results.append(report)
    print(f"Fold {fold.fold_idx}: Sharpe={report.sharpe:.2f}, Accuracy={report.accuracy:.2%}")

# Aggregate out-of-sample Sharpe across folds
import numpy as np
oos_sharpe = np.mean([r.sharpe for r in all_results])
print(f"OOS Sharpe: {oos_sharpe:.2f}")
# If > 2.0, check for look-ahead bias (see Section 9)
```

### Step 7: Save approved model

```python
from research.model_training import save_model

# Only save if validation passed
best_report = max(all_results, key=lambda r: r.sharpe)
if best_report.deployment_approved:
    artifact_dir, sha256 = save_model(
        model, model_config, best_report, base_path="artifacts/"
    )
    pipeline.save(f"artifacts/pipeline_{model_config.version}/")
    print(f"Model saved to {artifact_dir}")
    print(f"SHA-256: {sha256}")
else:
    print("Model did NOT pass validation — do not deploy")
```

---

## 6. The Signal-to-Order Path

What happens live for each completed 5-minute bar:

```
1. Bar completes (is_complete=True — close time is in the past)
   ↓
2. fetch_recent_bars_mt5() — last 112 bars from MT5
   ↓
3. pipeline.transform(df) — computes RSI, ATR, momentum features (all shifted 1 bar)
   ↓
4. model.predict_proba(feature_row)
   → [p_short, p_abstain, p_long]
   ↓
5. compute_signal(proba, min_confidence=0.55, min_margin=0.10)
   → (direction: -1/0/1, confidence: float)
   ↓ (direction==0 → ABSTAIN, log and continue)
6. adapter.get_price(pair) — fresh live bid/ask
   adapter.get_account_state() — current equity
   ↓
7. risk_engine.evaluate(...)
   → RiskDecision(
       trade_permitted=True,
       position_size_units=N,
       stop_price=X,
       take_profit_price=Y,    ← computed as entry ± 1.5×ATR
       max_loss_usd=Z,
     )
   ↓ (trade_permitted=False → log reason, continue)
8. Build Order(pair, side, units, stop_loss_price, take_profit_price, ...)
   ↓
9. IdempotentOrderSubmitter.submit(order):
   a. Write PENDING_SUBMIT to SQLite (fsync before network call)
   b. Send to MT5 via adapter.send_order() — includes TP and SL
   c. MT5 places order with broker; broker holds TP and SL server-side
   d. Update order status in SQLite
   ↓ (FILLED)
10. Create Position in state store
    ↓
11. Log trade to trade_log (JSONL)
```

The entire path from bar completion to order submission typically takes 50-200ms. For 5-minute bars, this is negligible.

---

## 7. Take-Profit System

### Why take-profit was added

The original system had no take-profit. This meant:
- A trade could be up 50 pips, then fall back to the stop loss and close at a loss
- The model might stay "long" for 12 bars even as a move fades
- P&L was entirely dependent on the stop loss and the model eventually generating an opposing signal

With a take-profit, the system now locks in gains when the price reaches the expected target. This changes the exit mechanic from "wait and hope" to "take the profit the model predicted."

### How take-profit is computed

```
tp_distance_pips = ATR × tp_atr_multiplier   (default: 1.5)
sl_distance_pips = ATR × sl_atr_multiplier + spread × 0.5   (default: 1.0 + buffer)

reward:risk ratio ≈ 1.5 / 1.0 = 1.5:1
```

The 1.5× multiplier matches the label construction in `research/labels.py` exactly. This is important: the model was trained on labels where LONG means "the price reached +1.5×ATR before -1.0×ATR". If the live TP is set differently, the exits won't match what the model predicted.

### Where take-profit is enforced

The TP is passed all the way from the risk engine → Order → MT5 adapter → broker. It is set **server-side at the broker** as the `tp` field in `mt5.order_send()`. This means:

- Your Python process does not need to be running to close the position at the TP
- If your VPS crashes, the broker will still close the trade when TP is hit
- The TP is not just a local alert — it is a real order at the broker

### Tuning take-profit

Edit `RiskParameters.tp_atr_multiplier` in `risk/engine.py`:

```python
tp_atr_multiplier: float = 1.5   # default — matches label construction
# 1.0  → tight TP, more wins but smaller gains (good if model has high accuracy)
# 2.0  → wide TP, fewer wins but larger gains when they hit
# 2.5+ → very wide, acts more like a trend-following system
```

If you change this value, retrain the model with the same TP ratio in `LabelConfig`. A model trained with 1.5× TP is tuned for that specific exit, and using it with a 2.5× TP will underperform.

---

## 8. Crash Safety and Recovery

What happens if the process is killed at each point:

**Killed before writing PENDING_SUBMIT:** No DB record. Clean — the order was never sent.

**Killed after PENDING_SUBMIT but before sending:** On restart, `_reconcile_orders()` finds the PENDING_SUBMIT order and marks it UNCONFIRMED. Operator reviews: was this sent? Check broker order history.

**Killed after sending but before receiving confirmation:** Order may or may not have been received. On restart, the SUBMITTED order is not in broker pending orders. Marked UNCONFIRMED. Check broker trade history.

**Killed with open position:** Broker has the position. Local DB also has it. Reconciliation verifies they match. The TP and SL orders remain active at the broker regardless.

**Killed with open position, local DB corrupted:** SQLite WAL mode means the DB cannot be corrupted mid-write. The WAL is replayed on next open.

### The kill switch

A DB record in `system_state` (`key='kill_switch'`, `value='true'`). Set by:
- The drawdown circuit breaker (3% daily drawdown)
- The heartbeat monitor (all 4 checks fail)
- Manually by the operator

Cleared **only manually**. After any automatic kill switch activation, a human must review the logs, understand what caused it, and decide it's safe to resume.

---

## 9. Look-Ahead Bias

Look-ahead bias is when your model is trained on information that would not have been available at the time the prediction was needed. It makes backtests look far better than live performance.

### The 5 places it can enter this system

1. **Rolling features not shifted:** All features use `shift(1)` — bar `t` features only use data up to bar `t-1`. Test: modify bar `t`'s price to 999.0 and verify features at bar `t-1` don't change.

2. **Labels looking into the future:** The triple barrier label for bar `t` looks at bars `t+1` through `t+12`. By design — labels represent what would happen if you entered at bar `t+1`. The purge zone (12 bars) ensures none of these future bars are in the training set.

3. **Pipeline fit on test data:** `pipeline.fit()` is called only on training data. Never on val or test.

4. **Train/val/test overlap:** Verified by `test_no_overlap_between_splits` unit test.

5. **Signals using current bar's close:** The live loop only generates signals on bars where `is_complete=True` (close time is in the past). Execution happens at the next bar's open.

### The suspicious Sharpe test

If your out-of-sample Sharpe across walk-forward folds is > 2.0, you almost certainly have a look-ahead bug. Genuine FX signals produce Sharpe of 0.6–1.5 at best after costs. If you see 3.0+, run:

1. Re-run backtesting with fills at next-bar open ask (not mid)
2. Re-run with 2× spread
3. Test on a completely different year
4. Verify every rolling calculation uses `shift(1)`
5. Compare vectorized vs. event-driven backtest Sharpe (should be similar; large divergence means the vectorized test is optimistic)

---

## 10. Risk Management

### Position-level controls

Every position has a hard stop loss set at entry with the broker (server-side). The TP is also set server-side. Even if the Python process crashes, the internet goes down, or the VPS reboots, the broker will execute both the SL and TP.

Stop distance:
```
stop_distance = ATR × 1.0 + spread × 0.5
```

### Account-level controls

| Control | Threshold | Action |
|---|---|---|
| Daily drawdown | 3% equity | Kill switch + halt all trading |
| Weekly drawdown | 7% equity | Halt, require manual review |
| Monthly drawdown | 12% equity | Halt, retrain required |
| Consecutive losses | 4 in a row | 12-bar cooldown (~1 hour) |
| Live Sharpe vs backtest | <30% of backtest | Model has failed |
| Mean slippage | >2 pips over 20 trades | Broker/execution issue |

### The capital ramp-up protocol

Never go from backtesting to full capital.

| Stage | Duration | Capital | Review |
|---|---|---|---|
| Demo forward test | 8 weeks | 0% (fake) | Pass/fail criteria |
| Stage 1 live | 4 weeks | 10% | Daily review |
| Stage 2 live | 6 weeks | 25% | Weekly review |
| Stage 3 live | 8 weeks | 50% | Biweekly review |
| Stage 4 live | Ongoing | 100% | Monthly review |

Demo pass criteria:
- ≥ 40 trades completed
- Sharpe ≥ 0.5
- Max drawdown ≤ 10%
- Zero reconciliation errors
- ABSTAIN rate matches validation ± 10%
- Tested against: news event (NFP/FOMC), gap open, manual disconnection

---

## 11. Deployment Guide

### Prerequisites

- Python 3.12+
- MetaTrader 5 terminal installed and running on **Windows** (MT5 Python API is Windows-only)
- 3+ years of 5-minute OHLCV data
- A demo account at a broker that provides MT5 access

### Install dependencies

```bash
pip install pandas numpy scikit-learn lightgbm xgboost statsmodels structlog
pip install MetaTrader5    # Windows only
```

### Configuration (in start.py)

```python
MT5_LOGIN    = 12345678
MT5_PASSWORD = "your_password"
MT5_SERVER   = "YourBroker-Demo"   # Find in MT5 → File → Open Account
BROKER_TZ    = "Etc/GMT-2"         # Find in MT5 → Tools → Options → Server
MODEL_ARTIFACT_DIR    = "artifacts/eurusd_5m_v001_20240115"
PIPELINE_ARTIFACT_DIR = "artifacts/pipeline_eurusd_5m_v001"
```

### Finding your broker's timezone

In MetaTrader 5:
- Go to View → Market Watch
- Right-click any symbol → Specification
- Look for "Trading Sessions" — times are in broker server time
- Compare to UTC to determine offset

Common values: `"Etc/GMT-2"` (most EU brokers), `"Etc/GMT-3"` (summer), `"US/Eastern"` (US brokers).

### Run tests first

```bash
cd fx_trading/
python -m unittest tests.test_all -v
# Expected: 48 tests, 0 errors, 0 failures
# If any test fails — do not proceed.
```

### Start the system

```bash
python start.py
```

The startup script will:
1. Run all 48 tests — abort if any fail
2. Connect to MT5
3. Load the model artifact (verifies SHA-256 and `deployment_approved=True`)
4. Run 7-step reconciliation
5. Start the execution loop

### Keep it running (Windows Task Scheduler)

Create a `run.bat`:
```batch
@echo off
cd C:\fx_trading
python start.py >> logs\startup.log 2>&1
```

Schedule it to run at system startup via Task Scheduler. Set to restart on failure.

### Keep it running (Linux with systemd)

If using a Linux VPS (see Section 12), create `/etc/systemd/system/fx-trader.service`:

```ini
[Unit]
Description=FX Trading Bot
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/fx_trading
ExecStart=/usr/bin/python3 start.py
Restart=on-failure
RestartSec=10s
StandardOutput=append:/home/ubuntu/fx_trading/logs/live.log
StandardError=append:/home/ubuntu/fx_trading/logs/live.log

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable fx-trader
sudo systemctl start fx-trader
sudo systemctl status fx-trader
```

---

## 12. Free Hosting Options

Running a trading bot requires a machine that is **always on, 24/7**. Your laptop won't do. Here are the options ranked by suitability:

### Option 1: Oracle Cloud Free Tier (Recommended)

**Permanently free.** Oracle's Always Free tier includes a VM with 4 ARM CPUs, 24 GB RAM — more than enough for this system.

Setup:
1. Create account at cloud.oracle.com (requires credit card for verification; won't be charged)
2. Create an instance: Compute → Instances → Create Instance
3. Choose "Ampere" shape (ARM) — this is the always-free option
4. Choose Ubuntu 22.04
5. Download your SSH key pair

Then SSH in and set up:
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git wine
git clone your-repo /home/ubuntu/fx_trading
cd /home/ubuntu/fx_trading
pip3 install -r requirements.txt
```

**The MT5 problem on Linux:** MetaTrader 5's Python API only works on Windows. On Linux, you have two options:

**Option A: Use Wine** — Install Wine on the Oracle VM, run the MT5 Windows executable under Wine, and use the MT5 Python API via Wine's Python. Works but finicky.

**Option B: Use OANDA instead of MT5** — OANDA has a REST API that works natively on Linux. Create `broker/oanda_adapter.py` implementing the same 8 abstract methods as `MT5Adapter`. The rest of the system doesn't change.

**Option C: Windows VM on Oracle** — Oracle's free tier can run Windows Server (bring your own license). Expensive in practice.

**Most practical path:** Use a cheap Windows VPS (Contabo, Hetzner, etc. at $5-10/month) for MT5, and use Oracle's free tier only for monitoring/logging if needed.

### Option 2: Windows VPS (~$5-10/month, not free)

Providers: Contabo, Hetzner, Vultr, DigitalOcean, OVH.

Best option if you need native MT5 support. MT5 runs natively on Windows, no Wine needed.

### Option 3: Your Own Windows PC

Fine for testing. Not reliable for live trading — power outages, Windows updates, sleep mode, and internet drops will all interrupt the bot. Use this for the demo forward test period only.

### Option 4: Railway / Render (Not suitable)

These platforms put processes to sleep after inactivity. A trading bot that sleeps misses bars, misses stop-loss triggers, and accumulates positions it can't track. Do not use.

---

## 13. How the System Connects to the Market

The connection chain:

```
Your Python code
    ↓ MetaTrader5 Python package (Windows DLL)
MT5 Terminal (running on Windows)
    ↓ proprietary encrypted protocol
Your Broker's Servers
    ↓ interbank network
Liquidity Providers (banks, ECNs)
    ↓
The actual FX market (decentralized)
```

Your Python code never touches the real market directly. It talks to the MT5 terminal, which talks to your broker, which routes your orders to liquidity providers.

Price data flows back the same way:
- Liquidity providers stream prices to your broker
- Your broker streams them to the MT5 terminal
- The terminal makes them available via `mt5.symbol_info_tick()` and `mt5.copy_rates_from_pos()`
- Your code reads them via the MT5 Python API

Order lifecycle:
1. You call `mt5.order_send()` with the order details (including SL and TP)
2. MT5 terminal sends it to the broker
3. The broker executes it at the best available price
4. The broker confirms back with a fill price and ticket number
5. MT5 returns the result to your Python code
6. Your code writes the fill to SQLite

The TP and SL are held at the broker. Once set, they are active regardless of whether your Python process is running.

---

## 14. Performance Tuning

### Conservative improvements (recommended)

**Add more currency pairs.** Your model trades one pair. Adding EUR/JPY, AUD/USD, or USD/CAD increases the number of signals per day without changing the strategy. Each pair needs its own `pipeline.fit()` and potentially its own model (correlations differ).

**Lower the confidence threshold.** `min_confidence=0.55` is conservative. Lowering to `0.51` produces more trades but some will be marginal quality. Monitor the ABSTAIN rate — if it drops below 30%, you're likely trading noise.

**Retrain quarterly.** FX market regimes change. A model trained on 2021-2023 data may underperform in 2024. Schedule quarterly retraining and compare new vs. old model OOS Sharpe before replacing.

**Add session filters.** The London session (07:00–16:00 UTC) and US session (13:00–22:00 UTC) have better liquidity and tighter spreads. Filtering to trade only during these windows reduces slippage.

### Aggressive improvements (higher risk)

**Increase `risk_per_trade_pct`** in `RiskParameters`:
```python
risk_per_trade_pct: float = 0.02   # 2% per trade instead of 1%
```
This directly doubles both gains and losses. The system's math is still correct; the risk envelope is just wider.

**Raise `max_open_positions`** to 5 or 6. Allows more concurrent trades. Increases exposure to correlated moves (multiple FX pairs often move together on USD news).

**Widen or remove the daily drawdown kill switch:**
```python
max_daily_drawdown_pct: float = 0.05   # 5% instead of 3%
```
Allows trading through worse days. The risk is that a bad regime runs further before the bot stops.

**Use higher leverage.** FX brokers offer 30:1 to 500:1. You access leverage by choosing a smaller `min_units` and larger `max_units`:
```python
min_units: int = 10_000    # 0.1 standard lot minimum
max_units: int = 500_000   # 5 standard lots maximum
```
With higher max_units, the position sizing formula will allocate larger positions at the same equity-risk percentage.

**Switch to 1-minute bars.** More signals per day. Significantly noisier — the model needs retraining. Set `timeframe_sec=60` in `LiveExecutionLoop` and `timeframe_mt5=mt5.TIMEFRAME_M1`. Spread costs become more significant at 1m.

**Remove the ABSTAIN class.** Force the model to always pick BUY or SELL. In `compute_signal()`, lower `min_confidence` to 0.40 and `min_margin` to 0.05. Way more trades, but many will be low-quality.

**News trading.** Manually disable the kill switch and widen stop losses before NFP/FOMC releases. These events cause 30-100 pip moves in seconds. Spreads also widen massively, so entry cost is high. Not recommended for automation — too unpredictable.

---

## 15. Common Pitfalls

**Wrong broker timezone.** If `BROKER_TZ` is wrong, bar timestamps will be off. Features computed at the wrong time introduce subtle look-ahead bias. Verify by comparing a bar's `utc_open` with the MT5 terminal's displayed time.

**Fitting the pipeline on all data.** Always fit on training data only. `pipeline.fit(df)` where `df` contains test data contaminates all features with future statistics. This is one of the most common (and hardest to detect) look-ahead bugs.

**Not running reconciliation after a crash.** `start.py` always runs reconciliation. Never bypass this — you may have open positions at the broker that don't exist in your local DB.

**Using float for prices.** `1.0900 + 0.0001` in Python float arithmetic gives `1.0900000000000001`. Always use `Decimal`. The schemas enforce this, but if you write any custom price logic outside the system, remember this.

**Ignoring DEAD_LETTERED orders.** A dead-lettered order means 3 retries failed. The position may or may not be open at the broker. This requires manual review. Set up a log alert on `"DEAD_LETTERED"` messages.

**Forgetting that SL and TP are at the broker.** If you open a position and then modify `stop_loss_price` in your local DB, the actual SL at the broker does not change. You must call `adapter.modify_stop_loss()` to move it.

---

## 16. Troubleshooting

**"MT5 initialization failed"**
- Is the MT5 terminal running?
- Is the terminal logged in to an account?
- Is Python 32-bit or 64-bit? MT5 Python API requires 64-bit Python.
- Are your login credentials correct?

**"Reconciliation requires human review"**
- Check the logs for `orphaned_positions` or `unconfirmed_orders`
- Log into MT5 manually and verify open positions
- If positions in broker match what you expect, update your local DB manually and restart

**"All-NaN column in bar data"**
- MT5 returned a bar with no tick volume. Often happens at market open or on illiquid symbols.
- Usually self-corrects on the next bar. If persistent, check that the symbol is available in Market Watch.

**"ATR too small: X pips"**
- The market is in an unusually quiet period (often ahead of major news)
- This is working as intended. The system won't trade in flat markets.

**OOS Sharpe > 2.0 but live performance is poor**
- You have look-ahead bias. Run the diagnostic checklist in Section 9.

**Kill switch keeps activating**
- Check daily P&L. If losses are consistent, the model may have stopped working (regime change).
- Review the `close_reason` field in `trade_log` — are all closures hitting the stop loss?
- Run Section 14's "Live Sharpe vs backtest" check. If live Sharpe is < 30% of backtest, retrain.

---

## Running the Test Suite

```bash
cd fx_trading/
python -m unittest tests.test_all -v
```

Expected: 48 tests, OK. If any test fails, do not proceed. The failing test is protecting you from a specific class of bug that can cause real financial losses.

---

*This document covers the full system as of March 2026. Implementation order: schemas → time_utils → aggregation → ingestion → pipeline → labels → walk_forward → model_training → risk engine → state store → broker adapter → execution. Each module must have its tests passing before the next is built.*
