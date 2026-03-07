# FX Trading System

A production-grade, ML-driven algorithmic trading system for foreign exchange markets. Trades EUR/USD (primary), GBP/USD, and USD/JPY on 5-minute bars using a LightGBM or XGBoost classifier trained with a walk-forward framework. Connects to MetaTrader 5 via its Python API.

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
13. [Monitoring Dashboard](#13-monitoring-dashboard)
14. [Performance Tuning](#14-performance-tuning)
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
│   └── ingestion.py            ← MT5 bar fetching (fetch_recent_bars_mt5, fetch_historical_bars_mt5)
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
│   └── engine.py               ← Circuit breakers, position sizing, stop + take-profit computation
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
│   └── live_loop.py            ← Main execution loop. Fully implemented.
│
├── monitoring/
│   ├── dashboard.py            ← Real-time Flask web dashboard (port 5050)
│   └── README.md               ← Dashboard usage and log format reference
│
├── backtesting/                ← Placeholder for event-driven backtester
├── tests/
│   └── test_all.py             ← 48 tests covering every critical safety property
│
├── artifacts/                  ← Model and pipeline artifacts stored here
├── logs/                       ← JSONL structured log files
├── start.py                    ← Copy-paste ready startup script
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
                  ←── monitoring/
```

What is never allowed:
- `research/` importing from `execution/` or `broker/`
- `risk/` importing model logic
- `broker/` importing feature or model logic
- `monitoring/` touching any execution code — it reads only from the DB and log files
- Circular imports of any kind

---

## 4. Module Reference

### 4.1 `shared/schemas.py`

Canonical data contracts. The single source of truth for every type used across the system. Key types: `OHLCVBar`, `Tick`, `Order`, `Position`, `RiskDecision`, `TradeRecord`.

Key design decisions:
- **Frozen dataclasses for market data** — immutable once created. Prevents silent mutation bugs.
- **`Decimal` for prices, never `float`** — floating-point arithmetic accumulates rounding errors in P&L calculations.
- **UTC enforcement on every timestamp** — `__post_init__` validators reject naive datetimes at construction time.

`RiskDecision` includes `take_profit_price` — the TP is computed by the risk engine at the same time as the stop loss, using the same ATR. This ensures the TP:SL ratio always matches what the model was trained to predict.

### 4.2 `data/time_utils.py`

UTC enforcement and broker timezone conversion. The only place `mt5_server_to_utc()` is implemented. Every timestamp that comes from an external system (MT5, OANDA, Reuters) must pass through this module before entering the rest of the system.

### 4.3 `data/aggregation.py`

Deterministic tick-to-bar aggregation with gap-filling for short runs within market hours. Gaps longer than 3 bars or outside session hours are left as-is.

### 4.4 `data/ingestion.py`

Concrete MT5 bar fetching. Implements what was previously a stub.

Key functions:
- `fetch_recent_bars_mt5(pair, n, timeframe_mt5, timeframe_sec, broker_tz_str)` — used by the live loop each bar
- `fetch_historical_bars_mt5(...)` — used by research for data collection

How `fetch_recent_bars_mt5` works:
- Calls `mt5.copy_rates_from_pos(symbol, timeframe, 0, n+2)` to fetch from the current bar backwards
- Converts all broker server timestamps to UTC via `mt5_server_to_utc()`
- Marks the currently-open bar as `is_complete=False` based on whether its close time is in the past
- Reverses the array (MT5 returns newest-first; we want ascending)
- Approximates bid/ask from OHLC + spread points (sufficient for feature computation; live order prices always come from `adapter.get_price()`)

### 4.5 `features/pipeline.py`

Feature computation pipeline. Stateless — takes a DataFrame of OHLCV bars and returns a DataFrame of features. All inputs are `shift(1)`'d by default, making every feature one bar lagged (the safest possible approach for avoiding look-ahead bias).

### 4.6 `research/labels.py`

Triple barrier label construction (offline only). For each bar, simulates both a LONG and SHORT trade and assigns a label based on which barrier is hit first: take profit, stop loss, or time.

Label barriers:
- TP: entry + 1.5 × ATR (LONG) or entry − 1.5 × ATR (SHORT)
- SL: entry − 1.0 × ATR (LONG) or entry + 1.0 × ATR (SHORT)
- Time: 12 bars maximum hold

**Important:** The live system's TP (1.5 × ATR) matches the label construction exactly. This is intentional — the model learned to predict which barrier would be hit first, so the live exits must use the same barriers.

### 4.7 `research/walk_forward.py`

Rolling walk-forward fold generation. Always 12 months training window, 1 month step, with purge and embargo zones to prevent data leakage at fold boundaries.

### 4.8 `research/model_training.py`

Train, save, and load models with SHA-256 integrity verification. A model artifact is only valid for deployment if `deployment_approved=True` in its manifest. The live loop and reconciliation both verify the SHA-256 on startup.

### 4.9 `risk/engine.py`

Computes `take_profit_price` and `stop_loss_price` together in `RiskDecision`.

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

Position sizing:
```
risk_usd           = equity × 1% × confidence_scalar
stop_distance_pips = (ATR × 1.0) + (spread × 0.5)
tp_distance_pips   = ATR × 1.5
units              = risk_usd / (stop_distance_pips × pip_value_per_unit)
units              = clamp(units, min=1000, max=100000)
```

Stop and TP placement:
```
# LONG (signal=1):
entry_price       = current_ask
stop_price        = entry_price - stop_distance_price
take_profit_price = entry_price + tp_distance_price

# SHORT (signal=-1):
entry_price       = current_bid
stop_price        = entry_price + stop_distance_price
take_profit_price = entry_price - tp_distance_price
```

### 4.10 `state/store.py`

Crash-safe SQLite persistence with WAL mode and `synchronous=FULL`. Every commit is fsynced to disk before the call returns. Survives `kill -9`.

### 4.11 `broker/adapter.py`

Abstract `BrokerAdapter` interface with 8 abstract methods. The `IdempotentOrderSubmitter` wraps order submission with UUID-based deduplication and exponential-backoff retry (1s, 2s, 4s). After 3 failures: `DEAD_LETTERED` + CRITICAL log.

### 4.12 `broker/mt5_adapter.py`

Concrete MetaTrader 5 implementation of `BrokerAdapter`. All 8 abstract methods implemented. Handles MT5 connection lifecycle, symbol normalization, tick fetching, and order placement with server-side SL and TP.

### 4.13 `execution/reconciliation.py`

7-step broker/local reconciliation. Runs on every startup before the bar loop begins. Detects orphaned positions, unconfirmed orders, and state drift between local DB and broker. If any step requires human review, the loop does not start.

### 4.14 `execution/live_loop.py`

Main execution loop. Polls on a 1-second interval. On each new complete bar:

1. Check kill switch
2. Check market hours
3. Fetch recent bars via `data/ingestion.py`
4. Compute features via `features/pipeline.py`
5. Get model signal via `research/model_training.predict_proba`
6. Evaluate risk via `risk/engine.py`
7. If approved: persist order, submit to broker via `IdempotentOrderSubmitter`

NaN in any feature → ABSTAIN. The loop never retries or patches NaN values.

Heartbeat is written to state every 30 seconds. The dashboard displays time since last heartbeat.

### 4.15 `monitoring/dashboard.py`

Real-time Flask web dashboard. Launched automatically by `start.py` in a background daemon thread on port 5050. Reads only from `state/trading.db` and `logs/*.jsonl` — it does not touch any execution code.

| Panel | Data Source |
|---|---|
| KPI bar (positions, P&L, high-water mark, heartbeat) | `state/trading.db` → `system_state` |
| Open Positions | `state/trading.db` → `positions` |
| Daily P&L chart (30-day bar chart) | `state/trading.db` → `daily_pnl` |
| Recent Orders (with confidence bar) | `state/trading.db` → `orders` |
| AI Decisions & Signals live feed | `logs/*.jsonl` |
| Kill-switch indicator | `state/trading.db` → `system_state` |

Auto-refreshes every 5 seconds. Any JSONL file in `logs/` is picked up automatically.

---

## 5. The ML Pipeline End-to-End

### Training (offline)

1. Fetch historical 5-minute bars using `fetch_historical_bars_mt5()`
2. Compute features with `FeaturePipeline.fit_transform(bars)` — all features are lag-1
3. Construct triple-barrier labels with `research/labels.py`
4. Generate walk-forward folds with `research/walk_forward.py`
5. Train LightGBM or XGBoost on each fold's training window, evaluate on the OOS window
6. Review OOS metrics — must pass minimum criteria before approving
7. Save with `save_model_artifact()`, then set `deployment_approved=True` in the manifest
8. Save the fitted pipeline separately with `pipeline.save(path)` — its SHA-256 is also verified at startup

### Live inference (each bar)

1. `fetch_recent_bars_mt5()` → last 110 5-minute bars
2. `pipeline.transform(bars)` → feature row for the latest complete bar
3. `predict_proba(model, features, model_class)` → class probabilities
4. `compute_signal(probabilities, threshold=0.60)` → LONG / SHORT / ABSTAIN
5. `risk_engine.evaluate(signal, ...)` → `RiskDecision` (approved/blocked + position size + SL + TP)
6. If approved: place order with server-side SL and TP

---

## 6. The Signal-to-Order Path

```
bar closes
    → fetch bars (ingestion.py)
    → compute features (pipeline.py)
    → model inference (model_training.py)
    → risk engine (risk/engine.py)
        → if APPROVED:
            → persist order to SQLite (state/store.py)
            → submit via IdempotentOrderSubmitter (broker/adapter.py)
                → MT5Adapter.place_order() (broker/mt5_adapter.py)
```

The broker receives a fully-formed order with entry, stop-loss, and take-profit already specified. There is no local exit management — SL and TP are server-side at the broker.

---

## 7. Take-Profit System

Both the stop loss and take profit are set server-side at the broker at the time the order is placed. Even if the Python process crashes, the internet goes down, or the VPS reboots, the broker will execute both.

The TP distance (1.5 × ATR) matches the label construction in `research/labels.py` exactly. This is not a coincidence — the model was trained to predict which barrier would be hit first, so using different distances in live trading would create a silent mismatch between what the model learned and what you're actually trading.

---

## 8. Crash Safety and Recovery

The system is designed to survive any of the following without human intervention:

- `kill -9` on the Python process
- VPS reboot (systemd auto-restarts)
- Network disconnection (exponential-backoff retry in `IdempotentOrderSubmitter`)
- MT5 terminal restart

On startup, `start.py` calls `run_reconciliation()` before the bar loop begins. Reconciliation:

1. Verifies model and pipeline SHA-256 integrity
2. Fetches open positions from broker
3. Compares to local state
4. Closes positions that exist at broker but not locally (orphaned)
5. Marks locally-open positions as closed if broker confirms they are
6. Flags any unconfirmed orders for review
7. If anything requires human review → loop does not start

SQLite WAL mode with `synchronous=FULL` means every write is fsynced before the function returns. There is no window where a crash can leave the DB in an inconsistent state.

---

## 9. Look-Ahead Bias

### What it is

Look-ahead bias occurs when the model sees information during training that would not have been available at prediction time. It causes backtest results to look significantly better than live performance. The mismatch is often invisible until you lose money on it.

### How this system prevents it

- All features use `shift(1)` — every input to the model is the previous bar's value, never the current bar.
- The live loop only processes bars where `is_complete=True`. The currently-open bar is never used.
- Walk-forward folds include a purge period (removes bars at the boundary between train and test) and an embargo period (removes bars at the start of the test window that might be contaminated).
- The feature pipeline is stateless. It cannot carry information forward between bars.
- The test suite includes a specific look-ahead detection test: it modifies the last bar's close price and verifies that the second-to-last bar's features are unchanged.

### Diagnostic checklist if you suspect look-ahead bias

1. Inspect every feature in `pipeline.py` — confirm every rolling calculation uses `shift(1)` or equivalent
2. Re-run the walk-forward OOS evaluation with 2× spread
3. Test on a completely different year
4. Compare vectorized backtest Sharpe vs. event-driven Sharpe — large divergence means the vectorized test is optimistic
5. Run `python -m unittest tests.test_all -v` and confirm the look-ahead test passes

---

## 10. Risk Management

### Position-level controls

Every position has a hard stop loss and take profit set at entry with the broker (server-side).

```
stop_distance = ATR × 1.0 + spread × 0.5
tp_distance   = ATR × 1.5
```

### Account-level controls

| Control | Threshold | Action |
|---|---|---|
| Daily drawdown | 3% equity | Kill switch + halt all trading |
| Weekly drawdown | 7% equity | Halt, require manual review |
| Monthly drawdown | 12% equity | Halt, retrain required |
| Consecutive losses | 4 in a row | 12-bar cooldown (~1 hour) |
| Live Sharpe vs backtest | < 30% of backtest | Model has failed — retrain |
| Mean slippage | > 2 pips over 20 trades | Broker/execution issue |

### Capital ramp-up protocol

Never go from backtesting to full capital.

| Stage | Duration | Capital | Review Cadence |
|---|---|---|---|
| Demo forward test | 8 weeks | 0% (paper) | Pass/fail criteria |
| Stage 1 live | 4 weeks | 10% | Daily |
| Stage 2 live | 6 weeks | 25% | Weekly |
| Stage 3 live | 8 weeks | 50% | Biweekly |
| Stage 4 live | Ongoing | 100% | Monthly |

Demo pass criteria:
- ≥ 40 trades completed
- Sharpe ≥ 0.5
- Max drawdown ≤ 10%
- Zero reconciliation errors
- ABSTAIN rate matches validation ± 10%
- Tested through: news event (NFP/FOMC), gap open, manual disconnection

---

## 11. Deployment Guide

### Prerequisites

- Python 3.12+
- MetaTrader 5 terminal installed and running on **Windows** (MT5 Python API is Windows-only)
- 3+ years of 5-minute OHLCV data
- A trained, approved model artifact in `artifacts/`
- A demo account at a broker that provides MT5 access

### Install dependencies

```bash
pip install -r requirements.txt
pip install MetaTrader5    # Windows only
```

`requirements.txt` currently pins:

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
xgboost>=2.0.0
statsmodels>=0.14.0
pandas-ta>=0.3.14b
pytz>=2023.3
structlog>=23.0.0
pytest>=7.4.0
```

Flask is required for the dashboard (`pip install flask`) — it is not in `requirements.txt` by default.

### Configuration (in `start.py`)

```python
MT5_LOGIN             = 12345678
MT5_PASSWORD          = "your_password"
MT5_SERVER            = "YourBroker-MT5"   # View → File → Open Account in MT5
BROKER_TZ             = "Etc/GMT-2"        # View → Tools → Options → Server

MODEL_ARTIFACT_DIR    = "artifacts/eurusd_5m_v001_20260305"
PIPELINE_ARTIFACT_DIR = "artifacts/pipeline_eurusd_5m_v001"
STATE_DB_PATH         = "state/trading.db"
LOG_DIR               = "logs"
DASHBOARD_PORT        = 5050
```

### Finding your broker's timezone

In MetaTrader 5: View → Market Watch → right-click any symbol → Specification → Trading Sessions. Times shown are in broker server time. Compare to UTC to determine offset.

Common values: `"Etc/GMT-2"` (most EU brokers), `"Etc/GMT-3"` (summer time), `"US/Eastern"` (US brokers).

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

`start.py` performs the following in order:

1. Runs all 48 tests — aborts if any fail
2. Connects to MT5
3. Loads the model artifact (verifies SHA-256 + `deployment_approved=True`)
4. Loads the feature pipeline artifact (verifies SHA-256)
5. Sets up `IdempotentOrderSubmitter`
6. Sets up `RiskEngine`
7. Determines trading pairs from the model manifest
8. Starts the monitoring dashboard in a background thread on port 5050
9. Runs 7-step reconciliation
10. Enters the live execution loop

### Keep it running on Windows (Task Scheduler)

Create `run.bat`:
```batch
@echo off
cd C:\fx_trading
python start.py >> logs\startup.log 2>&1
```

Schedule via Task Scheduler → run at system startup → restart on failure.

### Keep it running on Linux (systemd)

```ini
# /etc/systemd/system/fx-trader.service
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

Running a trading bot requires a machine that is always on, 24/5. Here are the options ranked by suitability.

### Option 1: Oracle Cloud Free Tier (Recommended)

Oracle's Always Free tier includes a VM with 4 ARM CPUs and 24 GB RAM — more than enough.

1. Create account at cloud.oracle.com (credit card required for verification; you won't be charged)
2. Create an instance: Compute → Instances → Create Instance
3. Choose "Ampere" shape (ARM) — this is the always-free option
4. Choose Ubuntu 22.04

**The MT5 problem on Linux:** MT5's Python API is Windows-only. On Linux you have three options:

- **Wine:** Install Wine, run the MT5 Windows executable under Wine, use the Python API via Wine's Python. Works but can be finicky.
- **OANDA adapter:** OANDA has a REST API that works natively on Linux. Create `broker/oanda_adapter.py` implementing the same 8 abstract methods as `MT5Adapter`. The rest of the system does not change.
- **Windows VPS:** Use a cheap Windows VPS (Contabo, Hetzner, ~$5–10/month) and run MT5 natively there.

The most practical path for most users: a cheap Windows VPS for MT5.

### Option 2: GitHub Actions / Railway / Render

These free-tier compute options have usage limits and sleep on inactivity — not suitable for a trading loop that must run every 5 minutes around the clock.

---

## 13. Monitoring Dashboard

The dashboard is launched automatically by `start.py` and runs in a background daemon thread.

```
http://localhost:5050    (or http://<your-host-ip>:5050)
```

Auto-refreshes every 5 seconds. No login required.

To run it standalone (without `start.py`):

```bash
pip install flask
python -m monitoring.dashboard          # port 5050
python -m monitoring.dashboard 8080     # custom port
```

The dashboard reads from `state/trading.db` and any JSONL files in `logs/`. It does not import from `execution/`, `broker/`, or `risk/` — it cannot accidentally interfere with the live loop.

---

## 14. Performance Tuning

### Reducing latency

The live loop polls every 1 second. In normal FX conditions on 5-minute bars, this is more than sufficient. If you need lower latency, reduce the `time.sleep(1)` interval in `live_loop._iteration()`.

### Aggressive position sizing

To increase `risk_per_trade_pct` beyond 1%, modify `RiskParameters` in `risk/engine.py`. Do not exceed 2% without extensive additional testing. Risk of ruin grows non-linearly.

### Model retraining schedule

Retrain when any of the following occur:
- Live Sharpe drops below 30% of out-of-sample backtest Sharpe over a 3-month rolling window
- Monthly drawdown exceeds 12%
- Market microstructure changes significantly (e.g., spread regime change)

Retrain workflow: fetch new data → run full walk-forward → review OOS metrics → set `deployment_approved=True` → restart the system (reconciliation runs automatically).

---

## 15. Common Pitfalls

**The model abstains constantly (ABSTAIN rate > 90%)**
- Confidence threshold (default 0.60) may be too high for current market conditions.
- Check that ATR is within normal range — the `atr_too_small` circuit breaker blocks trades when ATR < 5 pips. The system won't trade in flat markets.

**OOS Sharpe > 2.0 but live performance is poor**
- You almost certainly have look-ahead bias. Run the diagnostic checklist in Section 9.
- Also verify that the production pipeline artifact matches the one used during walk-forward evaluation.

**Kill switch keeps activating**
- Check daily P&L. If losses are consistent, the model may have stopped working (regime change).
- Review the `close_reason` field in `trade_log` — if all closures are hitting the stop loss, the model has no edge.
- Run the Sharpe comparison: if live Sharpe is < 30% of backtest Sharpe, retrain.

**Reconciliation fails on startup**
- Check `logs/live.jsonl` for the specific step that failed.
- Most common cause: an order was placed just before the system crashed, and its fill status is ambiguous. Inspect the order in MT5 manually and update `state/trading.db` accordingly, then restart.

**Dashboard shows stale heartbeat**
- The live loop writes heartbeat every 30 seconds. If the dashboard shows > 60 seconds since last heartbeat, the loop has likely crashed. Check `logs/live.jsonl` for the last CRITICAL message.

---

## 16. Troubleshooting

**MT5 connection fails**
- Confirm MT5 terminal is open and logged in
- Confirm `MT5_SERVER` in `start.py` matches exactly what is shown in MT5 → File → Open Account
- The MT5 Python API only works on Windows (or Wine on Linux)

**SHA-256 mismatch on model load**
- The model `.pkl` file has been modified since training. Do not proceed.
- Restore from backup or retrain.

**`deployment_approved` error**
- You must manually set `"deployment_approved": true` in `artifacts/<your_artifact>/model.manifest.json` after reviewing OOS metrics. This is intentional — it prevents accidentally deploying an untested model.

**Feature NaN after warmup period**
- Run `pipeline.transform(bars)` on your bar data and inspect with `df.isnull().any()`.
- Ensure you are passing at least `lookback_bars` (default 110) complete bars.
- Check that `ingestion.py` is not returning bars with missing OHLCV fields.

---

## Running the Test Suite

```bash
cd fx_trading/
python -m unittest tests.test_all -v
```

Expected output: **48 tests, OK.**

If any test fails, do not proceed. Each test protects against a specific class of bug that can cause real financial losses. The test suite covers: UTC enforcement, look-ahead detection, pipeline save/load integrity, artifact tamper detection, risk circuit breakers, idempotent order submission, state persistence, and reconciliation logic.

---

*Last updated: March 2026. Implementation is complete across all core modules: schemas, time_utils, aggregation, ingestion, pipeline, labels, walk_forward, model_training, risk engine, state store, broker adapter, live loop, reconciliation, and monitoring dashboard. The `backtesting/` module remains a placeholder pending an event-driven backtester implementation.*