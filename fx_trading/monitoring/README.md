# monitoring/dashboard.py

Real-time web dashboard for the FX Trading system.

## What it shows

| Panel | Data source |
|---|---|
| **KPI bar** — open positions, today's P&L, high-water mark, last heartbeat, orders & signals count | `state/trading.db` |
| **Open Positions** (left) — each live trade with entry, stop-loss, units, unrealized P&L | `state/trading.db` → `positions` table |
| **Daily P&L chart** (center) — 30-day bar chart, green/red per day | `state/trading.db` → `daily_pnl` table |
| **Recent Orders** (center bottom) — full order table with confidence bar and model version | `state/trading.db` → `orders` table |
| **AI Decisions & Signals** (right) — live feed of every signal the model generated and every risk engine decision, with confidence, pair, bar time, and block reason | `logs/*.jsonl` |
| **Kill-switch indicator** (header) | `state/trading.db` → `system_state` |

## How to run

```bash
# from the fx_trading/ root
pip install flask
python -m monitoring.dashboard          # port 5050 (default)
python -m monitoring.dashboard 8080     # custom port
```

Then open **http://localhost:5050** in your browser.  
The page auto-refreshes every **5 seconds**.

## File location rationale

The README lists `monitoring/` as the placeholder for *"alerting/heartbeat"* functionality.  
This dashboard is the natural resident — it reads the existing SQLite state store and JSONL  
log files without touching any execution code, keeping dependency rules intact.

## Dependencies

- `flask` — only additional package needed (`pip install flask`)  
- All data comes from files already written by the running system:  
  - `state/trading.db` (SQLite WAL)  
  - `logs/*.jsonl` (structured logging from `live_loop.py`)

## Log format expected

The dashboard looks for JSONL records with fields written by the live loop's `logger.info()` calls, e.g.:

```json
{"timestamp": "2024-01-15T10:32:00Z", "message": "signal_generated", "pair": "EURUSD", "signal": 1, "confidence": 0.73, "model_version": "v2024-01-10"}
{"timestamp": "2024-01-15T10:32:01Z", "message": "Order persisted, submitting to broker", "client_order_id": "...", "pair": "EURUSD"}
```

Any JSONL file in `logs/` is picked up automatically.
