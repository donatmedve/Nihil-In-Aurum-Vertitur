"""
monitoring/dashboard.py
=======================
Pure JSON API backend for the FX trading dashboard.
The frontend (monitoring/static/) is hosted separately on GitHub Pages.

Run:
    cd fx_trading/
    pip install flask flask-cors
    python -m monitoring.dashboard

Endpoints:
    GET /api/status     — kill switch, HWM, heartbeat, equity, drawdown
    GET /api/positions  — all positions (filter to OPEN on frontend)
    GET /api/orders     — last 100 orders
    GET /api/pnl        — daily P&L last 30 days
    GET /api/signals    — signal_generated events from JSONL logs
    GET /api/decisions  — risk/block events from JSONL logs
    GET /api/logs       — last 200 raw log events
    GET /health         — simple health check
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "state" / "trading.db"
LOGS_DIR = BASE_DIR / "logs"

app = Flask(__name__)
CORS(app)   # allow GitHub Pages (and localhost) to call this API


def _db():
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _sys_value(conn, key):
    try:
        row = conn.execute(
            "SELECT value FROM system_state WHERE key = ? LIMIT 1", (key,)
        ).fetchone()
        return row["value"] if row else None
    except Exception:
        return None


def _read_jsonl(filename, last_n=300):
    path = LOGS_DIR / filename
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    out = []
    for line in lines[-last_n:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def _all_events(last_n=500):
    if not LOGS_DIR.exists():
        return []
    events = []
    for f in LOGS_DIR.glob("*.jsonl"):
        events.extend(_read_jsonl(f.name, last_n))
    events.sort(key=lambda e: e.get("timestamp", e.get("time", "")), reverse=True)
    return events[:last_n]


@app.route("/health")
def health():
    return jsonify({"status": "ok", "db": DB_PATH.exists()})


@app.route("/api/status")
def api_status():
    conn = _db()
    if conn is None:
        return jsonify({"db_available": False})
    try:
        ks  = _sys_value(conn, "kill_switch")
        hwm = _sys_value(conn, "high_water_mark")
        hb  = _sys_value(conn, "last_heartbeat_utc")

        equity = None
        try:
            row = conn.execute(
                "SELECT equity_usd FROM account_snapshots ORDER BY recorded_at DESC LIMIT 1"
            ).fetchone()
            if row:
                equity = float(row["equity_usd"])
        except Exception:
            pass

        drawdown = None
        if hwm and equity:
            hwm_f = float(hwm)
            if hwm_f > 0:
                drawdown = max(0.0, (hwm_f - equity) / hwm_f)

        return jsonify({
            "db_available":    True,
            "kill_switch":     bool(int(ks)) if ks is not None else False,
            "high_water_mark": float(hwm) if hwm else None,
            "last_heartbeat":  hb,
            "equity_usd":      equity,
            "drawdown":        drawdown,
            "server_time_utc": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        return jsonify({"db_available": True, "error": str(e)})
    finally:
        conn.close()


@app.route("/api/positions")
def api_positions():
    conn = _db()
    if conn is None:
        return jsonify([])
    try:
        rows = conn.execute(
            """SELECT pair, side, units, entry_price, stop_loss_price,
                      opened_at_utc, closed_at_utc, status,
                      client_order_id, pnl_usd
               FROM positions ORDER BY opened_at_utc DESC LIMIT 100"""
        ).fetchall()
        return jsonify([dict(r) for r in rows])
    except Exception:
        return jsonify([])
    finally:
        conn.close()


@app.route("/api/orders")
def api_orders():
    conn = _db()
    if conn is None:
        return jsonify([])
    try:
        rows = conn.execute(
            """SELECT client_order_id, broker_order_id, pair, side, order_type,
                      units, stop_loss_price, filled_price, status,
                      signal_confidence, model_version,
                      submitted_at, filled_at, rejection_reason, pnl_usd
               FROM orders ORDER BY submitted_at DESC LIMIT 100"""
        ).fetchall()
        return jsonify([dict(r) for r in rows])
    except Exception:
        return jsonify([])
    finally:
        conn.close()


@app.route("/api/pnl")
def api_pnl():
    conn = _db()
    if conn is None:
        return jsonify([])
    try:
        rows = conn.execute(
            "SELECT date, pnl_usd FROM daily_pnl ORDER BY date DESC LIMIT 30"
        ).fetchall()
        return jsonify([dict(r) for r in rows])
    except Exception:
        return jsonify([])
    finally:
        conn.close()


@app.route("/api/signals")
def api_signals():
    events = _all_events(500)
    return jsonify([
        e for e in events
        if "signal_generated" in (e.get("message", "") + e.get("event", ""))
    ][:100])


@app.route("/api/decisions")
def api_decisions():
    events = _all_events(500)
    kw = {"risk_decision","trade_blocked","trade_permitted","Order persisted",
          "order_persisted","circuit_breaker","drawdown_check","kill_switch","ABSTAIN"}
    return jsonify([
        e for e in events
        if any(k in str(e.get("message","")) + str(e.get("event","")) for k in kw)
    ][:100])


@app.route("/api/logs")
def api_logs():
    return jsonify(_all_events(200))


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5050
    print(f"\n  FX Trading API  →  http://0.0.0.0:{port}")
    print(f"  DB              →  {DB_PATH}")
    print(f"  Logs            →  {LOGS_DIR}")
    print(f"  CORS: open — all origins\n")
    app.run(host="0.0.0.0", port=port, debug=False)
