#!/usr/bin/env python3
# start.py
# ============================================================
# COPY-PASTE READY STARTUP SCRIPT
#
# Fill in the five configuration values below, then run:
#   python start.py
#
# Prerequisites:
#   1. All 48 tests pass:   python -m unittest tests.test_all -v
#   2. A trained, approved model artifact exists in artifacts/
#   3. MetaTrader 5 terminal is running and logged in on Windows
#   4. Your demo forward test has passed (see README Section 10)
# ============================================================

import sys
import logging
import structlog
from pathlib import Path

# ── CONFIGURATION — fill these in ──────────────────────────────────────────
MT5_LOGIN    = 5047402269          # Your MT5 account number
MT5_PASSWORD = "T@Nj3aYk"  # Your MT5 password
MT5_SERVER   = "MetaQuotes-Demo"  # Your broker's MT5 server name (find in MT5 terminal)
BROKER_TZ    = "Etc/GMT-2"      # Your broker's server timezone — CHECK YOUR BROKER DOCS
                                  # Common values: "Etc/GMT-2", "Etc/GMT-3", "US/Eastern"

MODEL_ARTIFACT_DIR  = "artifacts/eurusd_5m_v001_20260305"   # path to your trained model
PIPELINE_ARTIFACT_DIR = "artifacts/pipeline_eurusd_5m_v001"  # path to your saved pipeline
STATE_DB_PATH = "state/trading.db"
LOG_DIR       = "logs"
# ──────────────────────────────────────────────────────────────────────────

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def setup_logging():
    Path(LOG_DIR).mkdir(exist_ok=True)

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    # Also write to file
    file_handler = logging.FileHandler(f"{LOG_DIR}/live.jsonl")
    file_handler.setLevel(logging.DEBUG)
    logging.basicConfig(handlers=[file_handler, logging.StreamHandler()], level=logging.INFO)
    return logging.getLogger(__name__)


def main():
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("FX Trading System — Starting up")
    logger.info("=" * 60)

    # ── 1. Run tests first ────────────────────────────────────────
    logger.info("Running test suite...")
    import unittest
    loader = unittest.TestLoader()
    suite  = loader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=0, stream=open("/dev/null", "w"))
    result = runner.run(suite)

    if not result.wasSuccessful():
        logger.critical(
            f"TEST SUITE FAILED: {len(result.failures)} failures, "
            f"{len(result.errors)} errors. DO NOT PROCEED."
        )
        print("\n" + "="*60)
        print("TESTS FAILED — run 'python -m unittest tests.test_all -v' to see details")
        print("="*60)
        sys.exit(1)

    logger.info(f"All {result.testsRun} tests passed.")

    # ── 2. Initialize state store ─────────────────────────────────
    from state.store import StateStore
    store = StateStore(STATE_DB_PATH)
    logger.info(f"State store initialized: {STATE_DB_PATH}")

    # ── 3. Connect to MT5 ─────────────────────────────────────────
    from broker.mt5_adapter import MT5Adapter
    adapter = MT5Adapter(
        broker_tz_str=BROKER_TZ,
        login=MT5_LOGIN,
        password=MT5_PASSWORD,
        server=MT5_SERVER,
    )

    if not adapter.is_connected():
        logger.critical(
            "Cannot connect to MT5. Is the terminal running and logged in?"
        )
        sys.exit(1)

    account = adapter.get_account_state()
    logger.info(
        f"Connected to MT5 — Account: {MT5_LOGIN} | "
        f"Equity: ${float(account.equity_usd):,.2f} | "
        f"Balance: ${float(account.balance_usd):,.2f}"
    )

    # Safety check: warn loudly if this looks like a live account on first run
    if float(account.equity_usd) > 10_000 and not Path(STATE_DB_PATH).exists():
        print("\n" + "!"*60)
        print("WARNING: Large account equity detected on first run.")
        print("Are you sure this is a DEMO account?")
        print("Type 'YES' to continue, anything else to abort:")
        confirm = input("> ").strip()
        if confirm != "YES":
            print("Aborted.")
            sys.exit(0)

    # ── 4. Load model + pipeline ──────────────────────────────────
    from research.model_training import load_model
    from features.pipeline import FeaturePipeline

    logger.info(f"Loading model from {MODEL_ARTIFACT_DIR}...")
    model, manifest = load_model(MODEL_ARTIFACT_DIR, require_approved=True)
    logger.info(
        f"Model loaded: version={manifest['version']} | "
        f"pair={manifest['pair']} | "
        f"sha256={manifest['sha256'][:12]}..."
    )

    logger.info(f"Loading pipeline from {PIPELINE_ARTIFACT_DIR}...")
    pipeline = FeaturePipeline.load(PIPELINE_ARTIFACT_DIR)
    logger.info("Pipeline loaded.")

    # Get pipeline SHA256 for reconciliation
    import hashlib, pickle, tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        tmp_path = tmp.name
    pipeline.save(tmp_path.replace(".pkl", ""))
    pipeline_sha256 = hashlib.sha256(Path(tmp_path).read_bytes()).hexdigest()
    os.unlink(tmp_path)

    # ── 5. Set up broker adapter idempotent submitter ─────────────
    from broker.adapter import IdempotentOrderSubmitter
    submitter = IdempotentOrderSubmitter(adapter=adapter, state_store=store)

    # ── 6. Set up risk engine ─────────────────────────────────────
    from risk.engine import RiskEngine, RiskParameters
    risk_params = RiskParameters()   # uses safe defaults — tune after demo period
    risk_engine = RiskEngine(params=risk_params, state_store=store)

    logger.info(
        f"Risk parameters: "
        f"risk_per_trade={risk_params.risk_per_trade_pct:.1%} | "
        f"max_daily_drawdown={risk_params.max_daily_drawdown_pct:.1%} | "
        f"max_positions={risk_params.max_open_positions} | "
        f"tp_atr_mult={risk_params.tp_atr_multiplier} | "
        f"sl_atr_mult={risk_params.sl_atr_multiplier}"
    )

    # ── 7. Determine trading pairs from model manifest ────────────
    from shared.schemas import Pair as PairEnum
    model_pair_str = manifest.get("pair", "EUR/USD")
    try:
        primary_pair = PairEnum(model_pair_str)
    except ValueError:
        logger.warning(f"Unknown pair in manifest: {model_pair_str}. Defaulting to EUR/USD.")
        primary_pair = PairEnum.EURUSD

    # Trade the pair the model was trained on.
    # To add more pairs, train separate models and instantiate separate loops,
    # OR add pairs here if your model is multi-pair (ensure pipeline handles all pairs).
    pairs = [primary_pair]
    logger.info(f"Trading pairs: {[p.value for p in pairs]}")

    # ── 8. Start live loop ────────────────────────────────────────
    from execution.live_loop import LiveExecutionLoop

    loop = LiveExecutionLoop(
        adapter=adapter,
        submitter=submitter,
        pipeline=pipeline,
        model=model,
        model_sha256=manifest["sha256"],
        pipeline_sha256=pipeline_sha256,
        model_version=manifest["version"],
        model_class=manifest["model_class"],  # FIX: pass model_class from manifest
        risk_engine=risk_engine,
        state_store=store,
        pairs=pairs,
        broker_tz_str=BROKER_TZ,
        timeframe_sec=300,      # 5-minute bars
        lookback_bars=110,      # rolling_window_bars + 10 bar buffer
    )

    logger.info("Handing off to execution loop. Press Ctrl+C to stop cleanly.")
    loop.run(
        model_sha256=manifest["sha256"],
        pipeline_sha256=pipeline_sha256,
    )

    logger.info("Execution loop exited cleanly.")

    logger.info("Handing off to execution loop. Press Ctrl+C to stop cleanly.")
    loop.run(
        model_sha256=manifest["sha256"],
        pipeline_sha256=pipeline_sha256,
    )

    logger.info("Execution loop exited cleanly.")


if __name__ == "__main__":
    main()
