# scripts/check_symbols.py
# Run this to see exactly what symbols are available on your MT5 account.

import MetaTrader5 as mt5
from datetime import datetime

if not mt5.initialize():
    print(f"ERROR: MT5 failed to initialize: {mt5.last_error()}")
    exit()

print(f"Connected to: {mt5.account_info().server}")
print(f"Account: {mt5.account_info().login}\n")

# Show all symbols that contain EUR or GBP
print("=== Symbols containing EUR ===")
symbols = mt5.symbols_get()
for s in symbols:
    if "EUR" in s.name:
        print(f"  {s.name}")

print("\n=== Symbols containing GBP ===")
for s in symbols:
    if "GBP" in s.name:
        print(f"  {s.name}")

# Also try a quick data fetch test
print("\n=== Quick data test ===")
for name in ["EURUSD", "EURUSDm", "EURUSD.", "EUR/USD"]:
    rates = mt5.copy_rates_from(name, mt5.TIMEFRAME_M5, datetime(2024,1,1), 5)
    if rates is not None and len(rates) > 0:
        print(f"  '{name}' WORKS ✓ — got {len(rates)} bars")
    else:
        print(f"  '{name}' failed")

mt5.shutdown()

'''
**Run it:**

python scripts/check_symbols.py'''