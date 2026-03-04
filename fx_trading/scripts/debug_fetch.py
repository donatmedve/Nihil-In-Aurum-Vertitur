import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MetaTrader5 as mt5
import time
from datetime import datetime

mt5.initialize()
print(f"Connected: {mt5.account_info().server}")

# Enable symbol
mt5.symbol_select("EURUSD", True)
time.sleep(3)

# Check symbol info
info = mt5.symbol_info("EURUSD")
print(f"Symbol visible: {info.visible}")
print(f"Symbol select:  {info.select}")

# Try fetch and print the actual error
rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M5, 0, 10)
print(f"Rates result: {rates}")
print(f"Last error:   {mt5.last_error()}")

# Try a different approach - copy from specific date
rates2 = mt5.copy_rates_from("EURUSD", mt5.TIMEFRAME_M5, datetime(2024, 12, 1), 10)
print(f"Rates2 result: {rates2}")
print(f"Last error:    {mt5.last_error()}")

mt5.shutdown()
