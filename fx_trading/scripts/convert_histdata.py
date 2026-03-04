# scripts/convert_histdata.py
# Loads Histdata.com 1-minute CSVs, resamples to 5m,
# and saves a pickle in the exact format the pipeline expects.

import os
import sys
import glob
import pickle
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_DIR  = "data/histdata_raw"
OUT_PATH = "data/eurusd_5m_bars.pkl"

# Histdata has one price stream — we simulate bid/ask with a fixed spread.
# 1.5 pip spread is realistic for EUR/USD during liquid hours.
SPREAD_PIPS = 1.5
PIP_SIZE     = 0.0001

def load_histdata_csvs(raw_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    print(f"Found {len(files)} CSV files")
    chunks = []
    for f in files:
        print(f"  Loading {os.path.basename(f)}...")
        df = pd.read_csv(
            f,
            sep=";",
            header=None,
            names=["datetime", "open", "high", "low", "close", "volume"],
            dtype={"datetime": str}
        )
        chunks.append(df)

    combined = pd.concat(chunks, ignore_index=True)
    print(f"Total 1m rows loaded: {len(combined):,}")
    return combined


def parse_and_resample(df: pd.DataFrame) -> pd.DataFrame:
    # Parse datetime — format is "20130102 170100"
    df["utc_open"] = pd.to_datetime(df["datetime"], format="%Y%m%d %H%M%S", utc=True)
    df = df.set_index("utc_open").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    # Remove weekends (Histdata sometimes includes Sunday open / Friday late)
    df = df[df.index.dayofweek < 5]

    print(f"After weekend filter: {len(df):,} 1m bars")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Resample 1m → 5m
    df_5m = df.resample("5min").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["open"])

    print(f"After resampling to 5m: {len(df_5m):,} bars")
    return df_5m


def build_pipeline_df(df_5m: pd.DataFrame) -> pd.DataFrame:
    # Simulate bid/ask from mid price using fixed spread
    half_spread = (SPREAD_PIPS * PIP_SIZE) / 2.0

    out = pd.DataFrame(index=df_5m.index)

    out["open_bid"]  = df_5m["open"]  - half_spread
    out["high_bid"]  = df_5m["high"]  - half_spread
    out["low_bid"]   = df_5m["low"]   - half_spread
    out["close_bid"] = df_5m["close"] - half_spread

    out["open_ask"]  = df_5m["open"]  + half_spread
    out["high_ask"]  = df_5m["high"]  + half_spread
    out["low_ask"]   = df_5m["low"]   + half_spread
    out["close_ask"] = df_5m["close"] + half_spread

    out["volume"]     = df_5m["volume"].astype(int)
    out["is_complete"] = True
    out["source"]     = "broker_historical"
    out["pair"]       = "EUR/USD"
    out["utc_close"]  = out.index + pd.Timedelta(minutes=5)
    out["timeframe_sec"] = 300

    return out


def main():
    print("Loading Histdata CSVs...")
    raw = load_histdata_csvs(RAW_DIR)

    print("\nResampling to 5m...")
    df_5m = parse_and_resample(raw)

    print("\nBuilding pipeline DataFrame...")
    df_out = build_pipeline_df(df_5m)

    print(f"\nFinal dataset:")
    print(f"  Rows      : {len(df_out):,}")
    print(f"  Date range: {df_out.index[0]} to {df_out.index[-1]}")
    print(f"  Columns   : {list(df_out.columns)}")

    with open(OUT_PATH, "wb") as f:
        pickle.dump(df_out, f)

    print(f"\nSaved → {OUT_PATH}")
    print("Done. Now re-run Steps 5, 6, 7, 8, 9 as normal.")


if __name__ == "__main__":
    main()
