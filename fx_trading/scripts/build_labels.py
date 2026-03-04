# scripts/build_labels.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from research.labels import construct_labels, LabelConfig, get_label_distribution

SYMBOL = "eurusd"

def main():
    in_path  = f"data/{SYMBOL}_5m_bars.pkl"
    out_path = f"data/{SYMBOL}_5m_labels.pkl"

    print(f"Loading bars from {in_path}...")
    with open(in_path, "rb") as f:
        df = pickle.load(f)

    print(f"  Loaded {len(df)} bars")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")

    # Remove gap-filled bars before labelling
    if "source" in df.columns:
        df = df[df["source"] != "gap_filled"]
        print(f"  After removing gap-filled: {len(df)} bars")

    print("\nConstructing labels...")
    config = LabelConfig(
        tp_atr_multiplier=1.5,
        sl_atr_multiplier=1.0,
        max_holding_bars=12,
        min_atr_pips=3.0,
        max_spread_pips=5.0,
    )

    labels = construct_labels(df, config, pair_is_jpy=False)

    dist = get_label_distribution(labels)
    print("\nLabel distribution:")
    print(f"  LONG    : {dist.get('long_pct', '0%')}")
    print(f"  SHORT   : {dist.get('short_pct', '0%')}")
    print(f"  ABSTAIN : {dist.get('abstain_pct', '0%')}")
    print(f"  Balanced: {dist.get('is_balanced', False)}")
    print("\nHealthy: LONG 20-30%, SHORT 20-30%, ABSTAIN 40-60%")

    with open(out_path, "wb") as f:
        pickle.dump(labels, f)
    print(f"\nSaved → {out_path}")

if __name__ == "__main__":
    main()