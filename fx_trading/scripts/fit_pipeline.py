# scripts/fit_pipeline.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from features.pipeline import FeaturePipeline, FeaturePipelineConfig

SYMBOL  = "eurusd"
VERSION = "eurusd_5m_v001"

def main():
    in_path = f"data/{SYMBOL}_5m_bars.pkl"
    out_dir = f"artifacts/pipeline_{VERSION}"

    print(f"Loading bars...")
    with open(in_path, "rb") as f:
        df = pickle.load(f)

    print(f"  Loaded {len(df)} bars")

    # Fit only on the first 80% of data — never fit on test data
    train_end = int(len(df) * 0.80)
    df_train  = df.iloc[:train_end]
    print(f"  Fitting pipeline on first {len(df_train)} bars (80%)")

    config = FeaturePipelineConfig(
        version="1.0.0",
        rolling_window_bars=100,
        atr_period=14,
        rsi_period=14,
        min_variance_threshold=1e-8,
    )

    pipeline = FeaturePipeline(config)
    pipeline.fit(df_train)

    os.makedirs(out_dir, exist_ok=True)
    pipeline.save(out_dir)
    print(f"\nPipeline saved → {out_dir}")

    # Quick sanity check
    features = pipeline.transform(df.iloc[:200])
    print(f"Feature columns: {len(features.columns)}")
    print(f"Sample columns : {list(features.columns[:5])}...")
    print("\nDone.")

if __name__ == "__main__":
    main()