import pickle
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.pipeline import FeaturePipeline

with open("data/eurusd_5m_bars.pkl", "rb") as f:
    df = pickle.load(f)
with open("data/eurusd_5m_labels.pkl", "rb") as f:
    labels = pickle.load(f)

pipeline = FeaturePipeline.load("artifacts/pipeline_eurusd_5m_v001")
features = pipeline.transform(df)

print("=== DATA QUALITY ===")
print(f"Bars         : {len(df):,}")
print(f"Labels       : {len(labels):,}")
print(f"Features     : {features.shape}")

print(f"\n=== NaN CHECK ===")
nan_pct = features.isnull().mean() * 100
print(f"Rows with any NaN : {features.isnull().any(axis=1).mean()*100:.1f}%")
for col in nan_pct[nan_pct > 5].index:
    print(f"  {col}: {nan_pct[col]:.1f}% NaN")

print(f"\n=== LABEL DISTRIBUTION ===")
valid = labels.dropna()
print(f"Valid labels : {len(valid):,}")
print(f"LONG  (+1)   : {(valid==1).mean()*100:.1f}%")
print(f"SHORT (-1)   : {(valid==-1).mean()*100:.1f}%")
print(f"ABSTAIN (0)  : {(valid==0).mean()*100:.1f}%")

print(f"\n=== FEATURE VARIANCE (bottom 5) ===")
variances = features.var().sort_values()
print(variances.head(5))

print(f"\n=== LABEL/FEATURE INDEX OVERLAP ===")
overlap = features.index.intersection(labels.dropna().index)
print(f"Overlapping rows: {len(overlap):,}")

