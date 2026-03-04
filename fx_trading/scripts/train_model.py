# scripts/train_model.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from features.pipeline import FeaturePipeline
from research.walk_forward import generate_folds, WalkForwardConfig, slice_fold
from research.model_training import (
    ModelConfig, ValidationReport, train_model,
    predict_proba, compute_signal, save_model, seed_everything
)

SYMBOL       = "eurusd"
VERSION      = "eurusd_5m_v001"
PIPELINE_DIR = f"artifacts/pipeline_{VERSION}"


def evaluate_fold(proba, y_test):
    signals = []
    for i in range(len(proba)):
        sig, conf = compute_signal(proba[i], min_confidence=0.55, min_margin=0.10)
        signals.append((sig, conf))

    n_total   = len(signals)
    n_abstain = sum(1 for s, _ in signals if s == 0)
    correct   = 0
    traded    = 0
    for (sig, _), true in zip(signals, y_test.values):
        if sig != 0:
            traded += 1
            if sig == true:
                correct += 1

    hit_rate = correct / traded if traded > 0 else 0.0
    return {
        "n_bars":      n_total,
        "n_abstain":   n_abstain,
        "abstain_pct": round(n_abstain / n_total * 100, 1) if n_total > 0 else 0,
        "hit_rate":    round(hit_rate, 3),
        "n_traded":    traded,
    }


def main():
    seed_everything(42)

    print("Loading bars and labels...")
    with open(f"data/{SYMBOL}_5m_bars.pkl", "rb") as f:
        df = pickle.load(f)
    with open(f"data/{SYMBOL}_5m_labels.pkl", "rb") as f:
        labels = pickle.load(f)

    print(f"  Bars: {len(df)}, Labels: {len(labels)}")

    print("Transforming features...")
    pipeline = FeaturePipeline.load(PIPELINE_DIR)
    features = pipeline.transform(df)
    labels   = labels.reindex(features.index)
    print(f"  Feature matrix: {features.shape}")

    # ── Walk-forward config adjusted for 100k bars (16 months of data) ──
    wf_config = WalkForwardConfig(
    train_bars=300_000,   # ~4 years
    purge_bars=12,
    val_bars=75_000,      # ~1 year
    embargo_bars=12,
    test_bars=37_500,     # ~6 months
    step_bars=37_500,
    min_folds=3,
    )

    try:
        folds = generate_folds(len(features), wf_config)
    except ValueError as e:
        print(f"\nERROR: {e}")
        return

    print(f"\nGenerated {len(folds)} walk-forward folds")

    model_config = ModelConfig(
        version=VERSION,
        pair="EUR/USD",
        timeframe="5m",
        model_class="lightgbm",
        model_params={
            "n_estimators": 1000,
            "learning_rate": 0.02,
            "num_leaves": 127,
            "max_depth": 8,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
},
    )

    fold_results = []
    final_model  = None

    for fold in folds:
        print(f"\nFold {fold.fold_id}/{len(folds)}...")
        X_train, y_train, X_val, y_val, X_test, y_test = slice_fold(features, labels, fold)
        print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        if len(X_train) < 100 or len(X_val) < 10:
            print("  Skipping — not enough samples")
            continue

        model        = train_model(X_train, y_train, X_val, y_val, model_config)
        proba        = predict_proba(model, X_test, model_config.model_class)
        result       = evaluate_fold(proba, y_test)
        result["fold_id"] = fold.fold_id
        fold_results.append(result)
        final_model  = model

        print(f"  Hit rate: {result['hit_rate']:.1%}  |  "
              f"Abstain: {result['abstain_pct']:.0f}%  |  "
              f"Traded: {result['n_traded']} bars")

    if final_model is None:
        print("\nERROR: No folds completed.")
        return

    # Summary
    avg_hit     = np.mean([r["hit_rate"] for r in fold_results])
    avg_abstain = np.mean([r["abstain_pct"] for r in fold_results])

    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Folds completed : {len(fold_results)}")
    print(f"Avg hit rate    : {avg_hit:.1%}  (good if > 52%)")
    print(f"Avg abstain     : {avg_abstain:.0f}%  (good if 40-70%)")
    print()

    report = ValidationReport(
        fold_results=fold_results,
        aggregate={
            "avg_hit_rate":    avg_hit,
            "avg_abstain_pct": avg_abstain,
            "n_folds":         len(fold_results),
        },
        deployment_approved=False,
    )

    model_config.train_end_date = str(features.index[-1].date())
    artifact_dir, sha256 = save_model(
        final_model, model_config, report, base_path="artifacts/"
    )

    print(f"Model saved to : {artifact_dir}")
    print(f"SHA256         : {sha256}")
    print()
    print("NEXT: Open these two files and change deployment_approved to true")
    print(f"  {artifact_dir}\\validation_report.json")
    print(f"  {artifact_dir}\\model.manifest.json")

if __name__ == "__main__":
    main()